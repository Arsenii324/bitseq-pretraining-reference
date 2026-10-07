"""Masked-diffusion primitives: corruption, likelihood surrogates, exact likelihood, samplers.

Notation used throughout (matches TraFL / LLaDA):

  x        prompt tokens, length P, never masked
  y        completion tokens, length L
  A        set of masked completion positions, |A| = l
  z_A      the sequence with positions in A replaced by MASK
  S(A)     = sum_{i in A} log p_theta(y_i | z_A, x)          (sum over the masked positions)
  m(l)     = E_{|A|=l}[ S(A) / l ]                            (mean per masked position)

Three quantities that this module keeps rigorously distinct:

  ELBO(y)          = sum_{l=1..L} m(l)          -- LLaDA's 1/t NELBO (extensive, ~ L * per-token)
  surrogate(y)     = (1/L) * ELBO(y)            -- TraFL Eq. (4), the 1/l per-position mean
  logp_exact(y)    = log p_theta(y | x)         -- exact marginal under a *stated* sampler

`ELBO(y) = L * surrogate(y)` is claim A1 and is asserted here as an exact identity, not an
approximation.  `tests/test_elbo_identity.py` checks it by full enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import combinations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
# building corrupted inputs
# --------------------------------------------------------------------------------------------

def build_z(prompt: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, mask_id: int) -> torch.Tensor:
    """prompt (B,P), y (B,L), mask (B,L) bool -> z (B,P+L) with masked completion positions."""
    ycor = torch.where(mask, torch.full_like(y, mask_id), y)
    return torch.cat([prompt, ycor], dim=1)


def masked_logprobs(model, prompt, y, mask):
    """Return (B,L) tensor of log p_theta(y_i | z_A, x) at every completion position.

    Values at unmasked positions are computed too (cheap) but callers must ignore them.
    """
    P = prompt.shape[1]
    z = build_z(prompt, y, mask, model.mask_id)
    logits = model(z)[:, P:, :]                       # (B,L,V-1)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)   # (B,L)


# --------------------------------------------------------------------------------------------
# exact quantities by enumeration (small L only)
# --------------------------------------------------------------------------------------------

def _all_subsets_mask(L: int, device) -> torch.Tensor:
    """(2^L, L) bool matrix of every subset of positions, as a mask."""
    idx = torch.arange(2 ** L, device=device)
    bits = (idx[:, None] >> torch.arange(L, device=device)[None, :]) & 1
    return bits.bool()


@torch.no_grad()
def exact_elbo_and_surrogate(model, prompt, y, chunk: int = 512):
    """Exact ELBO(y) and surrogate(y) for one (prompt, y) pair, by enumerating all 2^L masks.

    Returns dict with:
      elbo_from_l_uniform : sum_l m(l)                       (the 1/l route, times L)
      elbo_from_t_integral: int_0^1 (1/t) E_{A~Bern(t)}[S(A)] dt   (the LLaDA 1/t route, exact)
      surrogate           : E_{l~U{1..L}} m(l)   = elbo / L
      m_of_l              : tensor (L,) of m(l)

    The two ELBO routes are computed by *independent* formulas and must agree exactly (A1).
    """
    device = prompt.device
    L = y.shape[1]
    assert prompt.shape[0] == 1 and y.shape[0] == 1
    masks = _all_subsets_mask(L, device)                       # (2^L, L)
    sizes = masks.sum(dim=1)                                   # (2^L,)
    S = torch.zeros(masks.shape[0], dtype=torch.float64, device=device)
    for s in range(0, masks.shape[0], chunk):
        mb = masks[s : s + chunk]
        pb = prompt.expand(mb.shape[0], -1)
        yb = y.expand(mb.shape[0], -1)
        lp = masked_logprobs(model, pb, yb, mb).double()       # (b,L)
        S[s : s + chunk] = (lp * mb.double()).sum(dim=1)

    # route 1: group by l, m(l) = mean over subsets of size l of S(A)/l
    m_of_l = torch.zeros(L, dtype=torch.float64, device=device)
    for l in range(1, L + 1):
        sel = sizes == l
        m_of_l[l - 1] = (S[sel] / l).mean()
    elbo_l = m_of_l.sum()

    # route 2: int_0^1 t^{|A|-1} (1-t)^{L-|A|} dt = Beta(|A|, L-|A|+1) = (|A|-1)!(L-|A|)!/L!
    #          so ELBO = sum_A S(A) * Beta(|A|, L-|A|+1)
    lgamma = torch.lgamma
    a = sizes.double()
    logw = lgamma(a) + lgamma(L - a + 1.0) - lgamma(torch.tensor(L + 1.0, dtype=torch.float64, device=device))
    w = torch.exp(logw)
    w = torch.where(sizes == 0, torch.zeros_like(w), w)
    elbo_t = (S * w).sum()

    return {
        "elbo_from_l_uniform": elbo_l.item(),
        "elbo_from_t_integral": elbo_t.item(),
        "surrogate": (elbo_l / L).item(),
        "m_of_l": m_of_l.cpu(),
    }


@torch.no_grad()
def exact_logp_uniform_order(model, prompt, y, chunk: int = 512):
    r"""Exact log p_theta(y | x) under the *uniform random order, one token per step* sampler.

    p(y) = (1/L!) sum_{sigma in S_L} prod_k p_theta(y_{sigma_k} | z_{S_{k-1}})

    Computed by a subset DP in 2^L states rather than L! paths:
        g(emptyset) = 1
        g(S) = sum_{i in S} g(S \ {i}) * p_theta(y_i | z_{complement of S\{i} is revealed})
    where the model input for state S\{i} has the positions *not yet revealed* masked, i.e.
    mask = complement of (S \ {i}) ... careful: S here is the set of *revealed* positions.

    Returns log p(y|x) as a float (natural log).
    """
    device = prompt.device
    L = y.shape[1]
    N = 2 ** L
    revealed = _all_subsets_mask(L, device)            # bit j set == position j revealed
    mask_in = ~revealed                                 # what the model sees masked
    # log p_theta(y_i | state) for every state and every position
    lp = torch.zeros(N, L, dtype=torch.float64, device=device)
    for s in range(0, N, chunk):
        mb = mask_in[s : s + chunk]
        pb = prompt.expand(mb.shape[0], -1)
        yb = y.expand(mb.shape[0], -1)
        lp[s : s + chunk] = masked_logprobs(model, pb, yb, mb).double()

    NEG = torch.tensor(float("-inf"), dtype=torch.float64, device=device)
    logg = torch.full((N,), float("-inf"), dtype=torch.float64, device=device)
    logg[0] = 0.0
    order = torch.argsort(revealed.sum(dim=1))
    for state in order.tolist():
        if state == 0:
            continue
        terms = []
        for i in range(L):
            bit = 1 << i
            if state & bit:
                prev = state ^ bit
                if logg[prev] > float("-inf"):
                    terms.append(logg[prev] + lp[prev, i])
        logg[state] = torch.logsumexp(torch.stack(terms), dim=0) if terms else NEG
    import math

    return (logg[N - 1] - math.lgamma(L + 1)).item()


# --------------------------------------------------------------------------------------------
# Monte-Carlo surrogates (what a real implementation must use)
# --------------------------------------------------------------------------------------------
#
# Every scheme below returns a list of (mask, weight) replicates.  The estimator is
#     score = mean_over_replicates[ weight * S(A) ],    S(A) = sum_{i in A} log p(y_i | z_A)
# and `scale="intensive"` targets TraFL's surrogate (per-position mean, Eq. 4) while
# `scale="extensive"` targets the LLaDA ELBO.  The two differ by exactly a factor L (claim A1).
#
# Bias status of each scheme w.r.t. its target, derived and then checked in
# tests/test_elbo_identity.py:
#   uniform_l      unbiased
#   bernoulli_t    unbiased
#   antithetic_t   unbiased  (complement of Bern(t) is Bern(1-t) and 1-t ~ U(0,1))
#   antithetic_l   BIASED    (pairing l with L-l forces l ~ U{1..L-1}: the fully-masked
#                             level l=L, the most negative m(l), is never drawn)
#   stratified_l   unbiased  stratifies l across the K replicates (ours; see E56)
#   systematic_l   unbiased  as above, one shared offset -- unbiased but weaker (E56)
#   antithetic_debiased
#                  unbiased  TraFL's pairing + the exact deterministic m(L) term (ours; E56)
#   antithetic_size_only
#                  BIASED    same bias as antithetic_l by construction; couples mask SIZES but
#                            draws token sets independently -- the E78c discriminator
#   adaptive_l     unbiased  importance sampling with a mean-based q(l) proportional to
#                            |m(l)| (exploratory; not variance-optimal with within-level noise)
#   antithetic_strat_debiased
#                  unbiased  the above with l stratified across complete pairs -- the
#                            paired-plus-stratified candidate (ours; E56f). Needs odd K;
#                            E64 found it tied with the simpler debiased pairing end to end.
#   coupled_espo   unbiased  ESPO's published construction (arXiv:2512.03759 App. D): draw
#                            l ~ U{0..L}, weight (L+1)/l, and average with the complementary
#                            mask.  Drawing l from {0..L} rather than {1..L} is exactly what
#                            makes the complementary pair unbiased, since then L-l has the same
#                            distribution as l.
#
# TraFL says only "4 antithetic replicates, i.e., complementary mask pairs" and (Eq. 4) draws
# l ~ U{1..L} with weight 1/l.  Taken literally that is `antithetic_l`, the biased one; ESPO,
# the neighbouring method it compares against, publishes the unbiased construction.  TraFL's
# text does not say which of the two it implements, so the honest statement is that the literal
# reading is biased by a computable amount and the paper does not disambiguate.

SCHEMES = ("uniform_l", "bernoulli_t", "antithetic_l", "antithetic_t", "coupled_espo",
           "stratified_l", "systematic_l", "antithetic_debiased",
           "antithetic_strat_debiased", "adaptive_l", "antithetic_size_only",
           "antithetic_break")


@dataclass
class SurrogateCfg:
    """How to estimate the completion score.  Defaults are the closest literal reading of §4.1."""
    n_replicates: int = 4              # K
    scheme: str = "antithetic_l"
    scale: str = "intensive"           # "intensive" = TraFL Eq.4 (1/l) | "extensive" = ELBO (1/t)
    pair_break: float = 0.0            # only for `antithetic_break`: fraction of positions whose
                                       # complementarity is broken (0 = exact TraFL pairing)

    def __post_init__(self):
        assert self.scheme in SCHEMES, self.scheme
        assert self.scale in ("intensive", "extensive")
        assert 0.0 <= self.pair_break <= 1.0, self.pair_break


def _ranks(B, L, gen, device, valid=None):
    r = torch.rand(B, L, generator=gen, device=device)
    if valid is not None:
        r = r.masked_fill(~valid, 2.0)              # invalid positions sort last
    return r.argsort(dim=1).argsort(dim=1)


def draw_masks(L: int, B: int, cfg: SurrogateCfg, gen: torch.Generator, device,
               valid: torch.Tensor | None = None, level_probs: torch.Tensor | None = None):
    """Return exactly cfg.n_replicates (mask (B,L) bool, weight (B,) float) pairs.

    `valid` (B,L) restricts masking to a per-row subset of positions -- TraFL masks "only
    non-EOS positions", so with variable-length completions the effective length is per row and
    every `L` below becomes that row's own count.  With `valid=None` this is the fixed-length
    case and `n_valid == L` for every row.
    """
    nv = (valid.sum(1) if valid is not None else
          torch.full((B,), L, device=device, dtype=torch.long)).clamp(min=1)
    nvf = nv.float()
    scale_div = nvf if cfg.scale == "intensive" else torch.ones_like(nvf)
    out = []
    K = cfg.n_replicates
    if cfg.scheme == "antithetic_break":
        # E84: the causal test for E83. E83 shows frontier quality tracks how much the per-token
        # scoring count varies across a replicate-set, and that complementary pairing drives that
        # variance to (nearly) zero -- but that is a correlation over five schemes. This turns it
        # into a dial on one scheme.
        #
        # Draw the pair exactly as TraFL does, then break complementarity on a `pair_break` fraction
        # of positions: for those, the partner's membership is resampled independently instead of
        # being `1 - A_i`. pair_break=0 IS TraFL's estimator; larger values leave more randomness in
        # which tokens get scored and change nothing else about the draw.
        #
        # Weights use the REALISED mask size rather than `l`, because breaking the pair changes the
        # partner's size -- so the estimator stays an n/|mask| reweighting at every setting of the
        # dial and the dial cannot smuggle in a weighting change.
        rho = float(cfg.pair_break)
        hi = (nv - 1).clamp(min=1)
        while len(out) < K:
            l = (torch.rand(B, generator=gen, device=device) * hi.float()).long().clamp(max=hi - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            out.append((mask, (nvf / mask.sum(1).clamp(min=1).float()) / scale_div))
            if len(out) < K:
                comp = ~mask if valid is None else ((~mask) & valid)
                if rho > 0.0:
                    q = ((nvf - l.float()) / nvf).clamp(0.0, 1.0)[:, None]
                    indep = torch.rand(B, L, generator=gen, device=device) < q
                    swap = torch.rand(B, L, generator=gen, device=device) < rho
                    comp = torch.where(swap, indep, comp)
                    if valid is not None:
                        comp = comp & valid
                out.append((comp, (nvf / comp.sum(1).clamp(min=1).float()) / scale_div))
        return out[:K]
    if cfg.scheme == "antithetic_size_only":
        # E78c's discriminating estimator. TraFL's `antithetic_l` couples two things at once:
        # the mask SIZES (l and n-l) and the token SETS (one is the complement of the other, so the
        # pair covers every token exactly once). E78 shows pairing schemes have a worse
        # quality/coverage frontier than non-pairing ones at matched coverage, and that neither
        # variance magnitude nor negative correlation explains the split.
        #
        # This scheme keeps the size coupling and DROPS the token coupling: `l` and `n-l` exactly as
        # in `antithetic_l`, but the second mask's positions are drawn independently rather than as
        # the complement of the first. Comparing the three tells you which coupling matters:
        #   antithetic_l        sizes coupled, tokens coupled     -> bad frontier (E78)
        #   antithetic_size_only sizes coupled, tokens independent -> ?
        #   uniform_l           neither coupled                   -> good frontier (E76)
        # Same weights as `antithetic_l`, so it is biased in exactly the same way and by the same
        # amount -- which also holds bias fixed across the comparison.
        hi = (nv - 1).clamp(min=1)
        while len(out) < K:
            l = (torch.rand(B, generator=gen, device=device) * hi.float()).long().clamp(max=hi - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            out.append((mask, (nvf / l.float()) / scale_div))
            if len(out) < K:
                lc = (nv - l).clamp(min=1)
                mask2 = _ranks(B, L, gen, device, valid) < lc[:, None]   # independent positions
                if valid is not None:
                    mask2 = mask2 & valid
                out.append((mask2, (nvf / lc.float()) / scale_div))
        return out[:K]
    if cfg.scheme == "adaptive_l":
        # Importance sampling over the mask level with a mean-based proposal.
        # The target is  ELBO = sum_{l=1..n} m(l),  and every scheme above draws l uniformly and
        # reweights.  q(l) ∝ |m(l)| is optimal only when the term at fixed l is deterministic.
        # With random masks, the exact optimum is q(l) ∝ sqrt(E[X_l^2]); see E69.  The mean-based
        # heuristic remains unbiased and is useful as a diagnostic baseline.
        #
        #     draw l ~ q,  estimator  m_hat(l) / q(l)  with  m_hat(l) = (1/l) * sum_{i in M} log p_i
        #     E_q[ m_hat/q ] = sum_l q(l) * m(l)/q(l) = sum_l m(l)          -> unbiased for any q>0
        #
        # so the weight is 1/(l*q(l)); with q uniform this collapses to the n/l used everywhere
        # above, which is the sanity check gated in T9.
        #
        # `level_probs` is a (n,) tensor over l = 1..n.  Fixed-length only: with per-row n the
        # proposal would have to be per-row too, and nothing here needs that yet.
        if level_probs is None:
            raise ValueError("adaptive_l needs level_probs (see mean_based_level_probs)")
        if valid is not None:
            raise ValueError("adaptive_l is implemented for fixed-length completions only")
        q = level_probs.to(device=device, dtype=torch.float32)
        q = q / q.sum()
        for _ in range(K):
            idx = torch.multinomial(q.expand(B, -1), 1, replacement=True, generator=gen).squeeze(1)
            l = idx + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            w = 1.0 / (l.float() * q[idx])
            out.append((mask, w / scale_div))
        return out[:K]
    if cfg.scheme == "antithetic_strat_debiased":
        # Paired-plus-stratified candidate (E56f): complementary pairs plus l-stratification.
        #
        # E13 showed stratification and complementary pairing attack DIFFERENT variance
        # components, so the right scheme is the combination, not either one:
        #   * pairing  (mask, complement) kills the WITHIN-l variance -- the variance over *which*
        #     tokens are masked at fixed l, which cancels because the pair covers every token
        #     exactly once.  This is the larger component on a real model.
        #   * stratifying l across the pairs kills the BETWEEN-l variance, large because f(l) is
        #     steeply monotone.
        #   * the deterministic m(n) slot removes the bias the pairing would otherwise carry.
        #
        # Unbiasedness needs the P strata to be EQUAL-SIZED and to partition {1..n-1}; they do NOT
        # need to be closed under l -> n-l.  The reason is that l -> n-l is a bijection of
        # {1..n-1}, so it carries the partition {S_p} to another equal-sized partition {n-S_p}.
        # Averaging a slot over each gives (1/(n-1)) * sum_{l<n} phi(l) twice, hence exactly the
        # uniform marginal.  What DOES break it is an odd number of sampled slots: the loop then
        # skips a stratum (its slot is spent on a complement instead), the strata no longer
        # partition the range, and the bias returns.  So K must be odd, i.e. K-1 pairs' worth of
        # sampled slots plus one deterministic slot.
        npair = K - 1
        if npair % 2 != 0:
            raise ValueError(
                f"antithetic_strat_debiased needs an odd n_replicates (got K={K}): the K-1 sampled "
                "slots must split into complete complementary pairs or the stratification biases.")
        P = npair // 2
        hi = (nv - 1).clamp(min=1)
        coef = float(K) * (nvf - 1.0).clamp(min=1.0) / float(npair)
        for p in range(P):
            u = torch.rand(B, generator=gen, device=device)
            l = (((p + u) * hi.float()) / P).long().clamp(max=hi - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            comp = ~mask if valid is None else ((~mask) & valid)
            out.append((mask, (coef / l.float()) / scale_div))
            out.append((comp, (coef / (nv - l).clamp(min=1).float()) / scale_div))
        full = valid.clone() if valid is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        out.append((full, (float(K) / nvf) / scale_div))
        return out[:K]
    if cfg.scheme == "antithetic_debiased":
        # An unbiased repair in this family (E56). TraFL's complementary pairing has by
        # far the lowest variance of the published schemes -- a mask and its complement together
        # cover every token exactly once -- but it is BIASED, because pairing l with n-l forces
        # l ~ U{1..n-1} and the fully-masked level l=n is never drawn (see the table above).
        #
        # That bias has a closed form, and the missing term is DETERMINISTIC:
        #     m(n) = (1/n) * sum_i log p(y_i | everything masked)
        # is a single forward pass with no sampling in it at all.  So the bias can be removed
        # exactly, at the cost of one extra forward pass per replicate-set.  Writing E_a for the
        # antithetic estimator,
        #     E[E_a] = (n/(n-1)) * sum_{l<n} m(l)   =>   ELBO = ((n-1)/n) * E[E_a] + m(n).
        # The K-1 sampled slots therefore carry weight  K(n-1) / ((K-1) * l)  and the last slot
        # is the full mask at weight  K/n,  so that the mean over K replicates telescopes to the
        # line above.
        #
        # MEASURED OUTCOME (T6), stated because it is not the one this was built for: the bias
        # does vanish (0.0004 against antithetic's 0.688), but the variance does NOT stay at
        # antithetic's -- 0.418 against 0.094, because at K=4 only one of the three sampled slots
        # is a real complementary pair and the survivors carry an inflated weight.  `stratified_l`
        # gets to 0.099 -- antithetic's variance, no bias. End-to-end E64 later found this repair
        # modestly better under the training sampler and tied under argmax at K=4, while the clean
        # odd-K layout was substantially better. This is not a decoder-independent ranking.
        pairs = []
        hi = (nv - 1).clamp(min=1)
        npair = max(K - 1, 1)
        while len(pairs) < npair:
            # l is drawn iid uniform on {1..n-1}, exactly as TraFL does, and NOT stratified.
            # Stratifying here would break the construction: the complement slot takes n-l, so
            # forcing l into stratum j leaves the partner in n-stratum_j rather than in stratum
            # j+1, and the per-slot marginal stops being uniform -- reintroducing a bias, which
            # is the one thing this scheme exists to remove.  (A stratified *pairing* is
            # possible -- stratify l over {1..(n-1)/2} and let each pair cover l and n-l -- but
            # it is only exact for odd n with an even number of sampled slots, and n varies per
            # row under variable-length completions.  Left out deliberately.)
            l = (torch.rand(B, generator=gen, device=device) * hi.float()).long().clamp(max=hi - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            pairs.append((mask, l))
            if len(pairs) < npair:
                comp = ~mask if valid is None else ((~mask) & valid)
                pairs.append((comp, (nv - l).clamp(min=1)))
        coef = float(K) * (nvf - 1.0).clamp(min=1.0) / float(npair)
        for mask, l in pairs[:npair]:
            out.append(((coef / l.float()) / scale_div, mask))
        out = [(m, w) for w, m in out]
        full = valid.clone() if valid is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        out.append((full, (float(K) / nvf) / scale_div))
        return out[:K]
    if cfg.scheme in ("stratified_l", "systematic_l"):
        # Variance-reduced drop-in for `uniform_l`, motivated by E43/F15: because the loss is
        # quadratic, the sampled objective is L(delta) + Var(estimator), so shrinking the
        # estimator's variance is not a precision improvement -- it removes an implicit penalty
        # from the objective itself.
        #
        # The quantity being estimated is  ELBO = sum_{l=1..n} m(l),  and drawing l ~ U{1..n}
        # makes the *between-l* variance dominant, because f(l) = n*E[mean log p over the mask]
        # is steeply monotone in l (masking more tokens is uniformly harder).  Stratifying l
        # across the K replicates deletes precisely that component.
        #
        # Construction: replicate k draws  l_k = floor((k + u_k) * n / K) + 1.  Marginally over
        # k this is exactly uniform on {1..n}, so the estimator stays UNBIASED and the weight
        # formula is unchanged (n/l); only the joint law of the K draws differs.
        #   stratified_l  independent u_k per replicate  -- 17.7x variance reduction (T5)
        #   systematic_l  one shared u across replicates   --  5.5x variance reduction (T5)
        # stratified wins, and the reason is worth stating because the opposite was expected
        # here (survey sampling often prefers systematic for monotone populations): with
        # independent offsets the K within-stratum errors are independent and average down as
        # 1/K^2 * sum_k Var_k, whereas a shared offset makes the K replicates move together, so
        # the estimator inherits the full variance of one sweep with no averaging benefit.
        # That cancellation only returns if f is near-linear across a stratum, which it is not.
        # `stratified_l` is the stronger l-only scheme in this gate; `systematic_l` is kept as
        # the measured comparison. E64 does not support a universal end-to-end ranking at K=4.
        u_shared = torch.rand(B, generator=gen, device=device)
        for k in range(K):
            u = u_shared if cfg.scheme == "systematic_l" else torch.rand(B, generator=gen, device=device)
            l = (((k + u) * nvf) / K).long().clamp(max=nv - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            out.append((mask, (nvf / l.float()) / scale_div))
        return out[:K]
    while len(out) < cfg.n_replicates:
        if cfg.scheme == "uniform_l":
            l = (torch.rand(B, generator=gen, device=device) * nvf).long().clamp(max=nv - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            out.append((mask, (nvf / l.float()) / scale_div))
        elif cfg.scheme == "antithetic_l":
            hi = (nv - 1).clamp(min=1)
            l = (torch.rand(B, generator=gen, device=device) * hi.float()).long().clamp(max=hi - 1) + 1
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            out.append((mask, (nvf / l.float()) / scale_div))
            if len(out) < cfg.n_replicates:
                comp = ~mask if valid is None else ((~mask) & valid)
                out.append((comp, (nvf / (nv - l).clamp(min=1).float()) / scale_div))
        elif cfg.scheme == "coupled_espo":
            l = (torch.rand(B, generator=gen, device=device) * (nvf + 1.0)).long().clamp(max=nv)
            mask = _ranks(B, L, gen, device, valid) < l[:, None]
            if valid is not None:
                mask = mask & valid
            w = torch.where(l > 0, (nvf + 1.0) / l.clamp(min=1).float(),
                            torch.zeros_like(nvf))
            out.append((mask, w / scale_div))
            if len(out) < cfg.n_replicates:
                comp = ~mask if valid is None else ((~mask) & valid)
                lc = nv - l
                wc = torch.where(lc > 0, (nvf + 1.0) / lc.clamp(min=1).float(),
                                 torch.zeros_like(nvf))
                out.append((comp, wc / scale_div))
        elif cfg.scheme in ("bernoulli_t", "antithetic_t"):
            t = torch.rand(B, generator=gen, device=device).clamp(1e-6, 1 - 1e-6)
            mask = torch.rand(B, L, generator=gen, device=device) < t[:, None]
            if valid is not None:
                mask = mask & valid
            out.append((mask, (1.0 / t) / scale_div))
            if cfg.scheme == "antithetic_t" and len(out) < cfg.n_replicates:
                comp = ~mask if valid is None else ((~mask) & valid)
                out.append((comp, (1.0 / (1.0 - t)) / scale_div))
        else:
            raise ValueError(cfg.scheme)
    return out[: cfg.n_replicates]


def mean_based_level_probs(m_of_l: torch.Tensor, floor: float = 0.02) -> torch.Tensor:
    """Defensive mean-based proposal q(l) proportional to |m(l)|.

    This is variance-optimal only when the within-level estimator is deterministic.  For random
    masks the optimum is proportional to sqrt(E[X_l^2]), which also includes within-level noise.

    `floor` mixes in a little uniform mass so no level can ever be starved: the weight is 1/(l*q(l)),
    so a level with q -> 0 that does get drawn contributes an unbounded term.  This is the standard
    defensive-mixture trick; it bounds weights at the cost of moving toward uniform sampling.
    """
    w = m_of_l.detach().abs().double()
    w = w / w.sum().clamp(min=1e-12)
    n = w.numel()
    return ((1.0 - floor) * w + floor / n).float()


def optimal_level_probs_2m(m_of_l: torch.Tensor, var_of_l: torch.Tensor,
                           floor: float = 0.02) -> torch.Tensor:
    """The *correct* minimum-variance proposal for `sum_l X_l` when `X_l` is itself random.

    `optimal_level_probs` uses `q ∝ |m(l)|`, which is the textbook answer for estimating a sum of
    KNOWN terms by sampling one of them.  Here the term drawn at level `l` is a random variable
    `X_l` (which tokens got masked), and for that problem the variance of `X_L/q(L)` is minimised by

        q*(l) ∝ sqrt( E[X_l^2] ) = sqrt( m(l)^2 + Var[X_l] )

    which is the second-moment, not the mean.  The two coincide only when the within-level variance
    is negligible -- exactly the assumption that fails on a real model, where `Var[X_l]` is large at
    the levels whose mean is small.  This is why the mean-based heuristic underperformed
    stratification on `mdlm-owt` despite a strongly skewed `m(l)` (E69).
    """
    w = (m_of_l.detach().double() ** 2 + var_of_l.detach().double().clamp(min=0)).sqrt()
    w = w / w.sum().clamp(min=1e-12)
    n = w.numel()
    return ((1.0 - floor) * w + floor / n).float()


def score_from_masks(model, prompt, y, masks):
    """mean over replicates of weight * S(A).  Differentiable w.r.t. model parameters."""
    acc = None
    for mask, w in masks:
        lp = masked_logprobs(model, prompt, y, mask)
        s = (lp * mask.float()).sum(dim=1)
        val = w * s
        acc = val if acc is None else acc + val
    return acc / len(masks)


def completion_score(model, prompt, y, cfg: SurrogateCfg, gen, shared_masks=None,
                     valid=None):
    """Convenience: draw (or reuse) masks and score.  Returns (score (B,), masks).

    Passing `shared_masks` is how the policy and the frozen reference are scored on the *same*
    corruptions, which is TraFL's stated variance-reduction ("use the same masking pattern for
    the current and reference models").
    """
    if shared_masks is None:
        shared_masks = draw_masks(y.shape[1], y.shape[0], cfg, gen, y.device, valid)
    return score_from_masks(model, prompt, y, shared_masks), shared_masks


# --------------------------------------------------------------------------------------------
# exact marginal log p_theta(y | x) for an arbitrary order rule, batched
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def exact_logp_batch(model, prompt, y, order: str = "random", order_temp: float = 0.0,
                     state_chunk: int = 64, with_elbo: bool = False):
    r"""Exact log p_theta(y|x) under the sampler defined by (order, order_temp).

    The generative process is: repeatedly pick a masked position i with probability
    P_order(i | z_S), then emit y_i with probability p_theta(y_i | z_S).  Because both factors
    depend on the *set* S of already-revealed positions and not on the order in which they were
    revealed, the L! sum collapses to a 2^L subset DP:

        g(S) = sum_{i in S} g(S \ {i}) * P_order(i | z_{S\{i}}) * p_theta(y_i | z_{S\{i}})

    with log p(y|x) = log g([L]).  (For the uniform order rule the P_order factors contribute a
    constant -log L!, matching `exact_logp_uniform_order`.)

    Returns (B,) float64 tensor.  Cost: 2^L forward passes of batch B.
    """
    B, L = y.shape
    dev = y.device
    acc = torch.device("cpu")          # MPS has no float64; the DP accumulates on CPU
    N = 2 ** L
    revealed = _all_subsets_mask(L, dev)                    # (N,L) bit j == position j revealed
    lp_tok = torch.zeros(N, B, L, dtype=torch.float64, device=acc)
    lp_ord = torch.zeros(N, B, L, dtype=torch.float64, device=acc)
    NEGBIG = -1e30
    for s in range(0, N, state_chunk):
        sl = slice(s, min(s + state_chunk, N))
        rv = revealed[sl]                                   # (c,L)
        c = rv.shape[0]
        mk = (~rv)[:, None, :].expand(c, B, L).reshape(c * B, L)
        pb = prompt[None].expand(c, B, prompt.shape[1]).reshape(c * B, -1)
        yb = y[None].expand(c, B, L).reshape(c * B, L)
        P = prompt.shape[1]
        z = build_z(pb, yb, mk, model.mask_id)
        logits = model(z)[:, P:, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        lp_tok[sl] = logp.gather(-1, yb.unsqueeze(-1)).squeeze(-1).view(c, B, L).cpu().double()
        if order == "random":
            navail = (~rv).sum(dim=1).clamp(min=1).float()           # (c,)
            o = (-torch.log(navail))[:, None, None].expand(c, B, L).clone()
        elif order == "ar":
            pos = torch.arange(L, device=dev)
            first = torch.where(rv, torch.full_like(pos, L + 1).expand(c, L), pos.expand(c, L))
            nxt = first.argmin(dim=1)
            o = torch.full((c, B, L), NEGBIG, device=dev)
            o.scatter_(-1, nxt[:, None, None].expand(c, B, 1), 0.0)
        else:
            conf = logp.max(dim=-1).values.view(c, B, L)
            conf = conf.masked_fill(rv[:, None, :].expand(c, B, L), NEGBIG)
            if order_temp == 0.0:
                am = conf.argmax(dim=-1, keepdim=True)
                o = torch.full((c, B, L), NEGBIG, device=dev)
                o.scatter_(-1, am, 0.0)
            else:
                o = torch.log_softmax(conf / order_temp, dim=-1)
            o = o
        lp_ord[sl] = o.cpu().double()

    step = (lp_tok + lp_ord)                                 # (N,B,L) on CPU/float64
    logg = torch.full((N, B), float("-inf"), dtype=torch.float64, device=acc)
    logg[0] = 0.0
    pc = revealed.sum(dim=1).cpu()
    for k in range(1, L + 1):
        states = torch.nonzero(pc == k, as_tuple=True)[0].tolist()
        for state in states:
            terms = []
            for i in range(L):
                bit = 1 << i
                if state & bit:
                    prev = state ^ bit
                    terms.append(logg[prev] + step[prev, :, i])
            logg[state] = torch.logsumexp(torch.stack(terms, 0), dim=0)
    if not with_elbo:
        return logg[N - 1]
    # The exact ELBO comes free from the same forward passes: for a *revealed* set `state` the
    # masked set is its complement A, and
    #     ELBO = sum_A S(A) * (|A|-1)! (L-|A|)! / L!      with  S(A) = sum_{i in A} log p(y_i|z_A)
    # (derivation and the two-route cross-check are in tests/test_elbo_identity.py).  Reporting
    # `log p(y|x) - ELBO(y|x)` is then the *exact Jensen gap* of the surrogate every dLLM RL
    # method optimises -- a quantity none of them measures.
    rev = revealed.cpu()
    sizes_masked = (L - rev.sum(dim=1)).double()                  # |A|
    lgam = torch.lgamma
    logw = (lgam(sizes_masked.clamp(min=1)) + lgam(L - sizes_masked + 1.0)
            - lgam(torch.tensor(L + 1.0, dtype=torch.float64)))
    w = torch.exp(logw)
    w = torch.where(sizes_masked == 0, torch.zeros_like(w), w)
    S = (lp_tok * (~rev).double()[:, None, :]).sum(-1)             # (N,B)
    elbo = (S * w[:, None]).sum(0)
    return logg[N - 1], elbo


def exact_logp_diff(model, prompt, y, order: str = "random", order_temp: float = 0.0):
    """Differentiable exact `log p_theta(y|x)` for small L (float32, on device).

    Same 2^L subset DP as `exact_logp_batch`, but kept in the autograd graph so it can be used
    *inside* a loss.  This exists for one purpose: to separate two things that are confounded in
    the trajectory-level objective.  `trafl_tau` differs from `trafl` both by being path-specific
    and by using an exact log-probability instead of a 4-sample Monte-Carlo ELBO surrogate.
    An objective built on the exact *completion* log-probability isolates the second effect, so
    whatever is left is attributable to the first.

    Only tractable because the toy uses L = 4-7; there is no version of this for a real dLLM,
    which is exactly why the toy is the instrument.
    """
    B, L = y.shape
    dev = y.device
    N = 2 ** L
    revealed = _all_subsets_mask(L, dev)
    P = prompt.shape[1]
    lp_tok, lp_ord = [], []
    for s in range(N):
        rv = revealed[s]
        mk = (~rv)[None, :].expand(B, L)
        z = build_z(prompt, y, mk, model.mask_id)
        logits = model(z)[:, P:, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        lp_tok.append(logp.gather(-1, y.unsqueeze(-1)).squeeze(-1))
        if order == "random":
            n = max(1, int((~rv).sum()))
            lp_ord.append(torch.full((B, L), -math.log(n), device=dev))
        elif order == "ar":
            first = int(torch.nonzero(~rv, as_tuple=True)[0][0]) if (~rv).any() else 0
            o = torch.full((B, L), -1e30, device=dev)
            o[:, first] = 0.0
            lp_ord.append(o)
        else:
            conf = logp.max(dim=-1).values.masked_fill(rv[None, :].expand(B, L), -1e30)
            if order_temp == 0.0:
                am = conf.argmax(dim=-1, keepdim=True)
                o = torch.full((B, L), -1e30, device=dev)
                o = o.scatter(-1, am, 0.0)
            else:
                o = torch.log_softmax(conf / order_temp, dim=-1)
            lp_ord.append(o)
    step = [lp_tok[s] + lp_ord[s] for s in range(N)]
    logg = [None] * N
    logg[0] = torch.zeros(B, device=dev)
    for k in range(1, L + 1):
        for state in range(N):
            if bin(state).count("1") != k:
                continue
            terms = [logg[state ^ (1 << i)] + step[state ^ (1 << i)][:, i]
                     for i in range(L) if state & (1 << i)]
            logg[state] = torch.logsumexp(torch.stack(terms, 0), dim=0)
    return logg[N - 1]
