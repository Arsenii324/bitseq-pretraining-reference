"""Exact finite mask moments: expected noisy loss is not squared mean score.

Chunk over completions in callers. No on-policy optimizer updates in a frozen
audit, and no causal training explanation from a gradient comparison alone.
"""

import math

import torch

from tdlm.mdm import build_z


def all_mask_tokens(model, y):
    batch, length = y.shape
    n = 2**length - 1
    if length < 2 or n * batch > 512:
        raise ValueError(
            "Audit supports length>=2 and at most512 context rows; chunk completions"
        )
    masks = torch.tensor(
        [[bool(a & (1 << i)) for i in range(length)] for a in range(1, n + 1)],
        dtype=torch.bool,
        device=y.device,
    )
    masks = masks[:, None, :].expand(n, batch, length).reshape(n * batch, length)
    labels = y[None, :, :].expand(n, batch, length).reshape(n * batch, length)
    prompt = torch.full((n * batch, 1), 5, dtype=torch.long, device=y.device)
    ids = build_z(prompt, labels, masks, model.mask_id)
    lp = model(ids)[:, 1:].float().log_softmax(-1)
    return lp.gather(-1, labels[:, :, None]).squeeze(-1).reshape(n, batch, length)


def intensive_from_tokens(tokens):
    n, batch, length = tokens.shape
    if n != 2**length - 1:
        raise ValueError("Complete nonempty mask tokens required")
    masks = torch.tensor(
        [[bool(a & (1 << i)) for i in range(length)] for a in range(1, n + 1)],
        dtype=tokens.dtype,
        device=tokens.device,
    )
    return (tokens * masks[:, None, :]).sum(-1) / masks.sum(-1)[:, None]


def all_mask_ratios(policy, reference, y):
    current = intensive_from_tokens(all_mask_tokens(policy, y))
    with torch.no_grad():
        ref = intensive_from_tokens(all_mask_tokens(reference, y))
    return current - ref


def log_marginal_from_masks(tokens):
    n, batch, length = tokens.shape
    if n != 2**length - 1:
        raise ValueError("Complete nonempty mask tokens required")
    values = [torch.zeros(batch, dtype=tokens.dtype, device=tokens.device)]
    for revealed in range(1, n + 1):
        count = revealed.bit_count()
        terms = []
        for pos in range(length):
            if revealed & (1 << pos):
                previous = revealed ^ (1 << pos)
                masked = n ^ previous
                terms.append(
                    values[previous]
                    + tokens[masked - 1, :, pos]
                    - math.log(length - count + 1)
                )
        values.append(torch.logsumexp(torch.stack(terms), 0))
    return values[-1]


def exact_mask_objectives(x, offset, *, length=6, k=4):
    n = 2**length - 1
    if (
        length < 2
        or k < 2
        or k % 2
        or x.ndim != 2
        or x.shape[0] != n
        or offset.shape != (x.shape[1],)
    ):
        raise ValueError(
            "Complete nonempty mask matrix, per-completion offsets and evenK required"
        )
    if not torch.isfinite(x).all() or not torch.isfinite(offset).all():
        raise ValueError("Finite scores required")
    wi = torch.tensor(
        [1 / (length * math.comb(length, a.bit_count())) for a in range(1, n + 1)],
        dtype=x.dtype,
        device=x.device,
    )
    mu = (wi[:, None] * x).sum(0)
    variance = (wi[:, None] * (x - mu).square()).sum(0) / k
    pair = (x[:-1] + x[torch.arange(n - 2, -1, -1, device=x.device)]) / 2
    wc = torch.tensor(
        [1 / ((length - 1) * math.comb(length, a.bit_count())) for a in range(1, n)],
        dtype=x.dtype,
        device=x.device,
    )
    mc = (wc[:, None] * pair).sum(0)
    vc = (wc[:, None] * (pair - mc).square()).sum(0) / (k // 2)
    return dict(
        elbo_loss=(mu + offset).square().mean(),
        iid_loss=((mu + offset).square() + variance).mean(),
        comp_loss=((mc + offset).square() + vc).mean(),
        iid_mean=mu,
        comp_mean=mc,
        comp_bias=mc - mu,
        iid_variance=variance,
        comp_variance=vc,
    )


def exact_policy_gradients(policy, reference, y, offset, *, chunk=8):
    """Full conditional loss/gradient, chunked over completions end-to-end.

    Group rewards/head are frozen by caller BEFORE chunking; don't recenter them
    per chunk. CPU64 moments/DP differentiate through device32 token scores.
    """
    if chunk < 1 or offset.shape != (len(y),) or len(y) < 1:
        raise ValueError(
            "Nonempty completions, frozen offsets and positivechunk required"
        )
    parameters = tuple(policy.parameters())
    total = sum(p.numel() for p in parameters)
    names = ("iid_loss", "comp_loss", "elbo_loss", "marginal_loss")
    gradients = {name: torch.zeros(total, dtype=torch.float64) for name in names}
    losses = {name: 0.0 for name in names}
    moments = {name: [] for name in ("iid_variance", "comp_variance", "comp_bias")}
    for start in range(0, len(y), chunk):
        labels = y[start : start + chunk]
        tokens = all_mask_tokens(policy, labels).cpu().double()
        with torch.no_grad():
            ref = all_mask_tokens(reference, labels).cpu().double()
        x = intensive_from_tokens(tokens - ref)
        result = exact_mask_objectives(
            x,
            offset[start : start + len(labels)].detach().cpu().double(),
            length=y.shape[1],
        )
        logratio = (
            log_marginal_from_masks(tokens) - log_marginal_from_masks(ref)
        ) / y.shape[1]
        result["marginal_loss"] = (
            (logratio + offset[start : start + len(labels)].detach().cpu().double())
            .square()
            .mean()
        )
        weight = len(labels) / len(y)
        for index, name in enumerate(names):
            grad = torch.autograd.grad(
                result[name] * weight, parameters, retain_graph=index < len(names) - 1
            )
            flat = torch.cat([g.detach().cpu().double().flatten() for g in grad])
            gradients[name].add_(flat)
            losses[name] += float(result[name].detach()) * weight
        for name in moments:
            moments[name].append(result[name].detach().cpu())
        del tokens, ref, x, result, grad, flat
    return dict(
        gradients=gradients,
        losses=losses,
        moments={k: torch.cat(v) for k, v in moments.items()},
    )
