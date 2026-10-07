"""Exact finite-DAG probabilities/log-moments of a declared word probability law.

Evaluation softmax is CPU64 of saved device32 logits. Native device sampling and
its precision are separately checked; no terminal renormalization hides errors.
"""

import math

import numpy as np
import torch


def cache_logwords(model, boards, chunk=256):
    device = next(model.parameters()).device
    training = model.training
    output = np.empty((len(boards), boards.shape[1], 4), dtype=np.float64)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(boards), chunk):
                b = torch.as_tensor(boards[start : start + chunk], device=device)
                ids = torch.cat(
                    [
                        torch.full((len(b), 1), 5, device=device),
                        torch.where(b == -1, 4, b),
                    ],
                    1,
                ).long()
                logits = model(ids)[:, 1:].cpu().double()
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite cached model logits")
                output[start : start + len(b)] = logits.log_softmax(-1).numpy()
    finally:
        model.train(training)
    return output


def _validate(graph, logwords):
    table = np.asarray(logwords, dtype=np.float64)
    if (
        table.shape != (len(graph.boards), graph.length, graph.vocab)
        or not np.isfinite(table).all()
    ):
        raise ValueError(
            "finite word log-probabilities with the declared shape required"
        )
    norm = np.exp(table).sum(-1)
    if np.max(np.abs(norm - 1)) > 1e-10:
        raise ValueError(
            "word table is not normalized; declare softmax precision explicitly"
        )
    return table


def _propagate(graph, table, order, cross=None):
    if order not in ("random", "ar"):
        raise ValueError("unknown decoder")
    n = len(graph.boards)
    logmass = np.full(n, -np.inf)
    logmass[graph.root] = 0.0
    mean = np.zeros(n)
    second = np.zeros(n)
    crossmean = np.zeros(n)
    first = (graph.boards == -1).argmax(1)
    for k, edges in enumerate(graph.edge_levels):
        src, dst, pos, word = [
            arr[edges] for arr in (graph.src, graph.dst, graph.pos, graph.word)
        ]
        usable = np.isfinite(logmass[src])
        if order == "ar":
            usable &= pos == first[src]
        src, dst, pos, word = [arr[usable] for arr in (src, dst, pos, word)]
        factor = -math.log(graph.length - k) if order == "random" else 0.0
        step = table[src, pos, word] + factor
        candidates = logmass[src] + step
        np.logaddexp.at(logmass, dst, candidates)
        weights = np.exp(candidates - logmass[dst])
        np.add.at(mean, dst, weights * (mean[src] + step))
        np.add.at(
            second, dst, weights * (second[src] + 2 * step * mean[src] + step**2)
        )
        if cross is not None:
            increment = cross[src, pos, word] + factor
            np.add.at(crossmean, dst, weights * (crossmean[src] + increment))
    term = graph.terminals
    logp = logmass[term]
    p = np.exp(logp)
    if abs(p.sum() - 1) > 1e-10:
        raise ValueError("terminal propagation lost mass; no end renormalization")
    layer_mass = np.array([np.exp(logmass[rows]).sum() for rows in graph.levels])
    if np.max(np.abs(layer_mass - 1)) > 1e-10:
        raise ValueError("layer propagation lost mass")
    conditional = logp - mean[term]
    if (
        conditional.min() < -1e-8
        or conditional.max() > math.lgamma(graph.length + 1) + 1e-8
    ):
        raise ValueError("conditional entropy violates path-count bounds")
    return dict(
        p=p,
        logp=logp,
        conditional_entropy=conditional,
        terminal_entropy=float(-np.dot(p, logp)),
        joint_entropy=float(-np.dot(p, mean[term])),
        logpath_mean=mean[term],
        logpath_second=second[term],
        cross_mean=crossmean[term],
        layer_mass=layer_mass,
        visits_total=float(layer_mass.sum()),
        underflow_terminals=int(np.sum((p == 0) & np.isfinite(logp))),
        log_occupancy=logmass,
    )


def evaluate_exact(graph, logwords, order="random", reference=None):
    table = _validate(graph, logwords)
    ref = None if reference is None else _validate(graph, reference)
    result = _propagate(graph, table, order, ref)
    result["current_conditional_entropy"] = float(
        np.dot(result["p"], result["conditional_entropy"])
    )
    if ref is not None:
        baseline = _propagate(graph, ref, order, table)
        p, pr = result["p"], baseline["p"]
        forward = (
            result["logpath_mean"]
            - result["cross_mean"]
            - result["logp"]
            + baseline["logp"]
        )
        reverse = (
            baseline["logpath_mean"]
            - baseline["cross_mean"]
            - baseline["logp"]
            + result["logp"]
        )
        result.update(
            path_kl_current_to_ref=float(np.dot(p, forward)),
            path_kl_ref_to_current=float(np.dot(pr, reverse)),
            fixed_reference_conditional_entropy=float(
                np.dot(pr, result["conditional_entropy"])
            ),
            terminal_kl_current_to_ref=float(
                np.dot(p, result["logp"] - baseline["logp"])
            ),
            terminal_kl_ref_to_current=float(
                np.dot(pr, baseline["logp"] - result["logp"])
            ),
        )
        if (
            min(result["path_kl_current_to_ref"], result["path_kl_ref_to_current"])
            < -1e-9
        ):
            raise ValueError("negative conditional KL beyond rounding")
    return result


def denoising_excess(graph, logwords, q):
    from bitseq.oracle import teacher_table

    table = _validate(graph, logwords)
    teacher, mass = teacher_table(graph, q)
    level_ce = []
    level_kl = []
    for k, edges in enumerate(graph.edge_levels):
        l = graph.length - k
        src, pos, word = [arr[edges] for arr in (graph.src, graph.pos, graph.word)]
        weights = (
            mass[src]
            * np.exp(teacher[src, pos, word])
            / (l * math.comb(graph.length, l))
        )
        level_ce.append(float(-np.dot(weights, table[src, pos, word])))
        level_kl.append(
            float(np.dot(weights, teacher[src, pos, word] - table[src, pos, word]))
        )
    entropy = float(-np.dot(q, np.log(q)))
    ce = sum(level_ce)
    kl = sum(level_kl)
    if abs(ce - entropy - kl) > 1e-9:
        raise ValueError("independent denoising excess routes disagree")
    return dict(
        cross_entropy=ce,
        teacher_entropy=entropy,
        excess_kl=kl,
        cross_entropy_minus_teacher_entropy=ce - entropy,
        per_word_excess=kl / graph.length,
        level_cross_entropy=level_ce,
        level_excess_kl=level_kl,
    )


def native_terminal_check(graph, logwords, order):
    """Conformance check using the unmodified native terminal solver/default dtype."""
    from bitseq.oracle import enumerated_env
    from gfn.estimators import DiscretePolicyEstimator

    # Native solver creates visits/pmf in default dtype; match it, don't patch it.
    table = torch.as_tensor(_validate(graph, logwords), dtype=torch.get_default_dtype())
    env = enumerated_env(graph)

    class ActionTable(torch.nn.Module):
        input_dim = 6

        def forward(self, boards):
            rows = torch.from_numpy(graph.index(boards.numpy()))
            empty = boards == -1
            count = empty.sum(1)
            words = table[rows].clone()
            if order == "random":
                words -= count.clamp(min=1).to(table.dtype).log()[:, None, None]
                allowed = empty
            elif order == "ar":
                allowed = (
                    torch.nn.functional.one_hot(empty.long().argmax(1), 6).bool()
                    & empty
                )
            else:
                raise ValueError("unknown order")
            words = words.masked_fill(~allowed[:, :, None], -torch.inf)
            exitlog = torch.where(count == 0, 0.0, -torch.inf).to(table.dtype)[:, None]
            return torch.cat([words.reshape(-1, 24), exitlog], 1)

    pf = DiscretePolicyEstimator(ActionTable(), n_actions=25, debug=True)
    return env.exact_terminating_distribution(pf).numpy()
