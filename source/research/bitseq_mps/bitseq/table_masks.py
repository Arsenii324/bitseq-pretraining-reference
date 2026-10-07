"""CPU-only mask moments from saved exact-evaluation word-probability tables.

This law is CPU64 softmax of device32 logits, not bit-identical device32
production scoring. No optimizer/gradient/model call occurs in this analysis.
"""

import math

import numpy as np

from bitseq.evaluate import _validate


def cached_mask_ratios(graph, current, reference):
    current, reference = _validate(graph, current), _validate(graph, reference)
    labels = graph.boards[graph.terminals]
    ratios = []
    for a in range(1, 2**graph.length):
        positions = np.array([i for i in range(graph.length) if a & (1 << i)])
        contexts = labels.copy()
        contexts[:, positions] = -1
        rows = graph.index(contexts)
        diff = (
            current[rows[:, None], positions[None, :], labels[:, positions]]
            - reference[rows[:, None], positions[None, :], labels[:, positions]]
        )
        ratios.append(diff.mean(1))
    return np.stack(ratios)


def mask_moments(x, *, length, k=4):
    x = np.asarray(x, dtype=np.float64)
    n = 2**length - 1
    if (
        x.ndim != 2
        or x.shape[0] != n
        or k < 2
        or k % 2
        or length < 2
        or not np.isfinite(x).all()
    ):
        raise ValueError("Complete finite mask matrix and even replicatecount required")
    wi = np.array(
        [1 / (length * math.comb(length, a.bit_count())) for a in range(1, n + 1)]
    )
    mu = wi @ x
    pair = (x[:-1] + x[-2::-1]) / 2
    wc = np.array(
        [1 / ((length - 1) * math.comb(length, a.bit_count())) for a in range(1, n)]
    )
    mc = wc @ pair
    return dict(
        iid_mean=mu,
        comp_mean=mc,
        full_mask_ratio=x[-1],
        comp_bias=mc - mu,
        iid_variance=wi @ ((x - mu) ** 2) / k,
        comp_variance=wc @ ((pair - mc) ** 2) / (k // 2),
    )
