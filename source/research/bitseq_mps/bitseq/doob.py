"""Exact reference-tilt feasibility in the fixed uniform-position family.

KL(Q_gamma || P_theta) >= occupancy-weighted position KL. The unrestricted
tabular word policy Q*(word|position,state) attains this floor with fixed positions;
the shared neural network need not attain it. This is not a practical-TraFL target.
"""

import math

import numpy as np

from bitseq.evaluate import _validate


def position_floor(graph, reference, terminal_reward, gamma):
    table = _validate(graph, reference)
    reward = np.asarray(terminal_reward, dtype=np.float64)
    if (
        reward.shape != (len(graph.terminals),)
        or not np.isfinite(reward).all()
        or not np.isfinite(gamma)
    ):
        raise ValueError("Finite terminal utility and tilt required")
    logh = np.full(len(graph.boards), -np.inf)
    logh[graph.terminals] = gamma * reward
    for k in range(graph.length - 1, -1, -1):
        edges = graph.edge_levels[k]
        src, dst, pos, word = [
            a[edges] for a in (graph.src, graph.dst, graph.pos, graph.word)
        ]
        logweight = table[src, pos, word] - math.log(graph.length - k) + logh[dst]
        np.logaddexp.at(logh, src, logweight)
    edge_logp = (
        table[graph.src, graph.pos, graph.word]
        - np.log(graph.length - graph.filled[graph.src])
        + logh[graph.dst]
        - logh[graph.src]
    )
    edge_p = np.exp(edge_logp)
    posmass = np.zeros((len(graph.boards), graph.length))
    np.add.at(posmass, (graph.src, graph.pos), edge_p)
    empty = graph.boards == -1
    if np.max(abs(posmass[graph.filled < graph.length].sum(1) - 1)) > 1e-10:
        raise ValueError("Doob kernel lost normalized mass")
    occupancy = np.zeros(len(graph.boards))
    occupancy[graph.root] = 1.0
    for edges in graph.edge_levels:
        np.add.at(
            occupancy, graph.dst[edges], occupancy[graph.src[edges]] * edge_p[edges]
        )
    layer_mass = np.array([occupancy[rows].sum() for rows in graph.levels])
    if np.max(abs(layer_mass - 1)) > 1e-10:
        raise ValueError("Doob occupancy lost mass")
    local = np.zeros(len(graph.boards))
    rows, positions = np.where(empty)
    positive = posmass[rows, positions] > 0
    r, p = rows[positive], positions[positive]
    np.add.at(
        local,
        r,
        posmass[r, p]
        * (np.log(posmass[r, p]) + np.log(graph.length - graph.filled[r])),
    )
    words = table.copy()
    words[graph.src, graph.pos, graph.word] = edge_logp - np.log(
        posmass[graph.src, graph.pos]
    )
    return dict(
        log_normalizer=float(logh[graph.root]),
        position_kl_floor=float(occupancy @ local),
        position_probabilities=posmass,
        occupancy=occupancy,
        local_position_kl=local,
        attaining_logwords=words,
        layer_mass=layer_mass,
        tilted_terminal_p=occupancy[graph.terminals],
        max_position_tv=float(
            np.max(
                0.5
                * np.sum(
                    abs(
                        posmass
                        - np.divide(
                            empty,
                            empty.sum(1)[:, None],
                            out=np.zeros_like(posmass),
                            where=empty.sum(1)[:, None] > 0,
                        )
                    ),
                    axis=1,
                )
            )
        ),
    )
