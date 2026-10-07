"""Independent finite truth/indexing and evaluation-only upstream enumeration.

No environment transitions/reward are reimplemented for training. Mathematical
graph/teacher construction is independent of torchgfn's implementation.
"""

import math

import numpy as np
import torch

from bitseq.env import MODES


class FiniteGraph:
    def __init__(self, length=6, vocab=4):
        if length < 1 or vocab < 2 or (vocab + 1) ** length > 1_000_000:
            raise ValueError("invalid or unbudgeted finite graph")
        self.length, self.vocab = length, vocab
        self.powers = (vocab + 1) ** np.arange(length - 1, -1, -1, dtype=np.int64)
        digits = np.indices((vocab + 1,) * length, dtype=np.int64).reshape(length, -1).T
        self.boards = np.where(digits == vocab, -1, digits)
        self.filled = (self.boards >= 0).sum(1)
        self.root = len(self.boards) - 1
        self.levels = [np.flatnonzero(self.filled == k) for k in range(length + 1)]
        self.terminals = self.levels[-1]
        rows, positions = np.where(self.boards == -1)
        self.src = np.repeat(rows, vocab)
        self.pos = np.repeat(positions, vocab)
        self.word = np.tile(np.arange(vocab), len(rows))
        self.dst = self.src + (self.word - vocab) * self.powers[self.pos]
        self.edge_levels = [
            np.flatnonzero(self.filled[self.src] == k) for k in range(length)
        ]

    def index(self, boards):
        boards = np.asarray(boards)
        if (
            boards.ndim != 2
            or boards.shape[1] != self.length
            or not np.issubdtype(boards.dtype, np.integer)
        ):
            raise ValueError("integer board matrix required")
        if ((boards < -1) | (boards >= self.vocab)).any():
            raise ValueError("invalid/sink board")
        return np.where(boards == -1, self.vocab, boards) @ self.powers


def task_distribution(graph, alpha):
    if (graph.length, graph.vocab) != (6, 4):
        raise ValueError("the declared reward task is six four-valued words")
    words = graph.boards[graph.terminals]
    integers = words @ (4 ** np.arange(5, -1, -1))
    modes = [int(s, 2) for s in MODES]
    distances = np.array(
        [min((int(y) ^ m).bit_count() for m in modes) for y in integers]
    )
    logits = -float(alpha) * distances / 12
    maximum = logits.max()
    logz = float(maximum + math.log(np.exp(logits - maximum).sum()))
    return dict(
        q=np.exp(logits - logz),
        logZ=logz,
        distance=distances,
        reward=np.exp(-2.0 * distances),
        log_reward=-2.0 * distances,
        basin_masks=np.array(
            [[(int(y) ^ m).bit_count() <= 2 for y in integers] for m in modes]
        ),
        mode_indices=np.array(modes, dtype=np.int64),
    )


def teacher_table(graph, q):
    q = np.asarray(q, dtype=np.float64)
    if (
        q.shape != (graph.vocab**graph.length,)
        or not np.isfinite(q).all()
        or (q <= 0).any()
        or abs(q.sum() - 1) > 1e-12
    ):
        raise ValueError(
            "teacher needs a positive normalized full-support terminal distribution"
        )
    mass = np.zeros(len(graph.boards), dtype=np.float64)
    mass[graph.terminals] = q
    for k in range(graph.length - 1, -1, -1):
        rows = graph.levels[k]
        first = (graph.boards[rows] == -1).argmax(1)
        children = (
            rows[:, None]
            + (np.arange(graph.vocab) - graph.vocab) * graph.powers[first, None]
        )
        mass[rows] = mass[children].sum(1)
    logwords = np.full(
        (len(graph.boards), graph.length, graph.vocab), -math.log(graph.vocab)
    )
    logwords[graph.src, graph.pos, graph.word] = np.log(mass[graph.dst]) - np.log(
        mass[graph.src]
    )
    return logwords, mass


def enumerated_env(graph):
    """Add only missing enumeration interfaces; inherit transition/reward code."""
    if (graph.length, graph.vocab) != (6, 4):
        raise ValueError("upstream enumeration check is for the declared task")
    from gfn.gym.bitSequenceNonAutoregressive import NonAutoregressiveBitSequence

    class Enumerated(NonAutoregressiveBitSequence):
        supports_enumeration = True

        @property
        def n_states(self):
            return len(graph.boards)

        @property
        def all_states(self):
            return self.States(torch.from_numpy(graph.boards))

        def get_states_indices(self, states):
            digits = torch.where(states.tensor == -1, 4, states.tensor)
            return (digits * torch.tensor(graph.powers, device=digits.device)).sum(-1)

        def get_terminating_states_indices(self, states):
            return (
                states.tensor
                * torch.tensor(4 ** np.arange(5, -1, -1), device=states.device)
            ).sum(-1)

    h = torch.tensor([[int(c) for c in mode] for mode in MODES])
    return Enumerated(
        word_size=2,
        seq_size=12,
        n_modes=8,
        reward_exponent=24.0,
        H=h,
        device_str="cpu",
        debug=True,
    )
