"""Teacher-conditional labels must preserve the actual hard-label expected loss/gradient."""

import itertools

import numpy as np
import torch


def test_soft_labels_equal_exhaustive_hard_label_expectation_at_one_context():
    from bitseq.model import make_denoiser
    from bitseq.oracle import FiniteGraph, task_distribution, teacher_table
    from bitseq.train import pretraining_loss, soft_pretraining_loss

    graph = FiniteGraph()
    q = task_distribution(graph, 12)["q"]
    teacher, mass = teacher_table(graph, q)
    board = np.array([[0, -1, 2, -1, 3, 0]])
    row = graph.index(board)[0]
    targets = torch.tensor(
        np.exp(teacher[row : row + 1]), dtype=torch.float32, requires_grad=True
    )
    model = make_denoiser(119)
    soft = soft_pretraining_loss(model, torch.from_numpy(board), targets)
    hard_rows = []
    probabilities = []
    for a, b in itertools.product(range(4), repeat=2):
        y = [0, a, 2, b, 3, 0]
        index = sum(word * 4 ** (5 - i) for i, word in enumerate(y))
        probabilities.append(q[index] / mass[row])
        hard_rows.append(y)
    masks = torch.tensor([[0, 1, 0, 1, 0, 0]] * 16, dtype=torch.bool)
    losses = []
    # Explicit complete conditional law, not the soft implementation's helper.
    for y in hard_rows:
        losses.append(pretraining_loss(model, torch.tensor([y]), masks[:1]))
    expected = torch.dot(
        torch.stack(losses), torch.tensor(probabilities, dtype=torch.float32)
    )
    torch.testing.assert_close(soft, expected, rtol=1e-6, atol=1e-6)
    ga = torch.autograd.grad(soft, tuple(model.parameters()), retain_graph=True)
    gb = torch.autograd.grad(expected, tuple(model.parameters()))
    a, b = [torch.cat([g.flatten() for g in gs]) for gs in [ga, gb]]
    assert (a - b).norm() <= 1e-5 * b.norm() + 1e-6
    assert targets.grad is None
