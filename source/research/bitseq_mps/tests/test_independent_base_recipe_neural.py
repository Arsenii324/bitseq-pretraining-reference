"""Opt-in actual hard→soft→LR-tail replay gate; NOT RUN at preparation.

This is a disposable CPU few-update check, not a calibrated independent base.
It is withheld while the primary scientific worker/numerical audit own compute.
"""

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("BITSEQ_BASE_REPLAY_GATE") != "1",
                    reason="Deferred by compute sequencing; set BITSEQ_BASE_REPLAY_GATE=1 explicitly")
def test_hard_soft_tail_checkpoint_replay_matches_literal_loop(tmp_path):
    import numpy as np
    import torch
    from bitseq.artifacts import load_checkpoint, save_checkpoint
    from bitseq.model import make_denoiser
    from bitseq.oracle import FiniteGraph, task_distribution, teacher_table
    from bitseq.train import draw_pretraining_masks, pretraining_loss, soft_pretraining_loss

    path = Path(__file__).parents[1] / "checks/reproduce_base.py"
    spec = importlib.util.spec_from_file_location("independent_base_neural", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    graph = FiniteGraph()
    prior = task_distribution(graph, 12)
    teacher, _ = teacher_table(graph, prior["q"])
    words, probabilities = torch.tensor(graph.boards[graph.terminals]), torch.tensor(prior["q"])
    model = make_denoiser(101)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0)
    rngs = {"data": torch.Generator().manual_seed(10101),
            "mask": torch.Generator().manual_seed(20101)}
    for _ in range(2):
        y = words[torch.multinomial(probabilities, 8, replacement=True, generator=rngs["data"])]
        masks = draw_pretraining_masks(8, rngs["mask"])
        optimizer.zero_grad(set_to_none=True)
        pretraining_loss(model, y, masks).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    parent = tmp_path / "honest_two_hard_updates.pt"
    save_checkpoint(parent, model, optimizer, rngs, {"step": 2, "seed": 101})
    parent_digest = recipe.digest(parent)
    restored = make_denoiser(999)  # proves this factory seed is overwritten.
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1, weight_decay=0)
    restored_rngs = {"data": torch.Generator(), "mask": torch.Generator()}
    load_checkpoint(parent, restored, restored_optimizer, restored_rngs, {"step": 2, "seed": 101})

    # Independent literal stage/reference, not another call to the helper.
    def literal_soft_step(lr):
        for group in optimizer.param_groups:
            group["lr"] = lr
        y = words[torch.multinomial(probabilities, 8, replacement=True, generator=rngs["data"])]
        masks = draw_pretraining_masks(8, rngs["mask"])
        boards = torch.where(masks, -1, y)
        targets = torch.tensor(np.exp(teacher[graph.index(boards.numpy())]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        soft_pretraining_loss(model, boards, targets).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()

    for lr in (6e-4, 6e-4, 6e-5, 6e-5):
        literal_soft_step(lr)
        for group in restored_optimizer.param_groups:
            group["lr"] = lr
        recipe._soft_update(restored, restored_optimizer, restored_rngs, graph, teacher,
                            words, probabilities, device="cpu", batch=8)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
        for name in ("data", "mask"):
            assert torch.equal(rngs[name].get_state(), restored_rngs[name].get_state())
        first, second = optimizer.state_dict(), restored_optimizer.state_dict()
        assert first["param_groups"] == second["param_groups"]
        for key, state in first["state"].items():
            for field in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(state[field], second["state"][key][field], rtol=0, atol=1e-7)
    assert recipe.digest(parent) == parent_digest
