"""Denoising weight/gradient, device/RNG and checkpoint continuation contracts."""

import copy

import numpy as np
import pytest
import torch


def test_pretraining_is_literal_masked_nelbo_and_random_masks_are_valid():
    from bitseq.model import make_denoiser
    from bitseq.train import pretraining_loss, draw_pretraining_masks

    model = make_denoiser(109)
    y = torch.tensor([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2], [1, 1, 0, 2, 3, 0]])
    mask = torch.tensor(
        [[1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    loss = pretraining_loss(model, y, mask)
    ids = torch.cat([torch.full((3, 1), 5), torch.where(mask, 4, y)], 1)
    lp = model(ids)[:, 1:].log_softmax(-1).gather(-1, y[:, :, None]).squeeze(-1)
    direct = -(lp.masked_fill(~mask, 0).sum(1) * 6 / mask.sum(1)).mean()
    torch.testing.assert_close(loss, direct, rtol=1e-6, atol=1e-7)
    a = torch.autograd.grad(loss, tuple(model.parameters()))
    b = torch.autograd.grad(direct, tuple(model.parameters()))
    torch.testing.assert_close(
        torch.cat([g.flatten() for g in a]),
        torch.cat([g.flatten() for g in b]),
        rtol=2e-5,
        atol=1e-7,
    )
    rng = torch.Generator().manual_seed(100)
    masks = draw_pretraining_masks(6000, rng)
    assert masks.shape == (6000, 6) and masks.dtype == torch.bool
    count = masks.sum(1)
    assert int(count.min()) == 1 and int(count.max()) == 6
    assert max(abs(int((count == l).sum()) - 1000) for l in range(1, 7)) < 150
    with pytest.raises(ValueError):
        pretraining_loss(model, y, torch.zeros_like(mask))


def test_actual_pretraining_cpu_mps_gradients_and_initialization_rng():
    from bitseq.model import make_denoiser
    from bitseq.train import pretraining_loss

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable: not passed")
    before = torch.mps.get_rng_state().clone()
    cpu = make_denoiser(110)
    assert torch.equal(before, torch.mps.get_rng_state())
    mps = copy.deepcopy(cpu).to("mps")
    y = torch.arange(48).reshape(8, 6) % 4
    mask = (torch.arange(48).reshape(8, 6) % 3) == 0
    ids = torch.cat([torch.full((8, 1), 5), torch.where(mask, 4, y)], 1)
    lc = cpu(ids)
    lm = mps(ids.to("mps")).cpu()
    assert (lc - lm).abs().max() < 2e-4
    losses = [
        pretraining_loss(model, y.to(device), mask.to(device))
        for model, device in [(cpu, "cpu"), (mps, "mps")]
    ]
    assert abs(float((losses[0] - losses[1].cpu()).detach())) < 5e-4
    grads = [
        torch.cat(
            [
                g.detach().cpu().flatten()
                for g in torch.autograd.grad(loss, tuple(model.parameters()))
            ]
        )
        for model, loss in zip((cpu, mps), losses)
    ]
    assert (grads[0] - grads[1]).norm() <= 1e-3 * grads[0].norm() + 1e-6


def test_atomic_checkpoint_rejects_overwrite_mismatch_and_reproduces_cpu_updates(
    tmp_path,
):
    from bitseq.model import make_denoiser
    from bitseq.train import pretraining_loss, draw_pretraining_masks
    from bitseq.artifacts import save_checkpoint, load_checkpoint

    model = make_denoiser(111)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0)
    rng = torch.Generator().manual_seed(500)
    y = torch.arange(48).reshape(8, 6) % 4

    def steps(m, o, g, n):
        for _ in range(n):
            o.zero_grad(set_to_none=True)
            loss = pretraining_loss(m, y, draw_pretraining_masks(8, g))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()

    steps(model, optimizer, rng, 5)
    path = tmp_path / "checkpoint.pt"
    metadata = {"config_hash": "test-config", "source_hash": "test-source", "step": 5}
    save_checkpoint(path, model, optimizer, {"mask": rng}, metadata)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, optimizer, {"mask": rng}, metadata)
    resumed = make_denoiser(112)
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=6e-4, weight_decay=0)
    rng2 = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, resumed, opt2, {"mask": rng2}, {"source_hash": "wrong"})
    actual = load_checkpoint(path, resumed, opt2, {"mask": rng2}, metadata)
    assert actual["step"] == 5
    steps(model, optimizer, rng, 5)
    steps(resumed, opt2, rng2, 5)
    for a, b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
    assert torch.equal(rng.get_state(), rng2.get_state())


def test_json_manifest_and_driver_smoke_are_identity_checked(tmp_path):
    from bitseq.artifacts import write_json, source_identity
    from bitseq.run import run_pretraining

    identity = source_identity()
    assert len(identity["hash"]) == 64 and "src/tdlm/model.py" in identity["files"]
    output = tmp_path / "smoke"
    result = run_pretraining(
        output,
        seed=113,
        device="cpu",
        steps=10,
        eval_every=5,
        batch=8,
        stop_on_calibration=False,
    )
    assert result["step"] == 10 and result["optimizer_steps"] == 10
    assert result["samples"] == 80 and result["status"] == "complete"
    assert (output / "manifest.json").exists() and (output / "ckpt_000010.pt").exists()
    with pytest.raises(FileExistsError):
        run_pretraining(output, seed=113, device="cpu", steps=10, eval_every=5, batch=8)
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"x": float("nan")})
