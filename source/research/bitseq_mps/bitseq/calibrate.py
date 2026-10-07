"""Bounded Q1 teacher-conditional continuation, with immutable parentage.

Same conditional expected NELBO as the registered soft-high arm. No RL or gate
relaxation. A new source epoch is explicit; original failed runs stay untouched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from bitseq.artifacts import (
    archive_sources,
    load_checkpoint,
    save_checkpoint,
    source_identity,
    write_json,
)
from bitseq.model import make_denoiser
from bitseq.oracle import FiniteGraph, task_distribution, teacher_table
from bitseq.run import _check_resources, _evaluate, _resources
from bitseq.train import draw_pretraining_masks, soft_pretraining_loss


def run_soft_continuation(
    parent,
    expected_sha256,
    output,
    *,
    device="mps",
    steps=4000,
    eval_every=500,
    batch=256,
    lr=6e-4,
    stop_on_calibration=True,
):
    if (
        device not in ("cpu", "mps")
        or steps < 1
        or batch < 1
        or eval_every < 1
        or any(not isinstance(v, int) for v in (steps, batch, eval_every))
        or not np.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("positive integer budgets and CPU/MPS required")
    parent, output = Path(parent), Path(output)
    if hashlib.sha256(parent.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Parent digest mismatch")
    record = torch.load(parent, weights_only=True, map_location="cpu")
    if record["metadata"].get("estimator") != "soft":
        raise ValueError("Requires a documented soft-estimator parent")
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    archive_sources(output, identity)
    initial = _resources(output)
    _check_resources(initial, initial)
    model = make_denoiser(100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0)
    generators = {"data": torch.Generator(), "mask": torch.Generator()}
    previous = load_checkpoint(parent, model, optimizer, generators, record["metadata"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    total_parent_steps = previous.get(
        "total_training_steps",
        previous.get("parent_training_steps", 0) + previous["step"],
    )
    parent_samples = previous["samples"]
    config = dict(
        schema="BS-Q1-soft-extension-v1",
        parent=str(parent),
        parent_sha256=expected_sha256,
        source=identity,
        device=device,
        estimator="soft",
        lr=lr,
        steps=steps,
        batch=batch,
        eval_every=eval_every,
        stop_on_calibration=stop_on_calibration,
        pid=os.getpid(),
        start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        initial_resources=initial,
        parent_total_training_steps=total_parent_steps,
    )
    write_json(output / "manifest.json", config)
    graph = FiniteGraph()
    truth = task_distribution(graph, 12)
    teacher, _ = teacher_table(graph, truth["q"])
    words = torch.tensor(graph.boards[graph.terminals])
    probabilities = torch.tensor(truth["q"])
    started = time.monotonic()
    metadata = dict(
        source_hash=identity["hash"],
        parent_sha256=expected_sha256,
        step=0,
        optimizer_steps=0,
        samples=parent_samples,
        total_training_steps=total_parent_steps,
        estimator="soft",
    )
    save_checkpoint(output / "ckpt_000000.pt", model, optimizer, generators, metadata)
    last_eval = _evaluate(model, graph, truth, output, 0)
    print(json.dumps(dict(started=config)), flush=True)
    status = "running"
    try:
        for step in range(1, steps + 1):
            model.train()
            indices = torch.multinomial(
                probabilities, batch, replacement=True, generator=generators["data"]
            )
            y = words[indices].to(device)
            masks = draw_pretraining_masks(batch, generators["mask"]).to(device)
            boards = torch.where(masks, -1, y)
            rows = graph.index(boards.cpu().numpy())
            targets = torch.as_tensor(
                np.exp(teacher[rows]), dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = soft_pretraining_loss(model, boards, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite soft continuation loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            metadata.update(
                step=step,
                optimizer_steps=step,
                total_training_steps=total_parent_steps + step,
                samples=parent_samples + step * batch,
            )
            entry = dict(
                **metadata,
                loss=float(loss.detach().cpu()),
                preclip_norm=float(norm.detach().cpu()),
            )
            if step % 25 == 0:
                entry["resources"] = _resources(output)
                _check_resources(entry["resources"], initial)
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if step % eval_every == 0 or step == steps:
                last_eval = _evaluate(model, graph, truth, output, step)
                checkpoint = output / f"ckpt_{step:06d}.pt"
                save_checkpoint(
                    checkpoint, model, optimizer, generators, metadata.copy()
                )
                result = dict(
                    **metadata,
                    evaluation=last_eval,
                    elapsed_seconds=time.monotonic() - started,
                )
                write_json(output / "progress.json", result, overwrite=True)
                print(json.dumps(result), flush=True)
                if last_eval["calibrated"] and stop_on_calibration:
                    status = "calibrated"
                    write_json(
                        output / "CALIBRATION.json",
                        dict(
                            **result,
                            checkpoint=checkpoint.name,
                            checkpoint_sha256=hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                            factory_sha256=identity["files"][
                                "research/bitseq_mps/bitseq/model.py"
                            ],
                            backbone_sha256=identity["files"]["src/tdlm/model.py"],
                        ),
                    )
                    break
        else:
            status = "calibration_failed" if stop_on_calibration else "complete"
    except BaseException as error:
        write_json(output / "failure.json", dict(**metadata, error=repr(error)))
        save_checkpoint(
            output / "failure.pt", model, optimizer, generators, metadata.copy()
        )
        raise
    result = dict(
        **metadata,
        status=status,
        evaluation=last_eval,
        elapsed_seconds=time.monotonic() - started,
        final_resources=_resources(output),
    )
    write_json(output / "progress.json", result, overwrite=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()
    run_soft_continuation(
        args.parent, args.parent_sha256, args.out, lr=args.lr, steps=args.steps
    )
