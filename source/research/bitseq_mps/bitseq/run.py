"""Single-run local pretraining entrypoint with explicit identities and exact gates.

No comparison scheduler/remote service. Outputs are this experiment's generated
artifacts; snapshots preserve source when later implementation changes files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np
import psutil
import torch

from bitseq.artifacts import (
    archive_sources,
    save_checkpoint,
    source_identity,
    write_json,
)
from bitseq.evaluate import cache_logwords, denoising_excess, evaluate_exact
from bitseq.model import make_denoiser
from bitseq.oracle import FiniteGraph, task_distribution
from bitseq.train import draw_pretraining_masks, pretraining_loss


def _resources(output):
    raw = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", raw)
    if match is None:
        raise RuntimeError("cannot monitor swap")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match[2]]
    pressure = int(
        subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
        )
    )
    return dict(
        rss_bytes=psutil.Process().memory_info().rss,
        swap_bytes=float(match[1]) * scale,
        pressure_level=pressure,
        disk_free_bytes=shutil.disk_usage(output).free,
        mps_allocated_bytes=torch.mps.current_allocated_memory()
        if torch.backends.mps.is_available()
        else None,
    )


def _check_resources(now, initial):
    if now["disk_free_bytes"] < 10 * 1024**3 or now["rss_bytes"] > 2 * 1024**3:
        raise RuntimeError("resource reserve/RSS envelope exceeded")
    if (
        now["swap_bytes"] - initial["swap_bytes"] > 1024**3
        or now["pressure_level"] != 1
    ):
        raise RuntimeError("swap/pressure launch envelope exceeded; cause not assigned")


def _evaluate(model, graph, prior, output, step):
    started = time.monotonic()
    table = cache_logwords(model, graph.boards)
    summary = {}
    arrays = {}
    for order in ["ar", "random"]:
        record = evaluate_exact(graph, table, order)
        p = record["p"]
        basin = prior["basin_masks"] @ p
        summary[order] = dict(
            reward=float(p @ prior["reward"]),
            tv_prior=float(abs(p - prior["q"]).sum() / 2),
            basin_masses=basin.tolist(),
            basin_total=float(basin.sum()),
            minimum_basin=float(basin.min()),
            terminal_entropy=record["terminal_entropy"],
            conditional_entropy=record["current_conditional_entropy"],
            underflow_terminals=record["underflow_terminals"],
        )
        for key in ["p", "logp", "conditional_entropy"]:
            arrays[f"{order}_{key}"] = record[key]
    excess = denoising_excess(graph, table, prior["q"])
    decoder_tv = float(abs(arrays["ar_p"] - arrays["random_p"]).sum() / 2)
    target_basin = prior["basin_masks"] @ prior["q"]
    eligible = (
        all(
            s["tv_prior"] <= 0.05
            and np.max(np.abs(np.array(s["basin_masses"]) / target_basin - 1)) <= 0.2
            for s in summary.values()
        )
        and decoder_tv <= 0.02
        and excess["per_word_excess"] <= 0.02
    )
    np.savez_compressed(output / f"table_{step:06d}.npz", logwords=table)
    np.savez_compressed(output / f"distribution_{step:06d}.npz", **arrays)
    record = dict(
        step=step,
        decoders=summary,
        decoder_tv=decoder_tv,
        denoising=excess,
        calibrated=bool(eligible),
        evaluation_seconds=time.monotonic() - started,
        evaluation_probability_law="CPU64 softmax of device32 logits; no terminal renormalization",
    )
    write_json(output / f"eval_{step:06d}.json", record)
    return record


def run_pretraining(
    output,
    *,
    seed=100,
    device="mps",
    steps=6000,
    eval_every=500,
    batch=256,
    stop_on_calibration=True,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        schema="BS-pretrain-v1.3",
        seed=seed,
        device=device,
        steps=steps,
        eval_every=eval_every,
        batch=batch,
        lr=6e-4,
        clip=1.0,
        weight_decay=0.0,
        target_alpha=12,
        stop_on_calibration=stop_on_calibration,
    )
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    identity = source_identity()
    archive_sources(output, identity)
    initial = _resources(output)
    _check_resources(initial, initial)
    manifest = dict(
        config=config,
        config_hash=config_hash,
        source=identity,
        initial_resources=initial,
        pid=os.getpid(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        platform=platform.platform(),
        start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(output / "manifest.json", manifest)
    graph = FiniteGraph()
    prior = task_distribution(graph, 12)
    model = make_denoiser(seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    data_rng = torch.Generator().manual_seed(seed + 10_000)
    mask_rng = torch.Generator().manual_seed(seed + 20_000)
    generators = {"data": data_rng, "mask": mask_rng}
    words = torch.from_numpy(graph.boards[graph.terminals])
    probabilities = torch.from_numpy(prior["q"])
    start = time.monotonic()
    last_eval = _evaluate(model, graph, prior, output, 0)
    metadata = dict(
        config_hash=config_hash,
        source_hash=identity["hash"],
        step=0,
        samples=0,
        optimizer_steps=0,
    )
    save_checkpoint(output / "ckpt_000000.pt", model, optimizer, generators, metadata)
    status = "running"
    logpath = output / "train.jsonl"
    try:
        for step in range(1, steps + 1):
            tick = time.monotonic()
            model.train()
            indices = torch.multinomial(
                probabilities, batch, replacement=True, generator=data_rng
            )
            y = words[indices].to(device)
            mask = draw_pretraining_masks(batch, mask_rng).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pretraining_loss(model, y, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite pretraining loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if device == "mps":
                torch.mps.synchronize()
            entry = dict(
                step=step,
                samples=step * batch,
                optimizer_steps=step,
                loss=float(loss.detach().cpu()),
                preclip_norm=float(norm.detach().cpu()),
                update_seconds=time.monotonic() - tick,
            )
            if step % 25 == 0:
                entry["resources"] = _resources(output)
                _check_resources(entry["resources"], initial)
            with logpath.open("a") as log:
                log.write(json.dumps(entry, allow_nan=False) + "\n")
            metadata.update(step=step, samples=step * batch, optimizer_steps=step)
            if step % eval_every == 0 or step == steps:
                last_eval = _evaluate(model, graph, prior, output, step)
                save_checkpoint(
                    output / f"ckpt_{step:06d}.pt",
                    model,
                    optimizer,
                    generators,
                    metadata.copy(),
                )
                write_json(
                    output / "progress.json",
                    dict(
                        **metadata,
                        status="running",
                        elapsed_seconds=time.monotonic() - start,
                        last_evaluation=last_eval,
                    ),
                    overwrite=True,
                )
                print(
                    json.dumps(
                        dict(step=step, loss=entry["loss"], evaluation=last_eval)
                    ),
                    flush=True,
                )
                if stop_on_calibration and last_eval["calibrated"]:
                    status = "calibrated"
                    write_json(
                        output / "CALIBRATION.json",
                        dict(
                            **metadata,
                            evaluation=last_eval,
                            checkpoint=f"ckpt_{step:06d}.pt",
                        ),
                    )
                    break
        else:
            status = "complete" if not stop_on_calibration else "calibration_failed"
    except Exception as exc:
        write_json(output / "failure.json", dict(error=repr(exc), **metadata))
        save_checkpoint(
            output / "failure.pt", model, optimizer, generators, metadata.copy()
        )
        raise
    result = dict(
        **metadata,
        status=status,
        elapsed_seconds=time.monotonic() - start,
        last_evaluation=last_eval,
        final_resources=_resources(output),
    )
    write_json(output / "progress.json", result, overwrite=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pretrain"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pretraining(
                args.out,
                seed=args.seed,
                device=args.device,
                steps=args.steps,
                eval_every=args.eval_every,
                batch=args.batch,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
