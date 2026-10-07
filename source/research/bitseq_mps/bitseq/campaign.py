"""Single-run primary RL lifecycle and exact checkpoint measurements.

Source epochs and recipe identities are explicit; calibrated input is mandatory.
No remote compute, method tuning, scheduler, new sampler or loss reimplementation.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile
import platform

import numpy as np
import torch

from bitseq.artifacts import archive_sources, source_identity, write_json
from bitseq.evaluate import cache_logwords, evaluate_exact
from bitseq.model import make_denoiser
from bitseq.oracle import FiniteGraph, task_distribution
from bitseq.run import _check_resources, _evaluate, _resources
from bitseq.updates import RLState


def check_storage(directory, cap_bytes=2 * 1024**3):
    seen = set()
    size = 0
    for p in Path(directory).rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        stat = p.stat()
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            size += stat.st_size
            seen.add(key)
    if size > cap_bytes:
        raise RuntimeError(
            "Campaign storage cap exceeded; preserve evidence, stop new work"
        )
    return size


def load_calibrated_base(directory, device="mps"):
    directory = Path(directory)
    if not (directory / "CALIBRATION.json").exists():
        raise ValueError("No accepted calibration artifact")
    calibration = json.loads((directory / "CALIBRATION.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if calibration.get("source_hash") != manifest["source"]["hash"]:
        raise ValueError("Calibration source epoch mismatch")
    path = directory / calibration["checkpoint"]
    if (
        path.parent != directory
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != calibration["checkpoint_sha256"]
    ):
        raise ValueError("Calibrated checkpoint identity mismatch")
    identity = source_identity()
    for name in [
        "research/bitseq_mps/bitseq/model.py",
        "src/tdlm/model.py",
        "research/bitseq_mps/bitseq/env.py",
        "research/bitseq_mps/bitseq/oracle.py",
    ]:
        if manifest["source"]["files"][name] != identity["files"][name]:
            raise ValueError("Calibrated model/task source mismatch")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["metadata"].get("source_hash") != calibration["source_hash"]:
        raise ValueError("Checkpoint/calibration source epoch mismatch")
    model = make_denoiser(100).to(device)
    model.load_state_dict(record["model"])
    graph = FiniteGraph()
    prior = task_distribution(graph, 12)
    # Re-evaluate actual checkpoint, not only a copied calibrated=True field.
    with tempfile.TemporaryDirectory(prefix="bitseq-base-verify-") as scratch:
        actual = _evaluate(model, graph, prior, Path(scratch), 0)
        table = np.load(Path(scratch) / "table_000000.npz")["logwords"].copy()
    if not actual["calibrated"]:
        raise ValueError("Actual checkpoint fails calibration gates")
    model.eval().requires_grad_(False)
    return model, table, calibration, graph, prior


def measure_checkpoint(graph, table, reference, prior):
    target = task_distribution(graph, 24)
    records, arrays = {}, {}
    for order in ("ar", "random"):
        evaluation = evaluate_exact(graph, table, order, reference=reference)
        p, logp = evaluation["p"], evaluation["logp"]
        basins = prior["basin_masks"] @ p
        total = basins.sum()
        qR = target["q"]
        terminal_kl_target = float(p @ (logp - np.log(qR)))
        ent = evaluation["current_conditional_entropy"]
        records[order] = dict(
            reward=float(p @ prior["reward"]),
            basin_masses=basins.tolist(),
            basin_total=float(total),
            minimum_basin=float(basins.min()),
            basin_entropy=None
            if total == 0
            else float(
                -np.sum(
                    (basins[basins > 0] / total) * np.log(basins[basins > 0] / total)
                )
            ),
            outside_mass=float(1 - total),
            peak_mass=float(p[prior["reward"] == 1].sum()),
            distance_mass={
                str(d): float(
                    p[
                        np.isclose(
                            prior["reward"], math.exp(-2 * d), rtol=1e-10, atol=0
                        )
                    ].sum()
                )
                for d in range(6)
            },
            tv_prior=float(abs(p - prior["q"]).sum() / 2),
            tv_target=float(abs(p - qR).sum() / 2),
            terminal_kl_target=terminal_kl_target,
            terminal_entropy=evaluation["terminal_entropy"],
            conditional_entropy=ent,
            fixed_reference_conditional_entropy=evaluation[
                "fixed_reference_conditional_entropy"
            ],
            path_kl_ref_to_current=evaluation["path_kl_ref_to_current"],
            path_kl_current_to_ref=evaluation["path_kl_current_to_ref"],
            terminal_kl_current_to_ref=evaluation["terminal_kl_current_to_ref"],
            terminal_kl_ref_to_current=evaluation["terminal_kl_ref_to_current"],
            canonical_joint_kl=None
            if order == "ar"
            else terminal_kl_target + math.lgamma(7) - ent,
            expected_distinct_basins={
                str(k): float(
                    np.count_nonzero(basins >= 1)
                    + np.sum(-np.expm1(k * np.log1p(-basins[basins < 1])))
                )
                for k in (1, 5, 16, 64, 256)
            },
            normalization_error=float(abs(p.sum() - 1)),
            underflow_terminals=evaluation["underflow_terminals"],
        )
        for key in (
            "p",
            "logp",
            "conditional_entropy",
            "logpath_mean",
            "logpath_second",
        ):
            arrays[order + "_" + key] = evaluation[key]
    return records, arrays


def run_rl(
    base_directory,
    output,
    *,
    method,
    device="mps",
    seed=201,
    lr=3e-4,
    beta=0.3,
    steps=1500,
    eval_every=100,
    audit_tables=True,
    resume_from=None,
):
    if (
        method not in RLState.METHODS
        or steps < 1
        or eval_every < 1
        or not isinstance(steps, int)
        or not isinstance(eval_every, int)
        or not np.isfinite(lr)
        or lr <= 0
        or not np.isfinite(beta)
        or beta < 0
    ):
        raise ValueError("Invalid method/budget/optimizer")
    storage_root = Path(__file__).resolve().parents[1] / "runs"
    check_storage(storage_root)
    launch_resources = _resources(storage_root)
    _check_resources(launch_resources, launch_resources)
    base, reference, calibration, graph, prior = load_calibrated_base(
        base_directory, device
    )
    output = Path(output)
    if resume_from is None:
        output.mkdir(parents=True, exist_ok=False)
        epoch_directory = output
    else:
        origin = json.loads((output / "manifest.json").read_text())
        for name, value in dict(
            method=method,
            device=device,
            seed=seed,
            lr=lr,
            beta=beta,
            steps=steps,
            eval_every=eval_every,
        ).items():
            if origin[name] != value:
                raise ValueError("Resume recipe mismatch: " + name)
        if origin["base_checkpoint_sha256"] != calibration["checkpoint_sha256"]:
            raise ValueError("Resume reference mismatch")
        current = source_identity()
        # Only lifecycle code changes here; sampler/loss/model/precision sources must match.
        unused_or_driver = {
            "research/bitseq_mps/bitseq/campaign.py",
            "research/bitseq_mps/bitseq/audit.py",
            "research/bitseq_mps/bitseq/path_analysis.py",
            "research/bitseq_mps/bitseq/doob.py",
        }
        for name, digest in origin["source"]["files"].items():
            if name not in unused_or_driver and current["files"].get(name) != digest:
                raise ValueError("Resume mathematical source mismatch: " + name)
        epoch_directory = (
            output / f"continuation_{len(list(output.glob('continuation_*'))) + 1:02d}"
        )
        epoch_directory.mkdir(exist_ok=False)
    identity = source_identity()
    archive_sources(epoch_directory, identity)
    initial = _resources(output)
    _check_resources(initial, initial)
    state = RLState(base, method, device=device, seed=seed, lr=lr, beta=beta)
    resume_record = None
    if resume_from is not None:
        resume_record = torch.load(resume_from, weights_only=True, map_location="cpu")
        state.restore(resume_from, source_hash=resume_record["metadata"]["source_hash"])
        if state.steps >= steps:
            raise ValueError("Resume checkpoint already at/beyond endpoint")
    protocol = Path(__file__).resolve().parents[1] / "PROTOCOL.md"
    contract_names = (
        "PROTOCOL.md",
        "PROPOSAL.md",
        "TWO_DAY_PLAN.md",
        "reports/06_CALIBRATION_EXTENSION_PROTOCOL.md",
        "reports/07_CALIBRATED_REFERENCE.md",
        "reports/11_LR_SELECTION_AND_PANEL.md",
    )
    contract_root = Path(__file__).resolve().parents[1]
    contract_hashes = {
        name: hashlib.sha256((contract_root / name).read_bytes()).hexdigest()
        for name in contract_names
    }
    with zipfile.ZipFile(epoch_directory / "source_snapshot.zip", "a") as archive:
        for name in contract_names:
            archive.write(contract_root / name, "research/bitseq_mps/" + name)
    base_table_path = (
        Path(base_directory) / f"table_{calibration['evaluation']['step']:06d}.npz"
    )
    stored_reference = np.load(base_table_path)["logwords"]
    manifest = dict(
        schema="BS-RL-v1",
        method=method,
        seed=seed,
        lr=lr,
        beta=beta,
        device=device,
        steps=steps,
        eval_every=eval_every,
        source=identity,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        research_contract_sha256=contract_hashes,
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=str(np.__version__),
        ),
        evaluation_probability_law="CPU64 softmax of device32 logits, no terminal renormalization",
        base_directory=str(base_directory),
        base_checkpoint_sha256=calibration["checkpoint_sha256"],
        base_reference_table=dict(
            path=str(base_table_path),
            sha256=hashlib.sha256(base_table_path.read_bytes()).hexdigest(),
        ),
        reference_recomputed_device=device,
        reference_table_max_logp_deviation=float(
            np.max(abs(reference - stored_reference))
        ),
        train_order=state.order,
        mask_K=4 if method.startswith("trafl") else None,
        utility="R=exp(-2*d)",
        loss_reward_interface="raw logR"
        if method == "tb_random"
        else "R, grouped according to method",
        groups=16,
        group_size=5,
        pid=__import__("os").getpid(),
        start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        initial_resources=initial,
    )
    if resume_from is not None:
        manifest.update(
            resume_from=str(resume_from),
            resume_sha256=hashlib.sha256(Path(resume_from).read_bytes()).hexdigest(),
            resumed_step=state.steps,
            origin_source_hash=origin["source"]["hash"],
            lifecycle_amendment="committed checkpoint resume; mathematical update sources verified unchanged",
        )
    write_json(epoch_directory / "manifest.json", manifest)
    started = time.monotonic()

    def evaluate(step):
        tick = time.monotonic()
        table = cache_logwords(state.policy, graph.boards)
        summary, arrays = measure_checkpoint(graph, table, reference, prior)
        np.savez_compressed(output / f"distribution_{step:06d}.npz", **arrays)
        if audit_tables and step in (500, 1500):
            np.savez_compressed(output / f"table_{step:06d}.npz", logwords=table)
        z = None
        if state.gfn is not None:
            z = float(state.gfn.logZ.detach().cpu())
        result = dict(
            step=step,
            decoders=summary,
            logZ=z,
            logZ_error=None if z is None else z - task_distribution(graph, 24)["logZ"],
            evaluation_seconds=time.monotonic() - tick,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(output / f"eval_{step:06d}.json", result)
        print(json.dumps(dict(method=method, seed=seed, **result)), flush=True)
        return result

    if resume_from is None:
        state.save(output / "ckpt_000000.pt", source_hash=identity["hash"])
        latest = evaluate(0)
    else:
        recorded = sorted(
            p
            for p in output.glob("eval_*.json")
            if int(p.stem.split("_")[-1]) <= state.steps
        )
        latest = json.loads(recorded[-1].read_text())
    try:
        for step in range(state.steps + 1, steps + 1):
            metrics = state.step()
            if step % 25 == 0:
                metrics["campaign_storage_bytes"] = check_storage(storage_root)
                metrics["resources"] = _resources(output)
                _check_resources(metrics["resources"], initial)
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step % eval_every == 0 or step == steps:
                latest = evaluate(step)
                state.save(
                    output / "latest.pt", source_hash=identity["hash"], overwrite=True
                )
                if step % 500 == 0 or step == steps:
                    os.link(output / "latest.pt", output / f"ckpt_{step:06d}.pt")
                write_json(
                    output / "progress.json",
                    dict(status="running", evaluation=latest),
                    overwrite=True,
                )
        if state._identity(identity["hash"])[
            "reference_sha256"
        ] != state_identity_reference(base):
            raise RuntimeError("Frozen reference changed")
    except BaseException as error:
        state.save(epoch_directory / "failure.pt", source_hash=identity["hash"])
        write_json(
            epoch_directory / "failure.json",
            dict(step=state.steps, error=repr(error), resources=_resources(output)),
        )
        raise
    result = dict(
        status="complete",
        steps=state.steps,
        samples=state.samples,
        evaluation=latest,
        forward_counts=state.forward_counts,
        elapsed_seconds=time.monotonic() - started,
        elapsed_seconds_scope="current execution epoch; prior epoch cost remains in its records",
        resumed_from_step=None
        if resume_record is None
        else resume_record["metadata"]["step"],
        final_resources=_resources(output),
    )
    write_json(output / "progress.json", result, overwrite=True)
    return result


def state_identity_reference(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=RLState.METHODS, required=True)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    run_rl(
        args.base,
        args.out,
        method=args.method,
        seed=args.seed,
        lr=args.lr,
        beta=args.beta,
        steps=args.steps,
    )
