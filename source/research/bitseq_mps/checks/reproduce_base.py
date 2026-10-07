"""Fixed independent q12 base recipe; never selects an earlier checkpoint.

Hard training is the unchanged bitseq.run.run_pretraining. The short soft loop
below is attributed to bitseq.calibrate.run_soft_continuation, with explicit
hard/soft parent provenance and cumulative counters. Existing loss/mask/teacher,
checkpoint, evaluation and resource primitives are reused unchanged. No source
rewriting, library edits, alternate model or generic training framework.

Import/plan uses only stdlib. Running the neural recipe is a separate command.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[3]
SUBTREE = ROOT / "research/bitseq_mps"
SCHEMA = "BS-independent-base-fixed-6000-6000-500-v1"
CAP_BYTES = 2 * 1024**3
STAGES = {
    "hard": dict(kind="hard", steps=6000, total=6000, lr=6e-4),
    "soft-high": dict(kind="soft", steps=6000, total=12000, lr=6e-4),
    "soft-low": dict(kind="soft", steps=500, total=12500, lr=6e-5),
}
# Source bytes deciding task, prior, initialization, masks, loss, gates and
# checkpoint semantics. Unused RL modules are archived but not recipe pins.
PINS = {
    "research/bitseq_mps/bitseq/run.py": "772b73afd076b8398509e26ebcf574ae8632e7e54193d38eb281c9cfd3aa7515",
    "research/bitseq_mps/bitseq/model.py": "52ce9a82d36aaf15446ec0d0c8f93e608b8e31bc946ecd9c85c5eb65d7b18180",
    "research/bitseq_mps/bitseq/train.py": "41829174ab3495ead6b44ca9505398a047d21d0ac3ecbbed6371f195d3ec843a",
    "research/bitseq_mps/bitseq/artifacts.py": "30b08fb217e473c66e2e8e3d52135596c1c1b3b354218881330764205c95618d",
    "research/bitseq_mps/bitseq/oracle.py": "26dfc4c2c8655e58ff7ca2640457f962a62082223c42fe3208301998231efd16",
    "research/bitseq_mps/bitseq/evaluate.py": "454f3e5d14c83daac59a30089d08a3eab7a474d93d134585aa67aa245cf86143",
    "research/bitseq_mps/bitseq/env.py": "b5fb9bfe08a8df059569b986d8cde293363d6e4f9f054f764f16269607cace5b",
    "src/tdlm/model.py": "7556870309ce21dfffd2ce1755ec150d27950d7ac66798c2cd239e3a805f23f6",
    "src/tdlm/mdm.py": "377e692ad42f821fcee86e5f34105745ac563273cd761e3801101e0afaf2c6c6",
    "research/bitseq_mps/setup_manifest.json": "2ba9445dd5394cbf1c4902c86af81482c6198a8611e6b834840d45a37eed024a",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_hard_config(config, seed):
    expected = dict(seed=seed, steps=6000, eval_every=6000, batch=256, lr=6e-4,
                    clip=1.0, weight_decay=0.0, target_alpha=12,
                    stop_on_calibration=False, schema="BS-pretrain-v1.3")
    if seed not in (101, 102) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Hard initializer/config is not the fixed independent recipe")
    if config.get("device") not in ("cpu", "mps"):
        raise ValueError("Unsupported hard device")


def validate_parent_metadata(metadata, *, seed, stage, recipe_hash, source_hash):
    spec = STAGES[stage]
    expected = dict(seed=seed, kind=spec["kind"], stage=stage,
                    recipe_hash=recipe_hash, source_hash=source_hash,
                    step=spec["steps"], optimizer_steps=spec["steps"],
                    total_training_steps=spec["total"], samples=spec["total"] * 256)
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Parent kind/seed/provenance/cumulative counter mismatch")


def final_status(step, total_training_steps, samples, evaluation):
    if (step, total_training_steps, samples) != (500, 12500, 3200000):
        raise ValueError("Only the fixed final500 checkpoint is eligible")
    if type(evaluation.get("calibrated")) is not bool:
        raise ValueError("Missing actual final calibration result")
    return "calibrated" if evaluation["calibrated"] else "calibration_failed"


def finalize_result(receipt):
    status = final_status(receipt["step"], receipt["total_training_steps"],
                          receipt["samples"], receipt["evaluation"])
    return dict(receipt, status=status,
                factory_sha256=PINS["research/bitseq_mps/bitseq/model.py"],
                backbone_sha256=PINS["src/tdlm/model.py"])


def check_sources():
    for name, expected in PINS.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen recipe source changed: {name}; review before repinning")


def storage_bytes(path):
    # Same inode semantics as campaign.check_storage, without neural imports.
    seen, size = set(), 0
    for path in Path(path).rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            size += stat.st_size
            seen.add(key)
    return size


def check_storage(storage_root):
    size = storage_bytes(storage_root)
    if size > CAP_BYTES:
        raise RuntimeError("2GiB campaign storage cap exceeded; preserve failed evidence")
    return size


def _read_parent(parent, expected_sha256, *, seed, stage, recipe_hash):
    """Validate native hard metadata by its manifest, soft by explicit lineage.

    The hard artifact is never relabelled or rewritten: its seed/kind/recipe are
    established by the immutable wrapper receipt and native hard config/hash.
    """
    import torch

    parent = Path(parent)
    if digest(parent) != expected_sha256:
        raise ValueError("Parent checkpoint digest mismatch")
    receipt = json.loads((parent.parent / "STAGE.json").read_text())
    manifest = json.loads((parent.parent / "manifest.json").read_text())
    if receipt.get("checkpoint") != parent.name or receipt.get("checkpoint_sha256") != expected_sha256:
        raise ValueError("Parent receipt/checkpoint mismatch")
    record = torch.load(parent, weights_only=True, map_location="cpu")
    meta = record["metadata"]
    source_hash = manifest["source"]["hash"]
    if meta.get("source_hash") != source_hash:
        raise ValueError("Parent checkpoint/source epoch mismatch")
    for name, expected in PINS.items():
        if manifest["source"]["files"].get(name) != expected:
            raise ValueError(f"Parent model/task/loss/gate source mismatch: {name}")
    actual = dict(meta)
    if stage == "hard":
        validate_hard_config(manifest["config"], seed)
        if meta.get("config_hash") != json_hash(manifest["config"]):
            raise ValueError("Hard checkpoint/config mismatch")
        actual.update(seed=seed, kind="hard", stage="hard", recipe_hash=recipe_hash,
                      total_training_steps=meta["step"])
    validate_parent_metadata(actual, seed=seed, stage=stage,
                             recipe_hash=recipe_hash, source_hash=source_hash)
    for key in ("seed", "kind", "stage", "recipe_hash", "total_training_steps", "samples"):
        if receipt.get(key) != actual[key]:
            raise ValueError(f"Parent receipt provenance mismatch: {key}")
    if set(record["generators"]) != {"data", "mask"}:
        raise ValueError("Parent data/mask RNG identities mismatch")
    groups = record["optimizer"]["param_groups"]
    if len(groups) != 1 or any(groups[0].get(k) != v for k, v in
                             dict(lr=STAGES[stage]["lr"], betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0).items()):
        raise ValueError("Parent Adam recipe mismatch")
    states = record["optimizer"]["state"]
    if not states or len(states) != len(groups[0]["params"]):
        raise ValueError("Missing Adam parameter moments")
    for state in states.values():
        if int(state["step"].item()) != STAGES[stage]["total"]:
            raise ValueError("Parent Adam update count mismatch")
        if any(not bool(torch.isfinite(state[k]).all()) for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Nonfinite Adam moments")
    return record, actual


def _soft_stage(parent, expected_sha256, output, *, seed, stage, parent_stage,
                recipe_hash, identity, device, storage_root):
    """Same loop as calibrate.run_soft_continuation; fixed final observation.

    Integrate masked labels under q12 conditionals, not a new expected NELBO.
    Adam moments and both CPU context RNG streams carry across both transitions.
    """
    import numpy as np
    import torch
    from bitseq.artifacts import archive_sources, load_checkpoint, save_checkpoint, write_json
    from bitseq.model import make_denoiser
    from bitseq.oracle import FiniteGraph, task_distribution, teacher_table
    from bitseq.run import _check_resources, _evaluate, _resources

    record, previous = _read_parent(parent, expected_sha256, seed=seed,
                                    stage=parent_stage, recipe_hash=recipe_hash)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output, identity)
    initial = _resources(output)
    _check_resources(initial, initial)
    spec = STAGES[stage]
    model = make_denoiser(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0)
    generators = {"data": torch.Generator(), "mask": torch.Generator()}
    load_checkpoint(parent, model, optimizer, generators, record["metadata"])
    for group in optimizer.param_groups:
        group["lr"] = spec["lr"]
    graph = FiniteGraph()
    truth = task_distribution(graph, 12)
    teacher, _ = teacher_table(graph, truth["q"])
    words, probabilities = torch.tensor(graph.boards[graph.terminals]), torch.tensor(truth["q"])
    config = dict(schema=SCHEMA, seed=seed, kind="soft", stage=stage,
                  steps=spec["steps"], batch=256, lr=spec["lr"], clip=1.0,
                  stop_on_calibration=False, eval_every=spec["steps"],
                  recipe_hash=recipe_hash, source=identity, device=device,
                  parent=str(parent), parent_sha256=expected_sha256,
                  parent_total_training_steps=previous["total_training_steps"],
                  pid=os.getpid(), initial_resources=initial)
    write_json(output / "manifest.json", config)
    metadata = dict(seed=seed, kind="soft", estimator="soft", stage=stage,
                    recipe_hash=recipe_hash, source_hash=identity["hash"],
                    parent_sha256=expected_sha256, step=0, optimizer_steps=0,
                    total_training_steps=previous["total_training_steps"],
                    samples=previous["samples"])
    save_checkpoint(output / "ckpt_000000.pt", model, optimizer, generators, metadata.copy())
    started = time.monotonic()
    try:
        for step in range(1, spec["steps"] + 1):
            loss, norm = _soft_update(model, optimizer, generators, graph, teacher,
                                      words, probabilities, device=device)
            metadata.update(step=step, optimizer_steps=step,
                            total_training_steps=previous["total_training_steps"] + step,
                            samples=previous["samples"] + step * 256)
            entry = dict(**metadata, loss=float(loss.detach().cpu()),
                         preclip_norm=float(norm.detach().cpu()))
            if step % 25 == 0:
                entry["resources"] = _resources(output)
                _check_resources(entry["resources"], initial)
                check_storage(storage_root)
                check_sources()
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        evaluation = _evaluate(model, graph, truth, output, spec["steps"])
        checkpoint = output / f"ckpt_{spec['steps']:06d}.pt"
        save_checkpoint(checkpoint, model, optimizer, generators, metadata.copy())
        result = dict(**metadata, status="complete", evaluation=evaluation,
                      checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint),
                      elapsed_seconds=time.monotonic() - started, final_resources=_resources(output))
        _check_resources(result["final_resources"], initial)
        check_storage(storage_root)
        write_json(output / "progress.json", result)
        write_json(output / "STAGE.json", result)
        return checkpoint, result
    except BaseException as error:
        write_json(output / "failure.json", dict(**metadata, error=repr(error)))
        save_checkpoint(output / "failure.pt", model, optimizer, generators, metadata.copy())
        raise


def _soft_update(model, optimizer, generators, graph, teacher, words, probabilities,
                 *, device, batch=256):
    """The attributed single step, also used by the deferred neural replay gate."""
    import numpy as np
    import torch
    from bitseq.train import draw_pretraining_masks, soft_pretraining_loss

    model.train()
    indices = torch.multinomial(probabilities, batch, replacement=True,
                                generator=generators["data"])
    y = words[indices].to(device)
    masks = draw_pretraining_masks(batch, generators["mask"]).to(device)
    boards = torch.where(masks, -1, y)
    rows = graph.index(boards.cpu().numpy())
    targets = torch.as_tensor(np.exp(teacher[rows]), dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = soft_pretraining_loss(model, boards, targets)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite soft continuation loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss, norm


def run_independent_base(output, *, seed, device="mps", storage_root=None):
    """Run exactly12500 updates; failed final gates end this base, without rescue."""
    if seed not in (101, 102) or device not in ("cpu", "mps"):
        raise ValueError("Independent seeds101/102 and CPU/MPS only")
    check_sources()
    output = Path(output).resolve()
    storage_root = Path(storage_root or SUBTREE / "runs").resolve()
    if not output.is_relative_to(storage_root):
        raise ValueError("Output must lie inside the explicitly capped storage root")
    storage_root.mkdir(parents=True, exist_ok=True)
    check_storage(storage_root)
    if storage_bytes(storage_root) + 160 * 1024**2 > CAP_BYTES:
        raise RuntimeError("Insufficient2GiB artifact headroom for one fixed base")
    from bitseq.artifacts import archive_sources, source_identity, write_json
    from bitseq.run import _check_resources, _resources, run_pretraining

    initial = _resources(storage_root)
    _check_resources(initial, initial)
    identity = source_identity()
    for path in (Path(__file__).resolve(), SUBTREE / "reports/15_INDEPENDENT_BASE_PROTOCOL.md"):
        identity["files"][str(path.relative_to(ROOT))] = digest(path)
    identity["hash"] = json_hash(identity["files"])
    recipe = dict(schema=SCHEMA, seed=seed, device=device, stages=STAGES, batch=256,
                  source=identity, final_eligibility_only=True,
                  observation_schedule="initial/final hard; initial snapshot/final eval soft stages",
                  total_training_steps=12500, total_terminal_draws=3200000)
    recipe_hash = json_hash(recipe)
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output, identity)
    write_json(output / "RECIPE.json", dict(**recipe, recipe_hash=recipe_hash,
                                          pid=os.getpid(), initial_resources=initial))
    try:
        hard = output / "hard"
        result = run_pretraining(hard, seed=seed, device=device, steps=6000,
                                 eval_every=6000, batch=256, stop_on_calibration=False)
        manifest = json.loads((hard / "manifest.json").read_text())
        validate_hard_config(manifest["config"], seed)
        checkpoint = hard / "ckpt_006000.pt"
        receipt = dict(**result, seed=seed, kind="hard", stage="hard", recipe_hash=recipe_hash,
                       total_training_steps=6000, checkpoint=checkpoint.name,
                       checkpoint_sha256=digest(checkpoint))
        write_json(hard / "STAGE.json", receipt)
        for stage, parent_stage in (("soft-high", "hard"), ("soft-low", "soft-high")):
            check_sources()
            checkpoint, receipt = _soft_stage(
                checkpoint, receipt["checkpoint_sha256"], output / stage,
                seed=seed, stage=stage, parent_stage=parent_stage,
                recipe_hash=recipe_hash, identity=identity, device=device, storage_root=storage_root)
        check_sources()
        result = finalize_result(receipt)
        # Adjacent output remains compatible with campaign.load_calibrated_base,
        # whose own reload performs a fresh actual-checkpoint evaluation again.
        if result["status"] == "calibrated":
            write_json(checkpoint.parent / "CALIBRATION.json", result)
        write_json(output / "RESULT.json", dict(**result, final_directory=str(checkpoint.parent),
                                              campaign_storage_bytes=check_storage(storage_root)))
        return result
    except BaseException as error:
        write_json(output / "FAILURE.json", dict(seed=seed, recipe_hash=recipe_hash,
                                               error=repr(error), status="unfinished"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--seed", type=int, choices=(101, 102), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--storage-root", default=str(SUBTREE / "runs"))
    args = parser.parse_args()
    if args.command == "plan":
        check_sources()
        print(json.dumps(dict(schema=SCHEMA, seed=args.seed, out=args.out, stages=STAGES,
                              terminal_draws=3200000, final_gate="tail500 actual evaluation only",
                              source_pins_verified=True, neural_recipe_verified=False), indent=2))
    else:
        print(json.dumps(run_independent_base(args.out, seed=args.seed, device=args.device,
                                               storage_root=args.storage_root), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
