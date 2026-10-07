"""Read-only release integrity and optional actual-weight finite evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile


IGNORED = {".git", "__pycache__", ".pytest_cache", "work", ".venv", ".DS_Store"}


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_inventory(root):
    root = Path(root).resolve()
    record = json.loads((root / "inventory.json").read_text())
    if record["schema"] != "bitseq-export-inventory-v1":
        raise ValueError("inventory schema mismatch")
    expected = record["files"]
    for name in expected:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or not p.parts or str(p) != name:
            raise ValueError(f"unsafe inventory path: {name}")
        if any(x in IGNORED for x in p.parts) or name == "inventory.json":
            raise ValueError(f"unsafe excluded inventory path: {name}")
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(x in IGNORED for x in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink prohibited: {relative}")
        if path.is_file() and str(relative) != "inventory.json":
            actual.add(str(relative))
    if actual != set(expected):
        raise ValueError(
            f"inventory file set mismatch: missing={set(expected) - actual}, "
            f"extra={actual - set(expected)}"
        )
    for name, item in expected.items():
        p = root / name
        if p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
            raise ValueError(f"bytes/hash mismatch: {name}")
    return len(actual)


def _identities(value):
    if isinstance(value, dict):
        if set(value) >= {"files", "hash"} and isinstance(value["files"], dict):
            yield value
        for child in value.values():
            yield from _identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identities(child)


def verify_clock(values, expected):
    values = list(values)
    if len(values) != 30 or any(
        not math.isfinite(x) or x != expected or x != int(x) for x in values
    ):
        raise ValueError("noninteger, inconsistent or unexpected Adam clock")
    return expected


def verify_sources(root):
    count = 0
    for archive in (root / "artifacts").rglob("source_snapshot.zip"):
        identities = []
        entrypoints = {}
        for p in archive.parent.glob("*.json"):
            record = json.loads(p.read_text())
            identities.extend(_identities(record))
            if p.name == "campaign.json" and "entrypoint_sha256" in record:
                entrypoints["research/bitseq_mps/checks/calibration_continue.py"] = (
                    record["entrypoint_sha256"]
                )
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError(f"CRC failure: {archive}")
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate source ZIP entry")
            for name in names:
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("unsafe source ZIP member")
            matching = [
                i
                for i in identities
                if set(i["files"]) | set(entrypoints) == set(names)
            ]
            if not matching:
                raise ValueError(f"no archive source identity: {archive}")
            for identity in matching:
                h = hashlib.sha256(
                    json.dumps(identity["files"], sort_keys=True).encode()
                ).hexdigest()
                if h != identity["hash"]:
                    raise ValueError("source identity hash mismatch")
                for name, sha in (identity["files"] | entrypoints).items():
                    if hashlib.sha256(z.read(name)).hexdigest() != sha:
                        raise ValueError(f"archived source mismatch: {name}")
        count += 1
    archive = root / "artifacts/BS-independent-base101/source_snapshot.zip"
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if (root / "source" / name).read_bytes() != z.read(name):
                raise ValueError(f"extracted source mismatch: {name}")
    setup = json.loads(
        (root / "source/research/bitseq_mps/setup_manifest.json").read_text()
    )
    for name, sha in setup["source_python_sha256"].items():
        if digest(root / "vendor/gfn" / name) != sha:
            raise ValueError(f"vendored gfn mismatch: {name}")
    return {
        "source_archives": count,
        "framework_python_files": len(setup["source_python_sha256"]),
    }


def verify_neural(root, device):
    # Only dependency overlays may come from the calling environment.
    sys.path[:0] = [
        str(root / "vendor"),
        str(root / "source/research/bitseq_mps"),
        str(root / "source/src"),
    ]
    import numpy as np
    import torch
    import bitseq.model as model_module
    import tdlm.model as backbone_module
    import gfn
    from bitseq.evaluate import cache_logwords, denoising_excess, evaluate_exact
    from bitseq.oracle import FiniteGraph, task_distribution

    for module in (model_module, backbone_module, gfn):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(f"import escaped export: {module.__name__}")
    facts = json.loads((root / "facts.json").read_text())
    checkpoints = list((root / "artifacts").rglob("*.pt"))
    by_hash = {digest(p): p for p in checkpoints}
    lineage = []
    for endpoint in facts["stage_endpoints"]:
        p = root / "artifacts" / endpoint["directory"] / endpoint["checkpoint"]
        r = torch.load(p, weights_only=True, map_location="cpu")
        clock = verify_clock(
            (s["step"].item() for s in r["optimizer"]["state"].values()),
            endpoint["adam_clock"],
        )
        if r["metadata"] != endpoint["metadata"] or digest(p) != endpoint["sha256"]:
            raise ValueError("endpoint identity mismatch")
        for state in r["optimizer"]["state"].values():
            if any(
                not bool(torch.isfinite(state[k]).all())
                for k in ("exp_avg", "exp_avg_sq")
            ):
                raise ValueError("nonfinite optimizer moments")
        parent = r["metadata"].get("parent_sha256")
        if parent and parent not in by_hash:
            raise ValueError(f"missing lineage parent bytes: {parent}")
        lineage.append(
            {
                "directory": endpoint["directory"],
                "adam_clock": clock,
                "ancestral_updates": endpoint["ancestral_updates"],
            }
        )
    graph = FiniteGraph()
    truth = task_distribution(graph, 12)
    results = []
    for seed, name in [
        (100, "BS-soft-tail-1"),
        (101, "BS-independent-base101/soft-low"),
        (102, "BS-independent-base102/soft-low"),
    ]:
        directory = root / "artifacts" / name
        r = torch.load(
            directory / "ckpt_000500.pt", weights_only=True, map_location="cpu"
        )
        model = model_module.make_denoiser(seed).to(device)
        model.load_state_dict(r["model"], strict=True)
        if model.n_params() != 398980:
            raise ValueError("parameter count mismatch")
        table = cache_logwords(model, graph.boards)
        with np.load(directory / "table_000500.npz", allow_pickle=False) as z:
            table_error = float(np.max(np.abs(table - z["logwords"])))
        record = json.loads((directory / "eval_000500.json").read_text())
        laws, errors, summaries = {}, [], {}
        mass_error = 0.0
        with np.load(
            directory / "distribution_000500.npz", allow_pickle=False
        ) as saved:
            for order in ("ar", "random"):
                law = evaluate_exact(graph, table, order)
                laws[order] = law
                basin = truth["basin_masks"] @ law["p"]
                summary = {
                    "reward": float(law["p"] @ truth["reward"]),
                    "tv_prior": float(abs(law["p"] - truth["q"]).sum() / 2),
                    "minimum_basin": float(basin.min()),
                    "basin_total": float(basin.sum()),
                    "terminal_entropy": law["terminal_entropy"],
                    "conditional_entropy": law["current_conditional_entropy"],
                    "basin_masses": basin.tolist(),
                }
                summaries[order] = summary
                for key, value in summary.items():
                    errors.append(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(value) - record["decoders"][order][key]
                                )
                            )
                        )
                    )
                mass_error = max(
                    mass_error, float(np.max(abs(law["p"] - saved[f"{order}_p"])))
                )
        excess = denoising_excess(graph, table, truth["q"])
        dtv = float(abs(laws["ar"]["p"] - laws["random"]["p"]).sum() / 2)
        errors += [
            abs(dtv - record["decoder_tv"]),
            abs(excess["per_word_excess"] - record["denoising"]["per_word_excess"]),
        ]
        calibrated = dtv <= 0.02 and excess["per_word_excess"] <= 0.02
        target_basin = truth["basin_masks"] @ truth["q"]
        for summary in summaries.values():
            calibrated &= summary["tv_prior"] <= 0.05
            calibrated &= (
                np.max(abs(np.asarray(summary["basin_masses"]) / target_basin - 1))
                <= 0.2
            )
        if (
            not calibrated
            or table_error > 1e-4
            or max(errors) > 1e-5
            or mass_error > 1e-6
        ):
            raise ValueError(
                f"fresh evaluation mismatch: seed{seed}, {table_error}, {max(errors)}, {mass_error}"
            )
        results.append(
            {
                "seed": seed,
                "calibrated": bool(calibrated),
                "logword_table_max_error": table_error,
                "scalar_max_error": max(errors),
                "terminal_mass_max_error": mass_error,
                "decoders": summaries,
                "decoder_tv": dtv,
                "per_word_excess": excess["per_word_excess"],
            }
        )
        del model
    return {
        "device": device,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "lineage": lineage,
        "accepted_weights": results,
        "known_issue": "seed100:12500 ancestral updates, Adam16500; not repaired",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result = {"files_verified": verify_inventory(root), **verify_sources(root)}
    if args.neural:
        result["neural"] = verify_neural(root, args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
