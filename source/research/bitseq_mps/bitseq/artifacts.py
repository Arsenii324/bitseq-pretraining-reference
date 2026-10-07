"""Atomic, identity-checked local tensor checkpoints; no pickled model classes."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import torch


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return value


def save_checkpoint(
    path, model, optimizer, generators, metadata, *, overwrite=False, extra=None
):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        model=_cpu(model.state_dict()),
        optimizer=_cpu(optimizer.state_dict()),
        generators={name: gen.get_state() for name, gen in generators.items()},
        cpu_rng=torch.get_rng_state(),
        metadata=metadata,
        extra=_cpu(extra),
        mps_rng=torch.mps.get_rng_state()
        if torch.backends.mps.is_available()
        else None,
    )
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(record, temp)
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path)  # atomic no-clobber publication
            os.unlink(temp)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path, model, optimizer, generators, expected):
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid/truncated tensor checkpoint") from exc
    for key, value in expected.items():
        if record["metadata"].get(key) != value:
            raise ValueError(f"checkpoint identity mismatch: {key}")
    if set(generators) != set(record["generators"]):
        raise ValueError("checkpoint generator identity mismatch")
    model.load_state_dict(record["model"])
    optimizer.load_state_dict(record["optimizer"])
    for name, gen in generators.items():
        gen.set_state(record["generators"][name])
    torch.set_rng_state(record["cpu_rng"])
    if record["mps_rng"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(record["mps_rng"])
    return record["metadata"]


def write_json(path, payload, *, overwrite=False):
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path)
            os.unlink(temp)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def source_identity():
    root = Path(__file__).resolve().parents[3]
    files = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        files[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for name in [
        "src/tdlm/model.py",
        "src/tdlm/mdm.py",
        ".repos/JustGRPO/grpo.py",
        "research/bitseq_mps/setup_manifest.json",
    ]:
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return dict(
        hash=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        files=files,
    )


def archive_sources(output, identity):
    root = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(
        Path(output) / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, digest in identity["files"].items():
            raw = (root / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("source changed during archival")
            archive.writestr(name, raw)
