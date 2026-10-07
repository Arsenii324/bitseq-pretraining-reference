"""Registered small panel and LR selection, separate from core run lifecycle."""

import argparse
import json
from pathlib import Path

import numpy as np

from bitseq.artifacts import write_json
from bitseq.campaign import run_rl

LRS = (1e-4, 3e-4, 1e-3)


def select_lrs(candidates, reference_basins):
    lookup = {(r["method"], r["lr"]): r for r in candidates}
    methods = ("jg_ar", "tb_random", "trafl_iid", "trafl_comp")
    if (
        len(candidates) != 12
        or len(lookup) != 12
        or set(lookup) != {(m, lr) for m in methods for lr in LRS}
    ):
        raise ValueError("Incomplete or duplicated pilot set")
    if len(reference_basins) != 8 or any(x <= 0 for x in reference_basins):
        raise ValueError("Positive eight reference basin masses required")
    if any(r["status"] != "complete" for r in candidates):
        raise ValueError("Failed pilot; do not silently select from surviving runs")

    def lower_lr_tie(values, maximize):
        best = max(values.values()) if maximize else min(values.values())
        return min(lr for lr, value in values.items() if abs(value - best) <= 0.001)

    jg = {lr: lookup["jg_ar", lr]["mean_reward"] for lr in LRS}
    tb = {lr: lookup["tb_random", lr]["mean_tv_target"] for lr in LRS}
    eligible = {
        lr: min(lookup[m, lr]["mean_reward"] for m in methods[2:])
        for lr in LRS
        if all(
            np.all(
                np.array(lookup[m, lr]["final_basins"])
                > 0.5 * np.array(reference_basins)
            )
            for m in methods[2:]
        )
    }
    return dict(
        jg_ar=lower_lr_tie(jg, True),
        tb_random=lower_lr_tie(tb, False),
        trafl=lower_lr_tie(eligible, True) if eligible else 3e-4,
        trafl_optimization_resolved=bool(eligible),
        eligible_trafl_lrs=sorted(eligible),
        candidates=candidates,
    )


def panel_configs(selection):
    configs = []
    for seed in (201, 202, 203):
        for method in (
            "jg_ar",
            "pg_random",
            "trafl_iid",
            "trafl_comp",
            "trafl_ar",
            "tb_random",
        ):
            key = (
                "jg_ar"
                if method in ("jg_ar", "pg_random")
                else "tb_random"
                if method == "tb_random"
                else "trafl"
            )
            configs.append(dict(method=method, seed=seed, lr=selection[key], beta=0.3))
    for beta in (1.0, 0.03, 0.1, 3.0):
        for seed in (201, 202, 203):
            for method in (
                ("trafl_iid", "trafl_comp", "trafl_ar")
                if beta == 1.0
                else ("trafl_iid", "trafl_comp")
            ):
                configs.append(
                    dict(method=method, seed=seed, lr=selection["trafl"], beta=beta)
                )
    return configs


def execute_pilots(base, output):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    reference = json.loads((Path(base) / "CALIBRATION.json").read_text())["evaluation"][
        "decoders"
    ]["random"]["basin_masses"]
    candidates = []
    for method in ("jg_ar", "tb_random", "trafl_iid", "trafl_comp"):
        for lr in LRS:
            directory = output / f"{method}_lr{lr:g}"
            result = run_rl(
                base,
                directory,
                method=method,
                seed=200,
                lr=lr,
                beta=0.3,
                steps=300,
                audit_tables=False,
            )
            order = "ar" if method == "jg_ar" else "random"
            endpoints = [
                json.loads((directory / f"eval_{step:06d}.json").read_text())[
                    "decoders"
                ][order]
                for step in (200, 300)
            ]
            candidates.append(
                dict(
                    method=method,
                    lr=lr,
                    status=result["status"],
                    mean_reward=float(np.mean([r["reward"] for r in endpoints])),
                    mean_tv_target=float(np.mean([r["tv_target"] for r in endpoints])),
                    final_basins=endpoints[-1]["basin_masses"],
                    path=str(directory),
                )
            )
            write_json(output / "candidates.json", candidates, overwrite=True)
    selection = select_lrs(candidates, reference)
    write_json(output / "selection.json", selection)
    print("LR_SELECTION", json.dumps(selection), flush=True)
    return selection


def execute_panel(base, output, selection):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    configs = panel_configs(selection)
    write_json(
        output / "queue.json",
        dict(base=str(base), selection=selection, configs=configs),
    )
    for index, config in enumerate(configs):
        directory = (
            output
            / f"{index:02d}_{config['method']}_s{config['seed']}_b{config['beta']:g}"
        )
        result = run_rl(base, directory, **config)
        write_json(
            output / "progress.json",
            dict(
                completed=index + 1,
                total=len(configs),
                last_run=str(directory),
                status=result["status"],
            ),
            overwrite=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilots", "panel"))
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection")
    args = parser.parse_args()
    if args.stage == "pilots":
        execute_pilots(args.base, args.out)
    else:
        if not args.selection:
            parser.error("panel requires --selection")
        execute_panel(args.base, args.out, json.loads(Path(args.selection).read_text()))
