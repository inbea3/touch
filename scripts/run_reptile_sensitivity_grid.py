#!/usr/bin/env python3
"""Run the pre-specified 27-cell Reptile sensitivity grid on Meta-validation only.

All training outputs remain under the requested local output directory. This
script prints aggregate-only summaries and never uploads data or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

INNER_STEPS = (1, 5, 10)
INNER_LRS = (1e-4, 5e-4, 1e-3)
OUTER_STEPS = (0.01, 0.05, 0.1)
DEFAULT_SEEDS = (20261116, 20261117, 20261118)


def build_grid() -> list[dict]:
    return [
        {"inner_steps": k, "inner_lr": lr, "outer_step": eps}
        for k, lr, eps in itertools.product(INNER_STEPS, INNER_LRS, OUTER_STEPS)
    ]


def selection_score(row: dict) -> float:
    """Equal-weight average of 1-shot and 2-shot Reptile adapted object MAE."""
    return (float(row["reptile_mae_1shot"]) + float(row["reptile_mae_2shot"])) / 2.0


def choose_best(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot select a configuration from an empty grid")
    return min(rows, key=lambda r: (
        selection_score(r), int(r["inner_steps"]),
        float(r["inner_lr"]), float(r["outer_step"])
    ))


def summarize_run(summary: dict, config: dict) -> dict:
    if summary.get("meta_test_objects") != 0:
        raise ValueError("Sensitivity selection must not run or use Meta-test")
    by_shots = summary["validation_summary"]["by_shots"]
    row = dict(config)
    for shots in (1, 2):
        reptile = by_shots[str(shots)]["reptile_support_adapted"]["mae"]
        ordinary = by_shots[str(shots)]["ordinary_mlp_support_adapted"]["mae"]
        row[f"reptile_mae_{shots}shot"] = reptile["mean"]
        row[f"reptile_ci95_low_{shots}shot"] = reptile["ci95_low"]
        row[f"reptile_ci95_high_{shots}shot"] = reptile["ci95_high"]
        row[f"ordinary_mae_{shots}shot"] = ordinary["mean"]
        row[f"ordinary_ci95_low_{shots}shot"] = ordinary["ci95_low"]
        row[f"ordinary_ci95_high_{shots}shot"] = ordinary["ci95_high"]
    row["selection_score_equal_shot_mae"] = selection_score(row)
    row["n_validation_objects"] = 4
    row["n_seeds"] = len(summary["seeds"])
    row["gradient_update_budget_matched"] = all(
        x["gradient_update_budget_matched"] for x in summary["runtime_by_seed"].values()
    )
    row["strict_task_recording_access_match"] = all(
        x["strict_task_and_recording_access_match"]
        for x in summary["runtime_by_seed"].values()
    )
    row["access_difference_disclosed"] = not row["strict_task_recording_access_match"]
    if not row["gradient_update_budget_matched"] or not row["access_difference_disclosed"]:
        raise ValueError("Unexpected baseline budget/access audit result")
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty sensitivity summary")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_grid(args) -> dict:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).with_name("run_reptile_meta_learning.py")
    seeds_arg = ",".join(str(s) for s in args.seeds)
    rows = []
    data_provenance = None
    configs = build_grid()
    print(f"Starting {len(configs)} pre-specified configurations", flush=True)

    for index, config in enumerate(configs, start=1):
        name = (
            f"k{config['inner_steps']}_"
            f"alpha{config['inner_lr']:.0e}_"
            f"epsilon{config['outer_step']:.2f}"
        )
        out = root / name
        command = [
            sys.executable, str(script_path),
            "--metadata", str(args.metadata),
            "--split-manifest", str(args.split_manifest),
            "--output-dir", str(out),
            "--nodes", str(args.nodes),
            "--sensor-seed", str(args.sensor_seed),
            "--seeds", seeds_arg,
            "--meta-iterations", str(args.meta_iterations),
            "--meta-batch-size", str(args.meta_batch_size),
            "--inner-steps", str(config["inner_steps"]),
            "--inner-lr", str(config["inner_lr"]),
            "--outer-step", str(config["outer_step"]),
            "--adapt-steps", str(args.adapt_steps),
            "--batch-size", str(args.batch_size),
            "--weight-decay", str(args.weight_decay),
            "--bootstrap-reps", str(args.bootstrap_reps),
            "--bootstrap-seed", str(args.bootstrap_seed),
            "--device", args.device,
        ]
        print(
            f"[{index}/{len(configs)}] K={config['inner_steps']} "
            f"alpha={config['inner_lr']} epsilon={config['outer_step']}",
            flush=True,
        )
        subprocess.run(command, check=True)
        summary_path = out / "run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        if data_provenance is None:
            data_provenance = provenance
        elif provenance != data_provenance:
            raise ValueError("Data/feature provenance changed across grid configurations")
        if summary["seeds"] != list(args.seeds):
            raise ValueError("Run summary seed list does not match the pre-specified seeds")
        row = summarize_run(summary, config)
        rows.append(row)
        print(
            f"[{index}/{len(configs)}] aggregate validation MAE: "
            f"1-shot={row['reptile_mae_1shot']:.6f}, "
            f"2-shot={row['reptile_mae_2shot']:.6f}, "
            f"selection_score={row['selection_score_equal_shot_mae']:.6f}",
            flush=True,
        )

    best = choose_best(rows)
    result = {
        "status": "real_data_meta_validation_sensitivity_grid_completed",
        "scope": "STAG/HumanGrasp pressure-only; no video or multimodal inputs",
        "meta_test_used": False,
        "n_configurations": len(rows),
        "seeds": list(args.seeds),
        "meta_iterations_per_seed": args.meta_iterations,
        "meta_batch_size": args.meta_batch_size,
        "selection_rule": (
            "Minimum equal-weight average of 1-shot and 2-shot Reptile "
            "Support-adapted object-level Meta-validation MAE; seed values "
            "are averaged within object before the four objects are equally weighted."
        ),
        "selected_configuration": {
            "inner_steps": best["inner_steps"],
            "inner_lr": best["inner_lr"],
            "outer_step": best["outer_step"],
            "selection_score_equal_shot_mae": best["selection_score_equal_shot_mae"],
        },
        "configuration_rows": rows,
        "data_provenance": data_provenance,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "data_artifact_policy": (
            "Raw data, extracted features, model checkpoints, scalers and detailed "
            "per-Episode/per-object outputs remain on the ephemeral runner; no artifact uploaded."
        ),
    }
    (root / "sensitivity_grid_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(root / "sensitivity_grid_summary.csv", rows)
    print("SENSITIVITY_GRID_AGGREGATE_ONLY_BEGIN", flush=True)
    print(json.dumps({
        "n_configurations": len(rows),
        "selected_configuration": result["selected_configuration"],
        "selection_rule": result["selection_rule"],
        "data_provenance": data_provenance,
        "configuration_rows_aggregate_only": rows,
    }, ensure_ascii=False, indent=2), flush=True)
    print("SENSITIVITY_GRID_AGGREGATE_ONLY_END", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--sensor-seed", type=int, default=20261116)
    parser.add_argument("--seeds", type=lambda x: [int(v) for v in x.split(",")],
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--meta-iterations", type=int, default=50)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--adapt-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261119)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args(argv)
    if args.meta_iterations < 1 or args.meta_batch_size != 4:
        parser.error("meta-iterations must be positive and meta-batch-size is pre-specified as 4")
    if args.adapt_steps < 0 or args.batch_size < 1 or not args.seeds:
        parser.error("invalid adaptation, batch-size, or seed settings")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    return args


def main(argv=None):
    run_grid(parse_args(argv))


if __name__ == "__main__":
    main()
