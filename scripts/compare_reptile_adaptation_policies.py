#!/usr/bin/env python3
"""Pre-specified validation-only comparison of conservative Support adaptation policies."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

from run_reptile_meta_learning import (
    EdgeMLP, build_tasks, enumerate_object_episodes, full_support_loss,
    model_state_sha256, regression_metrics, tensors_for, validate_manifest,
)
from run_object_coactivation_baselines import extract

METHODS = {
    "reptile": "reptile_seed_{}",
    "support_matched_mlp": "support_matched_mlp_seed_{}",
    "ordinary_mlp": "ordinary_mlp_seed_{}",
}
POLICIES = (
    {"policy": "zero_shot", "steps": 0, "lr": None},
    {"policy": "one_step_lr_1e-3", "steps": 1, "lr": 1e-3},
    {"policy": "two_steps_lr_1e-3", "steps": 2, "lr": 1e-3},
    {"policy": "one_step_lr_1e-4", "steps": 1, "lr": 1e-4},
    {"policy": "two_steps_lr_1e-4", "steps": 2, "lr": 1e-4},
)
METRICS = ("mae", "rmse", "spearman")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result file: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_scaler(path: Path) -> StandardScaler:
    with np.load(path) as saved:
        scaler = StandardScaler()
        scaler.mean_ = saved["mean"]
        scaler.scale_ = saved["scale"]
        scaler.var_ = saved["var"]
        scaler.n_features_in_ = int(saved["n_features_in_"])
    return scaler


def load_model(path: Path, n_features: int, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("n_features", n_features)) != n_features:
        raise ValueError(f"Checkpoint feature count mismatch: {path}")
    model = EdgeMLP(n_features).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def adapt_model(model, support, steps: int, lr: float | None, weight_decay: float):
    adapted = copy.deepcopy(model)
    if steps:
        if lr is None or lr <= 0:
            raise ValueError("Positive adaptation steps require a positive learning rate")
        optimizer = torch.optim.Adam(adapted.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = full_support_loss(adapted, support)
            loss.backward()
            optimizer.step()
    adapted.eval()
    return adapted


def object_seed_aggregate(episode_rows: list[dict]) -> list[dict]:
    buckets = {}
    for row in episode_rows:
        key = (row["seed"], row["object_id"], row["shots"], row["method"], row["policy"])
        buckets.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(buckets.items()):
        seed, oid, shots, method, policy = key
        item = {
            "seed": seed, "object_id": oid, "shots": shots,
            "method": method, "policy": policy,
            "adapt_steps": values[0]["adapt_steps"], "adapt_lr": values[0]["adapt_lr"],
            "n_query_recording_evaluations": len(values),
        }
        for metric in METRICS:
            valid = [v[metric] for v in values
                     if v[metric] is not None and np.isfinite(v[metric])]
            item[metric] = float(np.mean(valid)) if valid else None
        output.append(item)
    return output


def object_aggregate(object_seed_rows: list[dict]) -> list[dict]:
    buckets = {}
    for row in object_seed_rows:
        key = (row["object_id"], row["shots"], row["method"], row["policy"])
        buckets.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(buckets.items()):
        oid, shots, method, policy = key
        item = {
            "object_id": oid, "shots": shots, "method": method, "policy": policy,
            "adapt_steps": values[0]["adapt_steps"], "adapt_lr": values[0]["adapt_lr"],
            "n_seeds": len(values),
        }
        for metric in METRICS:
            valid = [v[metric] for v in values
                     if v[metric] is not None and np.isfinite(v[metric])]
            item[metric] = float(np.mean(valid)) if valid else None
        output.append(item)
    return output


def bootstrap_mean(values, seed: int, repetitions: int = 5000) -> dict:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return {"n_objects": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(repetitions, len(x)))
    means = x[draws].mean(axis=1)
    return {
        "n_objects": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize_policy_results(object_rows: list[dict], bootstrap_seed: int):
    grouped = {}
    for row in object_rows:
        grouped.setdefault((row["shots"], row["method"], row["policy"]), []).append(row)
    summary_rows = []
    summary_json = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        shots, method, policy = key
        item = {
            "shots": shots, "method": method, "policy": policy,
            "adapt_steps": values[0]["adapt_steps"], "adapt_lr": values[0]["adapt_lr"],
            "n_objects": len(values),
        }
        metrics = {}
        for metric in METRICS:
            stats = bootstrap_mean(
                [v[metric] for v in values],
                bootstrap_seed + index * 101 + sum(map(ord, metric)),
            )
            metrics[metric] = stats
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_sd"] = stats["sd"]
            item[f"{metric}_ci95_low"] = stats["ci95_low"]
            item[f"{metric}_ci95_high"] = stats["ci95_high"]
        summary_rows.append(item)
        summary_json.append({**item, "metrics": metrics})
    return summary_rows, summary_json


def paired_comparisons(object_rows: list[dict], bootstrap_seed: int):
    lookup = {
        (r["object_id"], r["shots"], r["method"], r["policy"]): r["mae"]
        for r in object_rows
    }
    object_ids = sorted({r["object_id"] for r in object_rows})
    rows = []
    summaries = []
    for shots in (1, 2):
        # Within-method adaptation gain: negative adapted-minus-zero-shot favors adaptation.
        for method in METHODS:
            for policy in POLICIES:
                name = policy["policy"]
                if name == "zero_shot":
                    continue
                diffs = []
                for oid in object_ids:
                    adapted = lookup.get((oid, shots, method, name))
                    zero = lookup.get((oid, shots, method, "zero_shot"))
                    if adapted is not None and zero is not None:
                        diffs.append({
                            "object_id": oid,
                            "difference_adapted_minus_zero_shot_mae": float(adapted - zero),
                        })
                vals = [d["difference_adapted_minus_zero_shot_mae"] for d in diffs]
                stats = bootstrap_mean(vals, bootstrap_seed + shots * 10000 + len(rows) * 37)
                row = {
                    "shots": shots, "comparison_type": "within_method_vs_zero_shot",
                    "method": method, "policy": name,
                    "difference_definition": "adapted_mae_minus_zero_shot_mae",
                    "positive_favors": "zero_shot", "object_differences": diffs,
                    "bootstrap": stats,
                }
                summaries.append(row)
                for d in diffs:
                    rows.append({
                        "shots": shots, "comparison_type": "within_method_vs_zero_shot",
                        "method": method, "policy": name, **d,
                    })
        # Same policy, same objects: baseline MAE minus Reptile MAE; positive favors Reptile.
        for policy in POLICIES:
            name = policy["policy"]
            for baseline in ("support_matched_mlp", "ordinary_mlp"):
                diffs = []
                for oid in object_ids:
                    reptile = lookup.get((oid, shots, "reptile", name))
                    other = lookup.get((oid, shots, baseline, name))
                    if reptile is not None and other is not None:
                        diffs.append({
                            "object_id": oid,
                            "difference_baseline_minus_reptile_mae": float(other - reptile),
                        })
                vals = [d["difference_baseline_minus_reptile_mae"] for d in diffs]
                stats = bootstrap_mean(vals, bootstrap_seed + shots * 20000 + len(rows) * 41)
                summaries.append({
                    "shots": shots, "comparison_type": "same_policy_model_comparison",
                    "method": baseline, "policy": name,
                    "difference_definition": "baseline_mae_minus_reptile_mae",
                    "positive_favors": "reptile", "object_differences": diffs,
                    "bootstrap": stats,
                })
                for d in diffs:
                    rows.append({
                        "shots": shots, "comparison_type": "same_policy_model_comparison",
                        "method": baseline, "policy": name, **d,
                    })
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--scaler-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="20261116,20261117,20261118")
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--sensor-seed", type=int, default=20261116)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--bootstrap-seed", type=int, default=20261120)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(1)
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Expected exactly three distinct pre-specified checkpoint seeds")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    mat = loadmat(args.metadata, simplify_cells=True)
    data, selected_sensors = extract(
        mat, manifest, n_nodes=args.nodes, seed=args.sensor_seed, k=5
    )
    scaler = load_scaler(args.scaler_npz)
    if scaler.n_features_in_ != int(next(iter(data.values()))["X"].shape[1]):
        raise ValueError("Scaler feature count does not match extracted pressure-only features")
    for recording in data.values():
        recording["X"] = scaler.transform(recording["X"]).astype(np.float32)

    train_tasks = build_tasks(data, "meta_train")
    validation_tasks = build_tasks(data, "meta_validation")
    if len(train_tasks) != 18 or len(validation_tasks) != 4:
        raise ValueError("Expected the fixed 18/4/4 split; Meta-test is deliberately not evaluated")
    episodes = enumerate_object_episodes(validation_tasks, np.random.default_rng(1))
    if len(episodes) != 24:
        raise ValueError(f"Expected 24 exhaustive validation episodes, got {len(episodes)}")

    n_features = int(next(iter(data.values()))["X"].shape[1])
    episode_rows = []
    checkpoint_hashes = {}
    checkpoint_state_hashes = {}

    for seed in seeds:
        models = {}
        checkpoint_hashes[str(seed)] = {}
        checkpoint_state_hashes[str(seed)] = {}
        for method, pattern in METHODS.items():
            checkpoint_path = args.checkpoint_dir / (pattern.format(seed) + ".pt")
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            models[method] = load_model(checkpoint_path, n_features, device)
            checkpoint_hashes[str(seed)][method] = sha256_file(checkpoint_path)
            checkpoint_state_hashes[str(seed)][method] = model_state_sha256(models[method])
        print(json.dumps({
            "seed": seed,
            "checkpoint_sha256": checkpoint_hashes[str(seed)],
            "state_sha256": checkpoint_state_hashes[str(seed)],
        }, sort_keys=True), flush=True)

        for episode_index, episode in enumerate(episodes, start=1):
            oid = int(episode["object_id"])
            shots = int(episode["shots"])
            lookup = {
                rid: (rid, x, y)
                for rid, x, y in tensors_for(data, validation_tasks[oid], device)
            }
            support = [lookup[rid] for rid in episode["support_recording_ids"]]
            if set(episode["support_recording_ids"]) & set(episode["query_recording_ids"]):
                raise RuntimeError("Support/Query recording leakage detected")

            for method, base_model in models.items():
                for policy in POLICIES:
                    fitted = adapt_model(
                        base_model, support, policy["steps"], policy["lr"], args.weight_decay
                    )
                    for query_rid in episode["query_recording_ids"]:
                        _rid, x, y = lookup[query_rid]
                        with torch.no_grad():
                            prediction = fitted(x).detach().cpu().numpy()
                        metrics = regression_metrics(y.detach().cpu().numpy(), prediction)
                        episode_rows.append({
                            "seed": seed, "episode_index": episode_index,
                            "object_id": oid, "shots": shots,
                            "support_recording_ids": ";".join(map(str, episode["support_recording_ids"])),
                            "query_recording_id": query_rid,
                            "method": method, "policy": policy["policy"],
                            "adapt_steps": policy["steps"], "adapt_lr": policy["lr"],
                            **metrics,
                        })

        print(json.dumps({
            "seed": seed, "episodes": len(episodes),
            "query_recording_evaluations": sum(len(ep["query_recording_ids"]) for ep in episodes),
            "completed_methods": list(models),
        }), flush=True)

    expected_rows = len(seeds) * len(METHODS) * len(POLICIES) * sum(
        len(ep["query_recording_ids"]) for ep in episodes
    )
    if len(episode_rows) != expected_rows:
        raise RuntimeError(f"Unexpected episode row count: {len(episode_rows)} != {expected_rows}")

    object_seed_rows = object_seed_aggregate(episode_rows)
    object_rows = object_aggregate(object_seed_rows)
    summary_rows, summary_json = summarize_policy_results(object_rows, args.bootstrap_seed)
    paired_rows, paired_json = paired_comparisons(object_rows, args.bootstrap_seed)

    write_csv(args.output_dir / "adaptation_policy_episode_metrics.csv", episode_rows)
    write_csv(args.output_dir / "adaptation_policy_object_seed_metrics.csv", object_seed_rows)
    write_csv(args.output_dir / "adaptation_policy_object_metrics.csv", object_rows)
    write_csv(args.output_dir / "adaptation_policy_summary.csv", summary_rows)
    write_csv(args.output_dir / "adaptation_policy_paired_differences.csv", paired_rows)

    summary = {
        "scope": "STAG/HumanGrasp pressure-only; 96 selected sensors; no video or multimodal input",
        "design": "pre-specified validation-only adaptation-policy comparison; no retraining",
        "source_checkpoints": "exact three-seed checkpoints from selected 200-meta-iteration validation run",
        "seeds": seeds,
        "meta_train_objects": sorted(train_tasks),
        "validation_objects": sorted(validation_tasks),
        "meta_test_evaluated": False,
        "independent_inferential_unit": "object",
        "n_validation_objects": len(validation_tasks),
        "episodes_per_object": 6,
        "total_validation_episodes": len(episodes),
        "query_recording_evaluations_per_seed": sum(len(ep["query_recording_ids"]) for ep in episodes),
        "methods": list(METHODS),
        "policies": list(POLICIES),
        "adaptation_loss": "equal mean of per-Support-Recording MSEs over all Support edges",
        "optimizer": "Adam",
        "weight_decay": args.weight_decay,
        "scaler_source": "reused exact train-only feature_scaler.npz from selected validation artifact",
        "sensor_subset_seed": args.sensor_seed,
        "n_selected_sensors": args.nodes,
        "selected_sensor_flat_indices_sha256": hashlib.sha256(
            np.asarray(selected_sensors, dtype=np.int64).tobytes()
        ).hexdigest(),
        "checkpoint_file_sha256": checkpoint_hashes,
        "checkpoint_state_sha256": checkpoint_state_hashes,
        "policy_summary": summary_json,
        "paired_comparisons": paired_json,
        "limitations": [
            "Only four Meta-validation objects are available; all estimates and object bootstrap intervals are exploratory.",
            "The fixed validation objects were previously used for model selection and earlier diagnostics; this is not confirmatory evidence.",
            "Support/Query permutations reuse the same three recordings and are correlated; summaries first aggregate within object and seed.",
            "The three initialization seeds are not additional independent objects.",
            "Meta-test objects are not loaded for evaluation or policy selection.",
            "No raw STAG pressure frames are included in the uploaded artifact."
        ],
    }
    (args.output_dir / "adaptation_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ADAPTATION_POLICY_COMPARISON_COMPLETE", flush=True)
    print(json.dumps({
        "validation_objects": summary["validation_objects"],
        "policy_summary": summary_json,
        "meta_test_evaluated": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
