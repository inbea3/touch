#!/usr/bin/env python3
"""First-order ANIL for pressure-only cross-Recording object adaptation.

ANIL adapts only the final regression head on Support Recordings. A first-order
Query-loss gradient is then transferred to the meta-initialization (no
higher-order derivative through the inner updates). The object is the task and
the Recording is the Support/Query split unit.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from run_object_coactivation_baselines import extract
from run_reptile_meta_learning import (
    EdgeMLP, build_tasks, enumerate_object_episodes, initial_model_sha256,
    model_state_sha256, regression_metrics, sample_episode, support_batch_loss,
    tensors_for, train_support_matched_mlp, validate_manifest,
)

METRICS = ("mae", "rmse", "spearman")
METHODS = ("anil", "support_matched_anil_schedule", "reptile_reference", "ordinary_pooled_reference")
EVAL_POLICIES = (
    {"policy": "zero_shot", "steps": 0, "lr": None},
    {"policy": "head_only_5step_lr_1e-3", "steps": 5, "lr": 1e-3},
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_train_scaler(path: Path):
    with np.load(path) as saved:
        mean = saved["mean"].astype(np.float64)
        scale = saved["scale"].astype(np.float64)
        n_features = int(saved["n_features_in_"])
    if mean.ndim != 1 or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("Invalid train-only feature scaler")
    return mean, scale, n_features


def head_parameters(model: EdgeMLP):
    return list(model.network[-1].parameters())


def adapt_head_only(model: EdgeMLP, support_tensors, steps: int, lr: float,
                    weight_decay: float) -> EdgeMLP:
    """Adapt a copy of the final Linear head only; preserve the source model."""
    if steps < 0 or (steps and lr <= 0):
        raise ValueError("Invalid head adaptation settings")
    adapted = copy.deepcopy(model)
    head = adapted.network[-1]
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        adapted.zero_grad(set_to_none=True)
        per_recording = []
        for _rid, x, y in support_tensors:
            per_recording.append(torch.mean((adapted(x) - y) ** 2))
        loss = torch.stack(per_recording).mean()
        loss.backward()
        optimizer.step()
    adapted.eval()
    return adapted


def full_query_loss(model: EdgeMLP, query_tensors) -> torch.Tensor:
    """Equal mean of full-Recording MSEs over Query recordings."""
    if not query_tensors:
        raise ValueError("Query must contain at least one Recording")
    return torch.stack([
        torch.mean((model(x) - y) ** 2) for _rid, x, y in query_tensors
    ]).mean()


def first_order_anil_task_gradients(model: EdgeMLP, support_tensors, query_tensors,
                                    inner_steps: int, inner_lr: float,
                                    weight_decay: float, batch_size: int,
                                    rng: np.random.Generator):
    """Return first-order Query gradients after adapting only the task head.

    The inner optimizer sees only final-layer parameters. Query gradients are
    evaluated at the adapted task model and transferred directly to the
    initialization, omitting the Hessian/Jacobian of the inner update.
    """
    if inner_steps < 1:
        raise ValueError("inner_steps must be positive")
    task_model = copy.deepcopy(model)
    head = head_parameters(task_model)
    optimizer = torch.optim.Adam(head, lr=inner_lr, weight_decay=weight_decay)
    support_losses = []
    for _ in range(inner_steps):
        task_model.zero_grad(set_to_none=True)
        loss = support_batch_loss(task_model, support_tensors, batch_size, rng)
        loss.backward()
        optimizer.step()
        support_losses.append(float(loss.detach().cpu()))

    task_model.zero_grad(set_to_none=True)
    query_loss = full_query_loss(task_model, query_tensors)
    parameters = list(task_model.parameters())
    gradients = torch.autograd.grad(query_loss, parameters, allow_unused=False)
    named = list(task_model.named_parameters())
    gradient_map = {
        name: grad.detach().clone()
        for (name, _parameter), grad in zip(named, gradients)
    }
    grad_norm = math.sqrt(sum(float(torch.sum(g * g).cpu()) for g in gradient_map.values()))
    return gradient_map, float(np.mean(support_losses)), float(query_loss.detach().cpu()), grad_norm


def train_anil(data: dict, train_tasks: dict[int, list[int]], n_features: int,
               seed: int, meta_iterations: int, meta_batch_size: int,
               inner_steps: int, inner_lr: float, outer_lr: float,
               batch_size: int, weight_decay: float, device: torch.device):
    """Train first-order ANIL with task-balanced Query-loss outer updates."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed + 505)
    model = EdgeMLP(n_features).to(device)
    init_hash = model_state_sha256(model)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr, weight_decay=weight_decay)
    object_ids = np.asarray(sorted(train_tasks), dtype=int)
    cached = {oid: tensors_for(data, train_tasks[oid], device) for oid in object_ids}
    task_log, trace = [], []
    support_updates = query_gradient_evaluations = outer_updates = 0
    start = time.time()

    for iteration in range(1, meta_iterations + 1):
        chosen = rng.choice(object_ids, size=meta_batch_size, replace=False).tolist()
        gradients_by_task = []
        support_losses, query_losses, grad_norms = [], [], []
        outer_optimizer.zero_grad(set_to_none=True)

        for slot, oid_raw in enumerate(chosen, start=1):
            oid = int(oid_raw)
            episode = sample_episode(oid, train_tasks[oid], rng)
            lookup = {rid: (rid, x, y) for rid, x, y in cached[oid]}
            support = [lookup[rid] for rid in episode["support_recording_ids"]]
            query = [lookup[rid] for rid in episode["query_recording_ids"]]
            if set(episode["support_recording_ids"]) & set(episode["query_recording_ids"]):
                raise RuntimeError("Support/Query Recording leakage")
            grads, support_loss, query_loss, grad_norm = first_order_anil_task_gradients(
                model, support, query, inner_steps, inner_lr, weight_decay, batch_size, rng
            )
            gradients_by_task.append(grads)
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            grad_norms.append(grad_norm)
            task_log.append({
                "seed": seed, "iteration": iteration, "meta_batch_slot": slot,
                "object_id": oid, "shots": episode["shots"],
                "support_recording_ids": ";".join(map(str, episode["support_recording_ids"])),
                "query_recording_ids": ";".join(map(str, episode["query_recording_ids"])),
                "query_used_for_outer_gradient": True,
                "inner_adapted_parameters": "final_regression_head_only",
            })
            support_updates += inner_steps
            query_gradient_evaluations += 1

        # First-order ANIL: average task Query gradients equally, then update theta once.
        for name, parameter in model.named_parameters():
            parameter.grad = torch.stack([g[name] for g in gradients_by_task], dim=0).mean(dim=0)
        outer_optimizer.step()
        outer_updates += 1
        trace.append({
            "seed": seed, "iteration": iteration,
            "mean_support_minibatch_mse": float(np.mean(support_losses)),
            "mean_query_full_recording_mse": float(np.mean(query_losses)),
            "query_loss_sd_across_tasks": float(np.std(query_losses, ddof=1)) if len(query_losses) > 1 else 0.0,
            "mean_query_gradient_norm": float(np.mean(grad_norms)),
            "meta_batch_size": meta_batch_size, "inner_steps": inner_steps,
            "initialization_sha256": init_hash,
            "support_head_gradient_updates_this_iteration": meta_batch_size * inner_steps,
            "query_gradient_evaluations_this_iteration": meta_batch_size,
            "outer_optimizer_updates_this_iteration": 1,
        })
        if iteration == 1 or iteration % 25 == 0 or iteration == meta_iterations:
            print(json.dumps({
                "seed": seed, "iteration": iteration,
                "mean_support_mse": trace[-1]["mean_support_minibatch_mse"],
                "mean_query_mse": trace[-1]["mean_query_full_recording_mse"],
                "query_gradient_norm": trace[-1]["mean_query_gradient_norm"],
            }), flush=True)

    runtime = {
        "support_head_gradient_updates": support_updates,
        "query_gradient_evaluations": query_gradient_evaluations,
        "outer_optimizer_updates": outer_updates,
        "total_backward_passes": support_updates + query_gradient_evaluations,
        "expected_total_backward_passes": meta_iterations * meta_batch_size * (inner_steps + 1),
        "initialization_sha256": init_hash,
        "training_seconds": time.time() - start,
    }
    if runtime["total_backward_passes"] != runtime["expected_total_backward_passes"]:
        raise RuntimeError("ANIL backward-pass accounting mismatch")
    return model, task_log, trace, runtime


def load_checkpoint(path: Path, n_features: int, device: torch.device, expected_seed: int):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("n_features", n_features)) != n_features:
        raise ValueError(f"Feature-count mismatch in {path}")
    if int(checkpoint.get("seed", expected_seed)) != expected_seed:
        raise ValueError(f"Seed mismatch in {path}")
    model = EdgeMLP(n_features).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, file_sha256(path)


def aggregate_episode_rows(rows: list[dict]) -> list[dict]:
    buckets = {}
    for row in rows:
        key = (row["seed"], row["object_id"], row["shots"], row["method"], row["policy"])
        buckets.setdefault(key, []).append(row)
    out = []
    for key, values in sorted(buckets.items()):
        seed, oid, shots, method, policy = key
        item = {"seed": seed, "object_id": oid, "shots": shots,
                "method": method, "policy": policy,
                "n_query_recording_evaluations": len(values)}
        for metric in METRICS:
            valid = [v[metric] for v in values if v[metric] is not None and np.isfinite(v[metric])]
            item[metric] = float(np.mean(valid)) if valid else None
        out.append(item)
    return out


def aggregate_seeds_by_object(rows: list[dict]) -> list[dict]:
    buckets = {}
    for row in rows:
        key = (row["object_id"], row["shots"], row["method"], row["policy"])
        buckets.setdefault(key, []).append(row)
    out = []
    for key, values in sorted(buckets.items()):
        oid, shots, method, policy = key
        item = {"object_id": oid, "shots": shots, "method": method, "policy": policy,
                "n_seeds": len(values)}
        for metric in METRICS:
            valid = [v[metric] for v in values if v[metric] is not None and np.isfinite(v[metric])]
            item[metric] = float(np.mean(valid)) if valid else None
        out.append(item)
    return out


def bootstrap_mean(values, seed: int, repetitions: int = 5000) -> dict:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return {"n_objects": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(repetitions, len(x)))
    means = x[draws].mean(axis=1)
    return {
        "n_objects": int(len(x)), "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize(object_rows: list[dict], bootstrap_seed: int):
    grouped = {}
    for row in object_rows:
        grouped.setdefault((row["shots"], row["method"], row["policy"]), []).append(row)
    summaries = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        shots, method, policy = key
        item = {"shots": shots, "method": method, "policy": policy,
                "n_objects": len(values)}
        for metric in METRICS:
            stats = bootstrap_mean([v[metric] for v in values],
                                   bootstrap_seed + index * 101 + sum(map(ord, metric)))
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_sd"] = stats["sd"]
            item[f"{metric}_ci95_low"] = stats["ci95_low"]
            item[f"{metric}_ci95_high"] = stats["ci95_high"]
        summaries.append(item)

    lookup = {(r["object_id"], r["shots"], r["method"], r["policy"]): r["mae"]
              for r in object_rows}
    object_ids = sorted({r["object_id"] for r in object_rows})
    paired = []
    for shots in (1, 2):
        for policy in ("zero_shot", "head_only_5step_lr_1e-3"):
            for baseline in METHODS:
                if baseline == "anil":
                    continue
                diffs = []
                for oid in object_ids:
                    anil = lookup.get((oid, shots, "anil", policy))
                    other = lookup.get((oid, shots, baseline, policy))
                    if anil is not None and other is not None:
                        diffs.append({"object_id": oid, "baseline_minus_anil_mae": float(other - anil)})
                stats = bootstrap_mean(
                    [d["baseline_minus_anil_mae"] for d in diffs],
                    bootstrap_seed + shots * 10000 + len(paired) * 71
                )
                paired.append({
                    "shots": shots, "policy": policy, "baseline": baseline,
                    "difference_definition": "baseline_mae_minus_anil_mae",
                    "positive_favors": "anil", "object_differences": diffs,
                    "bootstrap": stats,
                })
    for shots in (1, 2):
        diffs = []
        for oid in object_ids:
            adapted = lookup.get((oid, shots, "anil", "head_only_5step_lr_1e-3"))
            zero = lookup.get((oid, shots, "anil", "zero_shot"))
            if adapted is not None and zero is not None:
                diffs.append({"object_id": oid, "adapted_minus_zero_shot_mae": float(adapted - zero)})
        paired.append({
            "shots": shots, "policy": "head_only_5step_lr_1e-3",
            "baseline": "anil_zero_shot",
            "difference_definition": "adapted_mae_minus_zero_shot_mae",
            "positive_favors": "zero_shot", "object_differences": diffs,
            "bootstrap": bootstrap_mean(
                [d["adapted_minus_zero_shot_mae"] for d in diffs],
                bootstrap_seed + shots * 30000 + 97
            ),
        })
    return summaries, paired


def evaluate(data: dict, validation_tasks: dict[int, list[int]], models: dict[str, EdgeMLP],
             seeds: list[int], adapt_steps: int, adapt_lr: float, weight_decay: float,
             device: torch.device):
    rows = []
    for seed in seeds:
        episodes = enumerate_object_episodes(validation_tasks, np.random.default_rng(seed + 606))
        cache = {oid: tensors_for(data, rids, device) for oid, rids in validation_tasks.items()}
        for episode_index, episode in enumerate(episodes, start=1):
            oid, shots = episode["object_id"], episode["shots"]
            lookup = {rid: (rid, x, y) for rid, x, y in cache[oid]}
            support = [lookup[rid] for rid in episode["support_recording_ids"]]
            query = [lookup[rid] for rid in episode["query_recording_ids"]]
            if set(episode["support_recording_ids"]) & set(episode["query_recording_ids"]):
                raise RuntimeError("Validation Support/Query leakage")
            for method, base_model in models.items():
                adapted = adapt_head_only(base_model, support, adapt_steps, adapt_lr, weight_decay)
                for policy, fitted in (("zero_shot", base_model), ("head_only_5step_lr_1e-3", adapted)):
                    with torch.no_grad():
                        for rid, x, y in query:
                            prediction = fitted(x).detach().cpu().numpy()
                            metrics = regression_metrics(y.detach().cpu().numpy(), prediction)
                            rows.append({
                                "seed": seed, "episode_index": episode_index,
                                "object_id": oid, "shots": shots,
                                "support_recording_ids": ";".join(map(str, episode["support_recording_ids"])),
                                "query_recording_id": rid, "method": method, "policy": policy,
                                **metrics,
                            })
        print(json.dumps({"seed": seed, "validation_episodes": len(episodes)}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--scaler-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="20261116,20261117,20261118")
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--sensor-seed", type=int, default=20261116)
    parser.add_argument("--meta-iterations", type=int, default=200)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=9)
    parser.add_argument("--inner-lr", type=float, default=0.001)
    parser.add_argument("--outer-lr", type=float, default=0.001)
    parser.add_argument("--adapt-steps", type=int, default=5)
    parser.add_argument("--adapt-lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--bootstrap-seed", type=int, default=20261121)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(1)
    seeds = [int(v) for v in args.seeds.split(",")]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct pre-specified seeds are required")
    if args.meta_iterations < 1 or args.meta_batch_size < 1 or args.inner_steps < 1:
        raise ValueError("Training counts must be positive")
    if args.meta_batch_size > 18 or args.adapt_steps < 0:
        raise ValueError("Invalid meta-batch or adaptation step count")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    mat = loadmat(args.metadata, simplify_cells=True)
    data, selected_sensors = extract(mat, manifest, n_nodes=args.nodes, seed=args.sensor_seed, k=5)
    mean, scale, n_features = load_train_scaler(args.scaler_npz)
    if n_features != int(next(iter(data.values()))["X"].shape[1]):
        raise ValueError("Scaler feature count does not match pressure-only features")
    for recording in data.values():
        recording["X"] = ((recording["X"].astype(np.float64) - mean) / scale).astype(np.float32)

    train_tasks = build_tasks(data, "meta_train")
    validation_tasks = build_tasks(data, "meta_validation")
    if len(train_tasks) != 18 or len(validation_tasks) != 4:
        raise ValueError("Expected fixed 18/4/4 split; Meta-test is deliberately not evaluated")
    if sorted(validation_tasks) != [6, 8, 10, 18]:
        raise ValueError("Validation object IDs differ from the frozen selected-validation protocol")

    reference_dir = args.reference_dir
    scaler_hash = file_sha256(args.scaler_npz)
    metadata_hash = file_sha256(args.metadata)
    manifest_hash = file_sha256(args.split_manifest)
    all_episode_rows, all_task_rows, all_trace_rows = [], [], []
    runtime_by_seed, checkpoint_hashes = {}, {}

    for seed in seeds:
        expected_init = initial_model_sha256(seed, n_features, device)
        reptile, reptile_hash = load_checkpoint(
            reference_dir / f"reptile_seed_{seed}.pt", n_features, device, seed
        )
        ordinary, ordinary_hash = load_checkpoint(
            reference_dir / f"ordinary_mlp_seed_{seed}.pt", n_features, device, seed
        )
        models_for_reference = {"reptile_reference": reptile, "ordinary_pooled_reference": ordinary}
        checkpoint_hashes[str(seed)] = {
            "reptile_reference": reptile_hash, "ordinary_pooled_reference": ordinary_hash
        }

        print(json.dumps({"seed": seed, "stage": "train_first_order_anil"}), flush=True)
        anil, task_log, trace, anil_runtime = train_anil(
            data, train_tasks, n_features, seed, args.meta_iterations, args.meta_batch_size,
            args.inner_steps, args.inner_lr, args.outer_lr, args.batch_size,
            args.weight_decay, device
        )
        if anil_runtime["initialization_sha256"] != expected_init:
            raise RuntimeError("ANIL initialization hash mismatch")
        support_matched, support_trace, support_seconds = train_support_matched_mlp(
            data, task_log, n_features, seed, args.inner_steps, args.inner_lr,
            args.batch_size, args.weight_decay, device
        )
        support_init = support_trace[0]["initialization_sha256"]
        if support_init != expected_init:
            raise RuntimeError("ANIL-schedule Support-matched baseline initialization mismatch")

        # Verify that the downloaded Reptile/ordinary checkpoints were trained
        # from the same deterministic seed initialization via their provenance traces.
        for trace_name in ("reptile_training_trace.csv", "ordinary_training_trace.csv"):
            trace_path = reference_dir / trace_name
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            with trace_path.open(newline="", encoding="utf-8-sig") as stream:
                first = next((row for row in csv.DictReader(stream) if int(row["seed"]) == seed), None)
            if first is None or first["initialization_sha256"] != expected_init:
                raise RuntimeError(f"Reference initialization hash mismatch: {trace_name}, seed={seed}")

        models = {
            "anil": anil,
            "support_matched_anil_schedule": support_matched,
            **models_for_reference,
        }
        validation_rows = evaluate(
            data, validation_tasks, models, [seed], args.adapt_steps,
            args.adapt_lr, args.weight_decay, device
        )
        all_episode_rows.extend(validation_rows)
        all_task_rows.extend(task_log)
        all_trace_rows.extend(trace)
        torch.save({
            "seed": seed, "state_dict": anil.state_dict(), "n_features": n_features,
            "method": "first_order_anil", "config": vars(args),
            "initialization_sha256": expected_init,
        }, args.output_dir / f"anil_seed_{seed}.pt")
        torch.save({
            "seed": seed, "state_dict": support_matched.state_dict(), "n_features": n_features,
            "method": "support_matched_anil_schedule", "config": vars(args),
            "initialization_sha256": expected_init,
        }, args.output_dir / f"support_matched_anil_schedule_seed_{seed}.pt")
        runtime_by_seed[str(seed)] = {
            **anil_runtime,
            "support_matched_schedule_gradient_updates": len(support_trace),
            "support_matched_schedule_training_seconds": support_seconds,
            "support_matched_schedule_initialization_sha256": support_init,
            "reference_reptile_checkpoint_sha256": reptile_hash,
            "reference_ordinary_checkpoint_sha256": ordinary_hash,
            "reference_models_initialization_verified_from_training_trace": True,
        }
        print(json.dumps({
            "seed": seed, "anil_backward_passes": anil_runtime["total_backward_passes"],
            "support_matched_updates": len(support_trace),
            "validation_query_rows": len(validation_rows),
        }), flush=True)

    object_seed_rows = aggregate_episode_rows(all_episode_rows)
    object_rows = aggregate_seeds_by_object(object_seed_rows)
    summary_rows, paired = summarize(object_rows, args.bootstrap_seed)
    expected_episode_rows = len(seeds) * 4 * 24 * 3  # four methods × two policies × 36 query evaluations
    if len(all_episode_rows) != expected_episode_rows:
        raise RuntimeError(f"Unexpected validation row count: {len(all_episode_rows)} != {expected_episode_rows}")
    if len(all_task_rows) != len(seeds) * args.meta_iterations * args.meta_batch_size:
        raise RuntimeError("Unexpected ANIL task schedule length")

    summary = {
        "scope": "STAG/HumanGrasp pressure-only; 96 selected sensors; no video or multimodal input",
        "algorithm": "first-order ANIL; adapt final regression head on Support; transfer first-order Query-loss gradients to the meta-initialization",
        "outer_objective": "equal mean of per-task Query losses; each Query loss is the equal mean of full-Recording MSEs",
        "inner_objective": "equal mean of per-Support-Recording minibatch MSEs",
        "meta_train_objects": sorted(train_tasks),
        "validation_objects": sorted(validation_tasks),
        "meta_test_evaluated": False,
        "n_meta_train_objects": len(train_tasks),
        "n_validation_objects": len(validation_tasks),
        "inferential_unit": "object",
        "seeds": seeds,
        "sensor_count": args.nodes,
        "selected_sensor_flat_indices_sha256": hashlib.sha256(
            np.asarray(selected_sensors, dtype=np.int64).tobytes()
        ).hexdigest(),
        "feature_dimension": n_features,
        "scaler_sha256": scaler_hash,
        "metadata_sha256": metadata_hash,
        "split_manifest_sha256": manifest_hash,
        "training_config": {
            "meta_iterations": args.meta_iterations, "meta_batch_size": args.meta_batch_size,
            "inner_head_steps": args.inner_steps, "inner_lr": args.inner_lr,
            "outer_lr": args.outer_lr, "adaptation_steps": args.adapt_steps,
            "adaptation_lr": args.adapt_lr, "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
        },
        "compute_accounting": {
            "anil_backward_passes_per_seed": args.meta_iterations * args.meta_batch_size * (args.inner_steps + 1),
            "support_head_updates_per_seed": args.meta_iterations * args.meta_batch_size * args.inner_steps,
            "query_gradient_evaluations_per_seed": args.meta_iterations * args.meta_batch_size,
            "outer_optimizer_updates_per_seed": args.meta_iterations,
            "support_matched_schedule_updates_per_seed": args.meta_iterations * args.meta_batch_size * args.inner_steps,
            "reference_reptile_inner_updates_per_seed": 8000,
            "reference_ordinary_gradient_updates_per_seed": 8000,
            "note": "ANIL matches 8000 total backward passes by allocating 7200 to Support head adaptation and 800 to Query-gradient evaluations; its Support-only control receives the same 7200 Support updates. Reptile/ordinary references are reused from the fixed 200x3 run and each received 8000 training updates.",
        },
        "methods": list(METHODS),
        "evaluation_policies": list(EVAL_POLICIES),
        "runtime_by_seed": runtime_by_seed,
        "reference_checkpoint_hashes": checkpoint_hashes,
        "policy_summary": summary_rows,
        "paired_comparisons": paired,
        "limitations": [
            "Only four Meta-validation objects are available and they were used in previous model selection and diagnostics; this experiment is exploratory.",
            "Object-level bootstrap intervals have n=4 and are descriptive, not confirmatory significance tests.",
            "The ANIL first-order approximation omits second-order derivatives through the inner head updates.",
            "The Support-matched ANIL-schedule control receives the same Support episodes and Support update count but no Query outer gradients.",
            "Reptile and ordinary reference checkpoints are reused from the selected 200-iteration, three-seed run; their original training schedules differ from ANIL's task schedule.",
            "Meta-test objects are not loaded or evaluated.",
        ],
    }
    write_csv(args.output_dir / "anil_episode_metrics.csv", all_episode_rows)
    write_csv(args.output_dir / "anil_object_seed_metrics.csv", object_seed_rows)
    write_csv(args.output_dir / "anil_object_metrics.csv", object_rows)
    write_csv(args.output_dir / "anil_training_trace.csv", all_trace_rows)
    write_csv(args.output_dir / "anil_task_schedule.csv", all_task_rows)
    write_csv(args.output_dir / "anil_policy_summary.csv", summary_rows)
    write_csv(args.output_dir / "anil_paired_comparisons.csv", [
        {"shots": row["shots"], "policy": row["policy"], "baseline": row["baseline"],
         "difference_definition": row["difference_definition"],
         "positive_favors": row["positive_favors"],
         "object_id": diff["object_id"],
         "object_difference": next(iter(diff.values())) if False else (
             diff.get("baseline_minus_anil_mae", diff.get("adapted_minus_zero_shot_mae"))
         )}
        for row in paired for diff in row["object_differences"]
    ])
    (args.output_dir / "anil_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "metadata_sha256": metadata_hash, "split_manifest_sha256": manifest_hash,
        "train_only_scaler_sha256": scaler_hash,
        "reference_dir": "selected-validation artifact from workflow run 44502406385",
        "pressure_only": True, "meta_test_loaded": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ANIL_EXPERIMENT_COMPLETE", flush=True)
    print(json.dumps({
        "validation_objects": summary["validation_objects"],
        "meta_test_evaluated": False,
        "policy_summary": summary_rows,
        "paired_comparisons": paired,
        "compute_accounting": summary["compute_accounting"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
