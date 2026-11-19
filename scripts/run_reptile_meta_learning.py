#!/usr/bin/env python3
"""Pressure-only Reptile meta-learning with a gradient-budget-matched MLP baseline.

Task unit: object. Sample unit: recording. Prediction unit: sensor edge.
Support and Query are separated strictly by recording. Query is never used by
Reptile's inner loop or outer parameter-displacement update.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from run_object_coactivation_baselines import extract

TAU = 0.7
HIDDEN = (64, 32)

class EdgeMLP(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_features, HIDDEN[0]), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN[0], HIDDEN[1]), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN[1], 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    """Stable hash of tensor names, shapes, dtypes and values."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initial_model_sha256(seed: int, n_features: int, device: torch.device) -> str:
    set_seed(seed)
    return model_state_sha256(EdgeMLP(n_features).to(device))


def validate_manifest(manifest: dict) -> dict[int, dict]:
    if "object_splits" not in manifest:
        raise ValueError("Split manifest missing object_splits")
    entries = [e for group in manifest["object_splits"].values() for e in group]
    object_ids = [int(e["object_id"]) for e in entries]
    recording_ids = [int(r) for e in entries for r in e["recording_ids"]]
    if len(object_ids) != 26 or len(set(object_ids)) != 26:
        raise ValueError("Expected 26 disjoint objects in the fixed 18/4/4 manifest")
    if len(recording_ids) != 78 or len(set(recording_ids)) != 78:
        raise ValueError("Expected 78 unique recordings")
    if any(len(e["recording_ids"]) != 3 for e in entries):
        raise ValueError("Every object must have exactly three recordings")
    return {int(e["object_id"]): e for e in entries}


def build_tasks(data: dict, split: str) -> dict[int, list[int]]:
    tasks: dict[int, list[int]] = {}
    for rid, d in data.items():
        if d["split"] == split:
            tasks.setdefault(int(d["object_id"]), []).append(int(rid))
    for oid in tasks:
        tasks[oid] = sorted(tasks[oid])
        if len(tasks[oid]) != 3:
            raise ValueError(f"Object {oid} has {len(tasks[oid])} recordings, expected 3")
    return dict(sorted(tasks.items()))


def sample_episode(object_id: int, recording_ids: list[int], rng: np.random.Generator,
                   shots: int | None = None) -> dict:
    if len(recording_ids) != 3 or len(set(recording_ids)) != 3:
        raise ValueError("A meta-task must contain three unique recordings")
    k = int(rng.choice([1, 2])) if shots is None else int(shots)
    if k not in (1, 2):
        raise ValueError("shots must be 1 or 2")
    support = sorted(map(int, rng.choice(recording_ids, size=k, replace=False).tolist()))
    query = sorted(set(map(int, recording_ids)) - set(support))
    if not support or not query or set(support) & set(query):
        raise RuntimeError("Invalid Support/Query episode")
    return {"object_id": int(object_id), "shots": k,
            "support_recording_ids": support, "query_recording_ids": query}


def tensors_for(data: dict, recording_ids: list[int], device: torch.device):
    out = []
    for rid in recording_ids:
        x = torch.as_tensor(data[rid]["X"], dtype=torch.float32, device=device)
        y = torch.as_tensor(data[rid]["y"], dtype=torch.float32, device=device)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
            raise ValueError(f"Invalid feature/target arrays for recording {rid}")
        out.append((rid, x, y))
    return out


def sample_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                 rng: np.random.Generator):
    n = len(y)
    m = min(int(batch_size), n)
    idx = rng.choice(n, size=m, replace=False)
    ti = torch.as_tensor(idx, dtype=torch.long, device=x.device)
    return x.index_select(0, ti), y.index_select(0, ti)


def support_batch_loss(model: torch.nn.Module, support_tensors, batch_size: int,
                       rng: np.random.Generator) -> torch.Tensor:
    """Equal mean of per-Recording MSEs; edges never define inferential n."""
    per_recording = []
    per_recording_batch = max(1, int(batch_size) // len(support_tensors))
    for _rid, x, y in support_tensors:
        xb, yb = sample_batch(x, y, per_recording_batch, rng)
        per_recording.append(torch.mean((model(xb) - yb) ** 2))
    return torch.stack(per_recording).mean()


def full_support_loss(model: torch.nn.Module, support_tensors) -> torch.Tensor:
    per_recording = [torch.mean((model(x) - y) ** 2) for _rid, x, y in support_tensors]
    return torch.stack(per_recording).mean()


def apply_reptile_update(model: torch.nn.Module, task_states: list[dict[str, torch.Tensor]],
                         outer_step: float) -> float:
    """Apply theta <- theta + eps * mean_i(phi_i - theta), task weighted equally.

    Returns the norm of the meta-batch mean parameter displacement.
    """
    if not task_states:
        raise ValueError("At least one task state is required")
    if not (0.0 < float(outer_step) <= 1.0):
        raise ValueError("outer_step must be in (0, 1]")
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    keys = list(base)
    deltas = []
    for state in task_states:
        if set(state) != set(base):
            raise ValueError("Task state does not match meta-model state")
        deltas.append({k: state[k].detach().to(base[k].device) - base[k] for k in keys})
    with torch.no_grad():
        for k in keys:
            mean_delta = torch.stack([d[k] for d in deltas], dim=0).mean(dim=0)
            base[k].add_(mean_delta, alpha=float(outer_step))
        model.load_state_dict(base)
    norm = math.sqrt(sum(float(torch.sum(torch.stack([d[k] for d in deltas]).mean(0) ** 2)) for k in keys))
    return norm


def train_reptile(data: dict, train_tasks: dict[int, list[int]], n_features: int,
                  seed: int, meta_iterations: int, meta_batch_size: int,
                  inner_steps: int, inner_lr: float, outer_step: float,
                  batch_size: int, weight_decay: float, device: torch.device):
    set_seed(seed)
    rng = np.random.default_rng(seed + 101)
    model = EdgeMLP(n_features).to(device)
    init_hash = model_state_sha256(model)
    task_log, trace = [], []
    object_ids = np.asarray(sorted(train_tasks), dtype=int)
    cached = {oid: tensors_for(data, train_tasks[oid], device) for oid in object_ids}
    start = time.time()
    for iteration in range(1, meta_iterations + 1):
        chosen = rng.choice(object_ids, size=meta_batch_size, replace=False).tolist()
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        task_states, losses = [], []
        for slot, oid_raw in enumerate(chosen, start=1):
            oid = int(oid_raw)
            ep = sample_episode(oid, train_tasks[oid], rng)
            lookup = {rid: (rid, x, y) for rid, x, y in cached[oid]}
            support = [lookup[r] for r in ep["support_recording_ids"]]
            inner_model = EdgeMLP(n_features).to(device)
            inner_model.load_state_dict(base_state)
            optimizer = torch.optim.Adam(inner_model.parameters(), lr=inner_lr, weight_decay=weight_decay)
            task_loss = []
            for _ in range(inner_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = support_batch_loss(inner_model, support, batch_size, rng)
                loss.backward()
                optimizer.step()
                task_loss.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(task_loss)))
            task_states.append({k: v.detach().clone() for k, v in inner_model.state_dict().items()})
            task_log.append({"seed": seed, "iteration": iteration, "meta_batch_slot": slot,
                "object_id": oid, "shots": ep["shots"],
                "support_recording_ids": ";".join(map(str, ep["support_recording_ids"])),
                "query_recording_ids": ";".join(map(str, ep["query_recording_ids"])),
                "query_used_for_gradient": False})
        delta_norm = apply_reptile_update(model, task_states, outer_step)
        trace.append({"seed": seed, "iteration": iteration,
            "mean_support_minibatch_mse": float(np.mean(losses)),
            "task_loss_sd": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "norm_of_mean_task_displacement": delta_norm,
            "meta_batch_size": meta_batch_size, "inner_steps": inner_steps,
            "initialization_sha256": init_hash,
            "gradient_updates_this_iteration": meta_batch_size * inner_steps})
    return model, task_log, trace, time.time() - start


def train_pooled_mlp(data: dict, train_tasks: dict[int, list[int]], n_features: int,
                     seed: int, total_gradient_updates: int, inner_lr: float,
                     batch_size: int, weight_decay: float, device: torch.device):
    """Ordinary pooled MLP; task and Recording sampled uniformly per gradient step."""
    set_seed(seed)
    rng = np.random.default_rng(seed + 202)
    model = EdgeMLP(n_features).to(device)
    init_hash = model_state_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr, weight_decay=weight_decay)
    object_ids = np.asarray(sorted(train_tasks), dtype=int)
    cached = {oid: tensors_for(data, train_tasks[oid], device) for oid in object_ids}
    trace = []
    start = time.time()
    for step in range(1, total_gradient_updates + 1):
        oid = int(rng.choice(object_ids))
        recs = cached[oid]
        rid, x, y = recs[int(rng.integers(0, len(recs)))]
        xb, yb = sample_batch(x, y, batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xb) - yb) ** 2)
        loss.backward()
        optimizer.step()
        trace.append({"seed": seed, "gradient_step": step, "object_id": oid,
            "recording_id": rid, "initialization_sha256": init_hash,
            "minibatch_mse": float(loss.detach().cpu())})
    return model, trace, time.time() - start


def train_support_matched_mlp(data: dict, task_log: list[dict], n_features: int,
                              seed: int, inner_steps: int, inner_lr: float,
                              batch_size: int, weight_decay: float, device: torch.device):
    """Single shared MLP trained sequentially on exactly Reptile's task Support schedule.

    It receives the same task order, Support Recording IDs, per-task inner-step
    count and per-step batch construction, but it has no task-specific clones
    and no Reptile outer update. Adam state is reset at each task episode.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed + 404)
    model = EdgeMLP(n_features).to(device)
    init_hash = model_state_sha256(model)
    trace = []
    start = time.time()
    for episode_index, task in enumerate(task_log, start=1):
        oid = int(task["object_id"])
        support_ids = list(map(int, task["support_recording_ids"].split(";")))
        support = tensors_for(data, support_ids, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr, weight_decay=weight_decay)
        for inner_step in range(1, inner_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = support_batch_loss(model, support, batch_size, rng)
            loss.backward()
            optimizer.step()
            trace.append({
                "seed": seed,
                "gradient_step": len(trace) + 1,
                "episode_index": episode_index,
                "object_id": oid,
                "support_recording_ids": ";".join(map(str, support_ids)),
                "inner_step": inner_step,
                "initialization_sha256": init_hash,
                "minibatch_mse": float(loss.detach().cpu()),
            })
    return model, trace, time.time() - start


def adapt_model(model: torch.nn.Module, support_tensors, steps: int, lr: float,
                weight_decay: float) -> torch.nn.Module:
    adapted = copy.deepcopy(model)
    optimizer = torch.optim.Adam(adapted.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = full_support_loss(adapted, support_tensors)
        loss.backward()
        optimizer.step()
    return adapted


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(pred) or not len(y):
        raise ValueError("Metric arrays must be non-empty and have equal length")
    rho = spearmanr(y, pred).statistic if np.std(y) > 1e-12 and np.std(pred) > 1e-12 else np.nan
    return {"mae": float(np.mean(np.abs(y-pred))),
            "rmse": float(np.sqrt(np.mean((y-pred)**2))),
            "spearman": float(rho) if np.isfinite(rho) else None}


def enumerate_object_episodes(tasks: dict[int, list[int]], rng: np.random.Generator):
    episodes = []
    for oid, rids in sorted(tasks.items()):
        for shots in (1, 2):
            for support in itertools.combinations(rids, shots):
                support = sorted(map(int, support))
                query = sorted(set(rids) - set(support))
                episodes.append({"object_id": int(oid), "shots": shots,
                    "support_recording_ids": support, "query_recording_ids": query})
    return episodes


def evaluate_split(data: dict, tasks: dict[int, list[int]], ordinary_model,
                   support_matched_model, reptile_model, train_mean: float, seed: int, adapt_steps: int,
                   inner_lr: float, weight_decay: float, device: torch.device,
                   split_label: str) -> list[dict]:
    rng = np.random.default_rng(seed + 303)
    episodes = enumerate_object_episodes(tasks, rng)
    rows = []
    cache = {oid: tensors_for(data, rids, device) for oid, rids in tasks.items()}
    for ep_idx, ep in enumerate(episodes, start=1):
        oid, shots = ep["object_id"], ep["shots"]
        lookup = {rid: (rid, x, y) for rid, x, y in cache[oid]}
        support = [lookup[r] for r in ep["support_recording_ids"]]
        ordinary_adapted = adapt_model(ordinary_model, support, adapt_steps, inner_lr, weight_decay)
        support_matched_adapted = adapt_model(support_matched_model, support, adapt_steps, inner_lr, weight_decay)
        reptile_adapted = adapt_model(reptile_model, support, adapt_steps, inner_lr, weight_decay)
        for rid in ep["query_recording_ids"]:
            _rid, x, y = lookup[rid]
            y_np = y.detach().cpu().numpy()
            predictions = {
                "constant_train_zero_shot": np.full_like(y_np, train_mean),
                "ordinary_mlp_zero_shot": ordinary_model(x).detach().cpu().numpy(),
                "ordinary_mlp_support_adapted": ordinary_adapted(x).detach().cpu().numpy(),
                "support_matched_mlp_zero_shot": support_matched_model(x).detach().cpu().numpy(),
                "support_matched_mlp_support_adapted": support_matched_adapted(x).detach().cpu().numpy(),
                "reptile_zero_shot": reptile_model(x).detach().cpu().numpy(),
                "reptile_support_adapted": reptile_adapted(x).detach().cpu().numpy(),
            }
            for model_name, pred in predictions.items():
                met = regression_metrics(y_np, pred)
                rows.append({"split": split_label, "seed": seed, "episode_index": ep_idx,
                    "object_id": oid, "shots": shots,
                    "support_recording_ids": ";".join(map(str, ep["support_recording_ids"])),
                    "query_recording_id": rid, "model": model_name, **met})
    return rows


def aggregate_by_object(episode_rows: list[dict]) -> list[dict]:
    buckets = {}
    for row in episode_rows:
        key = (row["split"], row["seed"], row["object_id"], row["shots"], row["model"])
        buckets.setdefault(key, []).append(row)
    out = []
    for key, values in sorted(buckets.items()):
        split, seed, oid, shots, model = key
        item = {"split": split, "seed": seed, "object_id": oid, "shots": shots, "model": model,
                "n_query_recording_evaluations": len(values)}
        for metric in ("mae", "rmse", "spearman"):
            vals = [v[metric] for v in values if v[metric] is not None and np.isfinite(v[metric])]
            item[metric] = float(np.mean(vals)) if vals else None
        out.append(item)
    return out


def aggregate_seeds_by_object(object_rows: list[dict]) -> list[dict]:
    buckets = {}
    for row in object_rows:
        key = (row["split"], row["object_id"], row["shots"], row["model"])
        buckets.setdefault(key, []).append(row)
    out = []
    for key, values in sorted(buckets.items()):
        split, oid, shots, model = key
        item = {"split": split, "object_id": oid, "shots": shots, "model": model,
                "n_seeds": len(values)}
        for metric in ("mae", "rmse", "spearman"):
            vals = [v[metric] for v in values if v[metric] is not None and np.isfinite(v[metric])]
            item[metric] = float(np.mean(vals)) if vals else None
        out.append(item)
    return out


def bootstrap_mean(values, seed: int, n_boot: int) -> dict:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return {"n_objects": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {"n_objects": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975))}


def summarize(object_rows: list[dict], seed: int, n_boot: int) -> dict:
    result = {}
    for shots in (1, 2):
        result[str(shots)] = {}
        models = sorted({r["model"] for r in object_rows if r["shots"] == shots})
        for model in models:
            result[str(shots)][model] = {}
            for metric in ("mae", "rmse", "spearman"):
                vals = [r[metric] for r in object_rows if r["shots"] == shots and r["model"] == model]
                result[str(shots)][model][metric] = bootstrap_mean(vals, seed + shots*100 + sum(map(ord, model+metric)), n_boot)
    paired = []
    comparisons = [
        ("ordinary_mlp_support_adapted", "ordinary_finetune_minus_reptile_finetune"),
        ("support_matched_mlp_support_adapted", "support_matched_finetune_minus_reptile_finetune"),
    ]
    for shots in (1, 2):
        lookup = {(r["object_id"], r["model"]): r["mae"] for r in object_rows if r["shots"] == shots}
        object_ids = sorted({r["object_id"] for r in object_rows if r["shots"] == shots})
        for baseline_model, comparison_name in comparisons:
            diffs = []
            for oid in object_ids:
                a = lookup.get((oid, baseline_model))
                b = lookup.get((oid, "reptile_support_adapted"))
                if a is not None and b is not None:
                    diffs.append({"object_id": oid, "baseline_minus_reptile_mae": float(a-b)})
            vals = [d["baseline_minus_reptile_mae"] for d in diffs]
            paired.append({"shots": shots, "baseline_model": baseline_model,
                "comparison": comparison_name, "positive_favors_reptile": True,
                "objects_improved": int(np.sum(np.asarray(vals) > 0)),
                "object_differences": diffs,
                "bootstrap": bootstrap_mean(vals, seed + shots*991 + sum(map(ord, baseline_model)), n_boot)})
    return {"by_shots": result, "paired_mae": paired}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result file: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    metadata_path = Path(args.metadata)
    manifest_path = Path(args.split_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    mat = loadmat(metadata_path, simplify_cells=True)
    data, sensors = extract(mat, manifest, n_nodes=args.nodes, seed=args.sensor_seed, k=5)
    scaler_train_rids = sorted(rid for rid, d in data.items() if d["split"] == "meta_train")
    scaler = StandardScaler().fit(np.concatenate([data[r]["X"] for r in scaler_train_rids], axis=0))
    for d in data.values():
        d["X"] = scaler.transform(d["X"]).astype(np.float32)
    train_tasks = build_tasks(data, "meta_train")
    val_tasks = build_tasks(data, "meta_validation")
    test_tasks = build_tasks(data, "meta_test") if args.include_meta_test else {}
    if len(train_tasks) != 18 or len(val_tasks) != 4 or (args.include_meta_test and len(test_tasks) != 4):
        raise ValueError("Unexpected object counts in fixed 18/4/4 partition")
    train_mean = float(np.mean([data[r]["y"].mean() for r in scaler_train_rids]))
    n_features = int(next(iter(data.values()))["X"].shape[1])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    all_episode_rows, all_task_rows, all_reptile_trace, all_pooled_trace, all_support_trace = [], [], [], [], []
    runtime = {}
    for seed in args.seeds:
        expected_init_hash = initial_model_sha256(seed, n_features, device)
        print(f"[seed {seed}] expected_initialization_sha256={expected_init_hash}", flush=True)
        print(f"[seed {seed}] training Reptile", flush=True)
        reptile, task_log, reptile_trace, reptile_seconds = train_reptile(
            data, train_tasks, n_features, seed, args.meta_iterations, args.meta_batch_size,
            args.inner_steps, args.inner_lr, args.outer_step, args.batch_size,
            args.weight_decay, device)
        total_updates = args.meta_iterations * args.meta_batch_size * args.inner_steps
        print(f"[seed {seed}] training Support-matched MLP ({total_updates} updates)", flush=True)
        support_matched, support_trace, support_seconds = train_support_matched_mlp(
            data, task_log, n_features, seed, args.inner_steps, args.inner_lr,
            args.batch_size, args.weight_decay, device)
        print(f"[seed {seed}] training full-data pooled MLP ({total_updates} updates)", flush=True)
        ordinary, pooled_trace, pooled_seconds = train_pooled_mlp(
            data, train_tasks, n_features, seed, total_updates, args.inner_lr,
            args.batch_size, args.weight_decay, device)
        init_hashes = {
            "reptile": reptile_trace[0]["initialization_sha256"],
            "support_matched": support_trace[0]["initialization_sha256"],
            "ordinary": pooled_trace[0]["initialization_sha256"],
        }
        if any(value != expected_init_hash for value in init_hashes.values()):
            raise RuntimeError(f"Initializations are not identical for seed {seed}: {init_hashes}")
        val_rows = evaluate_split(data, val_tasks, ordinary, support_matched, reptile,
            train_mean, seed, args.adapt_steps, args.inner_lr, args.weight_decay,
            device, "meta_validation")
        all_episode_rows.extend(val_rows)
        if args.include_meta_test:
            test_rows = evaluate_split(data, test_tasks, ordinary, support_matched, reptile,
                train_mean, seed, args.adapt_steps, args.inner_lr, args.weight_decay,
                device, "meta_test_exploratory")
            all_episode_rows.extend(test_rows)
        all_task_rows.extend(task_log)
        all_reptile_trace.extend(reptile_trace)
        all_pooled_trace.extend(pooled_trace)
        all_support_trace.extend(support_trace)
        torch.save({"seed": seed, "state_dict": reptile.state_dict(), "n_features": n_features,
            "method": "Reptile", "config": vars(args)}, output_dir / f"reptile_seed_{seed}.pt")
        torch.save({"seed": seed, "state_dict": support_matched.state_dict(), "n_features": n_features,
            "method": "support_matched_mlp", "config": vars(args)}, output_dir / f"support_matched_mlp_seed_{seed}.pt")
        torch.save({"seed": seed, "state_dict": ordinary.state_dict(), "n_features": n_features,
            "method": "ordinary_pooled_mlp", "config": vars(args)}, output_dir / f"ordinary_mlp_seed_{seed}.pt")
        reptile_support_draws = sum(
            len(row["support_recording_ids"].split(";")) for row in task_log
        )
        support_matched_draws = sum(
            len(row["support_recording_ids"].split(";")) for row in task_log
        )
        runtime[str(seed)] = {
            "reptile_training_seconds": reptile_seconds,
            "support_matched_training_seconds": support_seconds,
            "ordinary_training_seconds": pooled_seconds,
            "initialization_sha256_shared_by_all_methods": expected_init_hash,
            "reptile_initialization_sha256": init_hashes["reptile"],
            "support_matched_initialization_sha256": init_hashes["support_matched"],
            "ordinary_initialization_sha256": init_hashes["ordinary"],
            "reptile_meta_task_episodes": len(task_log),
            "reptile_support_recording_draws": reptile_support_draws,
            "support_matched_task_episodes": len(task_log),
            "support_matched_recording_draws": support_matched_draws,
            "ordinary_object_recording_draws": len(pooled_trace),
            "reptile_inner_gradient_updates": total_updates,
            "support_matched_gradient_updates": len(support_trace),
            "ordinary_gradient_updates": total_updates,
            "gradient_update_budget_matched": len(support_trace) == total_updates,
            "same_initialization_verified_by_state_hash": True,
            "same_meta_train_object_pool": True,
            "support_matched_task_and_recording_access_match": (
                len(support_trace) == total_updates and support_matched_draws == reptile_support_draws
            ),
            "full_data_pool_strict_task_and_recording_access_match": False,
            "strict_task_and_recording_access_match": False,
        }
        print(f"[seed {seed}] validation evaluations={len(val_rows)}", flush=True)
    object_seed_rows = aggregate_by_object(all_episode_rows)
    object_rows = aggregate_seeds_by_object(object_seed_rows)
    val_objects = [r for r in object_rows if r["split"] == "meta_validation"]
    test_objects = [r for r in object_rows if r["split"] == "meta_test_exploratory"]
    summary = {"scope": "STAG/HumanGrasp pressure-only; no video or multimodal inputs",
        "status": "real_data_exploratory_run",
        "target": "continuous Pearson correlation across future valid-contact frames after the earliest five valid input frames",
        "auxiliary_threshold": TAU, "n_objects": 26, "n_recordings": 78,
        "meta_train_objects": 18, "meta_validation_objects": 4,
        "meta_test_objects": 4 if args.include_meta_test else 0,
        "inferential_unit": "object; recordings are within-task samples; sensor edges are not independent inferential units",
        "outer_split": "fixed 18/4/4 seed 20261118; prior stages used all objects, so validation/test are exploratory, not untouched confirmatory sets",
        "nodes": args.nodes, "edges_per_recording": args.nodes*(args.nodes-1)//2,
        "sensor_seed": args.sensor_seed, "selected_sensor_flat_indices": sensors,
        "feature_dimension": n_features, "scaler_fit_recordings": len(scaler_train_rids),
        "model": {"type": "MLP", "hidden_layers": list(HIDDEN), "activation": "ReLU", "output": 1,
            "optimizer": "Adam", "weight_decay": args.weight_decay},
        "reptile": {"meta_iterations": args.meta_iterations, "meta_batch_size": args.meta_batch_size,
            "inner_steps": args.inner_steps, "inner_lr": args.inner_lr, "outer_step": args.outer_step,
            "task_sampling": "uniform over meta-train objects without replacement within each meta-batch; 1/2-shot sampled uniformly",
            "inner_loss": "equal mean of per-Recording MSE; each Recording minibatch receives equal weight",
            "query_used_for_meta_gradient": False},
        "support_matched_baseline": {
            "training": "single shared MLP; processes the exact Reptile task order and Support Recording IDs, K sequential updates per task; Adam state resets per task; no task clones or Reptile outer update",
            "gradient_updates_per_seed": args.meta_iterations*args.meta_batch_size*args.inner_steps,
            "same_architecture": True,
            "same_training_object_pool": True,
            "same_seeded_initialization": True,
            "same_gradient_update_count": True,
            "same_task_episode_schedule": True,
            "same_support_recording_schedule": True,
            "same_per_recording_loss_weighting_and_batch_size": True,
        },
        "ordinary_baseline": {
            "training": "full-data pooled MLP; object then Recording sampled uniformly per gradient step from all three Meta-train Recordings",
            "gradient_updates_per_seed": args.meta_iterations*args.meta_batch_size*args.inner_steps,
            "same_architecture_and_optimizer": True,
            "same_training_objects": True,
            "same_seeded_initialization": True,
            "same_gradient_update_count": True,
            "strict_recording_access_match": False,
            "access_difference": (
                "The full-data pooled MLP draws one object/Recording per gradient step from all "
                "three Meta-train Recordings, whereas Reptile uses task-specific Support sets. "
                "This is retained as a secondary full-data benchmark, not the primary fair control."
            ),
        },
        "adaptation": {"steps": args.adapt_steps, "optimizer": "Adam", "learning_rate": args.inner_lr,
            "same_support_recordings_and_budget_for_all_three_initializations": True},
        "seeds": args.seeds, "runtime_by_seed": runtime,
        "validation_summary": summarize(val_objects, args.bootstrap_seed, args.bootstrap_reps),
        "meta_test_exploratory_summary": summarize(test_objects, args.bootstrap_seed+1, args.bootstrap_reps) if test_objects else None,
        "limitations": ["Only 18 independent meta-training object tasks.",
            "Four validation and four exploratory test objects yield highly uncertain object-level intervals.",
            "The fixed 18/4/4 test objects and all 26 objects were used in earlier research stages; no untouched final test remains.",
            "Only 96/548 sensors are used; no full-array claim.",
            "Sensor edges are correlated and are averaged within Recording; no edge-level inferential sample size is claimed.",
            "Reptile is compared with a Support-matched sequential MLP control and a full-data pooled MLP benchmark; object-level validation remains exploratory."]}
    args_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(args_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "task_sampling.csv", all_task_rows)
    write_csv(output_dir / "reptile_training_trace.csv", all_reptile_trace)
    write_csv(output_dir / "ordinary_training_trace.csv", all_pooled_trace)
    write_csv(output_dir / "support_matched_training_trace.csv", all_support_trace)
    write_csv(output_dir / "episode_metrics.csv", all_episode_rows)
    write_csv(output_dir / "object_seed_metrics.csv", object_seed_rows)
    write_csv(output_dir / "object_metrics.csv", object_rows)
    np.savez_compressed(output_dir / "feature_scaler.npz", mean=scaler.mean_, scale=scaler.scale_,
                        var=scaler.var_, n_features_in_=scaler.n_features_in_)
    provenance = {"metadata_sha256": sha256(metadata_path), "split_manifest_sha256": sha256(manifest_path),
        "metadata_file": metadata_path.name, "split_manifest_file": manifest_path.name,
        "feature_source": "scripts/run_object_coactivation_baselines.py::extract",
        "pressure_only": True}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--nodes", type=int, default=96)
    p.add_argument("--sensor-seed", type=int, default=20261116)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[20261116, 20261117, 20261118])
    p.add_argument("--meta-iterations", type=int, default=200)
    p.add_argument("--meta-batch-size", type=int, default=4)
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--inner-lr", type=float, default=5e-4)
    p.add_argument("--outer-step", type=float, default=0.05)
    p.add_argument("--adapt-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=20261119)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--include-meta-test", action="store_true",
                   help="Run the already-exposed meta-test objects; outputs remain exploratory.")
    a = p.parse_args(argv)
    if a.meta_iterations < 1 or a.meta_batch_size < 1 or a.inner_steps < 1 or a.adapt_steps < 0:
        p.error("iteration, batch and inner-step counts must be positive; adapt-steps may be zero")
    if a.meta_batch_size > 18:
        p.error("meta-batch-size cannot exceed 18 meta-train objects")
    if not (0 < a.outer_step <= 1) or a.inner_lr <= 0 or a.batch_size < 1:
        p.error("invalid optimizer settings")
    if not a.seeds:
        p.error("at least one seed is required")
    return a


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    validation = summary["validation_summary"]
    aggregate_only = {
        "by_shots": validation["by_shots"],
        "paired_mae": [
            {"shots": row["shots"], "baseline_model": row["baseline_model"],
             "comparison": row["comparison"],
             "positive_favors_reptile": row["positive_favors_reptile"],
             "objects_improved": row["objects_improved"],
             "bootstrap": row["bootstrap"]}
            for row in validation["paired_mae"]
        ],
    }
    print(json.dumps({"validation_summary_aggregate_only": aggregate_only,
        "meta_test_exploratory_summary": None if summary["meta_test_exploratory_summary"] is None else {
            "by_shots": summary["meta_test_exploratory_summary"]["by_shots"],
            "paired_mae": [
                {"shots": row["shots"], "comparison": row["comparison"],
                 "positive_favors_reptile": row["positive_favors_reptile"],
                 "objects_improved": row["objects_improved"],
                 "bootstrap": row["bootstrap"]}
                for row in summary["meta_test_exploratory_summary"]["paired_mae"]
            ],
        },
        "runtime_by_seed": summary["runtime_by_seed"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
