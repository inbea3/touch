#!/usr/bin/env python3
"""Strict pressure-only metadata/quality audit for TUM-ICS/Hide-and-Seek.

Only the four protocol-approved fields are retained in the audit process.
No source rows or raw force vectors are written to disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATASET = "TUM-ICS/Hide-and-Seek"
ALLOWED = ["obs_skin_force_left", "timestamp", "episode_index", "classification_label"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--revision", default=None, help="Pin an immutable HF commit SHA for reproducibility.")
    p.add_argument("--output-dir", default="results/has_external_audit")
    p.add_argument("--max-rows-per-split", type=int, default=0, help="0 means full split; nonzero is diagnostic only.")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_log = out / "run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    mapping_path = Path("data/has_label_mapping.json")
    if not mapping_path.exists():
        raise FileNotFoundError("Expected data/has_label_mapping.json in the project checkout.")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_map = {int(v): k for k, v in mapping["classification_labels"].items()}

    episodes: dict[tuple[str, int], dict] = {}
    split_objects: dict[str, set[str]] = defaultdict(set)
    split_rows: dict[str, int] = defaultdict(int)
    split_labels: dict[str, set[int]] = defaultdict(set)
    channel_n = np.zeros(44, dtype=np.int64)
    channel_mean = np.zeros(44, dtype=np.float64)
    channel_m2 = np.zeros(44, dtype=np.float64)
    channel_min = np.full(44, np.inf)
    channel_max = np.full(44, -np.inf)
    channel_zero = np.zeros(44, dtype=np.int64)
    channel_nonfinite = np.zeros(44, dtype=np.int64)
    channel_abs_sum = np.zeros(44, dtype=np.float64)

    config = {
        "dataset": args.dataset, "revision": args.revision,
        "allowed_fields": ALLOWED, "splits": ["train", "validation", "test"],
        "streaming": True, "max_rows_per_split": args.max_rows_per_split,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"START dataset={args.dataset} revision={args.revision or 'hub-default'} fields={ALLOWED}")

    for split in config["splits"]:
        kwargs = {"split": split, "streaming": True}
        if args.revision:
            kwargs["revision"] = args.revision
        ds = load_dataset(args.dataset, **kwargs)
        log(f"LOADED split={split}")
        for row_idx, row in enumerate(ds):
            # Immediately project to the frozen whitelist; never serialize source rows.
            rec = {k: row[k] for k in ALLOWED}
            eid = int(rec["episode_index"])
            label = int(rec["classification_label"])
            ts = float(rec["timestamp"])
            x = np.asarray(rec["obs_skin_force_left"], dtype=np.float64)
            if x.shape != (44,):
                x = np.full(44, np.nan)
            key = (split, eid)
            st = episodes.setdefault(key, {
                "split": split, "episode_index": eid, "n_frames": 0,
                "label_set": set(), "first_ts": ts, "last_ts": ts,
                "nonincreasing_steps": 0, "duplicate_ts": 0,
                "min_dt": math.inf, "max_dt": -math.inf,
                "finite_vector_frames": 0, "bad_shape_frames": 0,
            })
            st["n_frames"] += 1
            st["label_set"].add(label)
            if st["n_frames"] > 1:
                dt = ts - st["last_ts"]
                if dt <= 0:
                    st["nonincreasing_steps"] += 1
                if dt == 0:
                    st["duplicate_ts"] += 1
                st["min_dt"] = min(st["min_dt"], dt)
                st["max_dt"] = max(st["max_dt"], dt)
            st["last_ts"] = ts
            if x.shape == (44,) and np.isfinite(x).all():
                st["finite_vector_frames"] += 1
                channel_n += 1
                delta = x - channel_mean
                channel_mean += delta / channel_n
                channel_m2 += delta * (x - channel_mean)
                channel_min = np.minimum(channel_min, x)
                channel_max = np.maximum(channel_max, x)
                channel_zero += (x == 0)
                channel_abs_sum += np.abs(x)
            else:
                st["bad_shape_frames"] += 1
                channel_nonfinite += (~np.isfinite(x)) if x.shape == (44,) else 1
            split_rows[split] += 1
            split_labels[split].add(label)
            if args.max_rows_per_split and split_rows[split] >= args.max_rows_per_split:
                log(f"TRUNCATED split={split} rows={split_rows[split]}")
                break
            if row_idx and row_idx % 100000 == 0:
                log(f"PROGRESS split={split} rows={row_idx}")

        log(f"FINISHED split={split} rows={split_rows[split]} episodes={sum(1 for k in episodes if k[0] == split)}")

    episode_rows = []
    object_label_rows = []
    for (split, eid), st in sorted(episodes.items()):
        labels = sorted(st["label_set"])
        names = [class_map.get(v, f"UNKNOWN_{v}") for v in labels]
        objects = sorted({n.rsplit("_", 1)[0] for n in names if n != "negative" and "_" in n})
        episode_rows.append({
            "split": split, "episode_index": eid, "n_frames": st["n_frames"],
            "n_labels": len(labels), "labels": "|".join(names),
            "physical_objects": "|".join(objects),
            "label_consistent": len(labels) == 1,
            "first_timestamp": st["first_ts"], "last_timestamp": st["last_ts"],
            "duration_seconds": st["last_ts"] - st["first_ts"],
            "nonincreasing_steps": st["nonincreasing_steps"],
            "duplicate_timestamps": st["duplicate_ts"],
            "min_delta_t": st["min_dt"] if st["n_frames"] > 1 else "",
            "max_delta_t": st["max_dt"] if st["n_frames"] > 1 else "",
            "finite_44d_frames": st["finite_vector_frames"],
            "bad_frames": st["bad_shape_frames"],
        })
        for obj in objects:
            split_objects[split].add(obj)
            object_label_rows.append({"split": split, "episode_index": eid, "object": obj, "labels": "|".join(names)})

    all_objects = set().union(*split_objects.values()) if split_objects else set()
    split_overlap = []
    for obj in sorted(all_objects):
        splits = [s for s in config["splits"] if obj in split_objects[s]]
        split_overlap.append({"object": obj, "n_splits": len(splits), "splits": "|".join(splits), "cross_split": len(splits) > 1})

    variance = np.divide(channel_m2, np.maximum(channel_n - 1, 1), where=np.ones(44, dtype=bool))
    channel_rows = [{
        "channel": i, "n_finite": int(channel_n[i]), "mean": float(channel_mean[i]),
        "std": float(np.sqrt(max(variance[i], 0))), "min": float(channel_min[i]),
        "max": float(channel_max[i]), "zero_fraction": float(channel_zero[i] / max(channel_n[i], 1)),
        "mean_abs": float(channel_abs_sum[i] / max(channel_n[i], 1)),
    } for i in range(44)]

    write_csv(out / "episode_inventory.csv", episode_rows, list(episode_rows[0]) if episode_rows else ["split", "episode_index"])
    write_csv(out / "object_split_audit.csv", split_overlap, ["object", "n_splits", "splits", "cross_split"])
    write_csv(out / "episode_object_labels.csv", object_label_rows, ["split", "episode_index", "object", "labels"])
    write_csv(out / "channel_quality.csv", channel_rows, list(channel_rows[0]))
    consistency_rows = [r for r in episode_rows if not r["label_consistent"]]
    write_csv(out / "label_consistency.csv", consistency_rows, list(episode_rows[0]) if episode_rows else ["split", "episode_index"])

    summary = {
        "config": config, "split_rows": dict(split_rows),
        "episode_count_by_split": {s: sum(1 for k in episodes if k[0] == s) for s in config["splits"]},
        "distinct_objects_by_split": {s: sorted(split_objects[s]) for s in config["splits"]},
        "distinct_physical_objects_union": len(all_objects),
        "cross_split_objects": [r["object"] for r in split_overlap if r["cross_split"]],
        "label_inconsistent_episode_count": len(consistency_rows),
        "episodes_with_nonincreasing_timestamps": sum(r["nonincreasing_steps"] > 0 for r in episode_rows),
        "episodes_with_duplicate_timestamps": sum(r["duplicate_timestamps"] > 0 for r in episode_rows),
        "episodes_with_bad_force_frames": sum(r["bad_frames"] > 0 for r in episode_rows),
        "channel_zero_variance_count": sum(r["std"] == 0 for r in channel_rows),
        "elapsed_seconds": time.time() - started,
        "interpretation": "If max_rows_per_split > 0, all counts and diagnostics are partial and cannot be used for Go/No-Go decisions."
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(
        "# HAS pressure-only audit summary\n\n" +
        "\n".join(f"- **{k}:** {v}" for k, v in summary.items() if k != "config") +
        "\n\nThis is an audit only; no model training was performed.\n", encoding="utf-8")
    env = f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n"
    (out / "environment.txt").write_text(env, encoding="utf-8")
    manifest = []
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name != "manifest.sha256":
            manifest.append(f"{sha256_file(f)}  {f.name}")
    (out / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    log(f"DONE elapsed_seconds={summary['elapsed_seconds']:.1f} objects={len(all_objects)} episodes={len(episodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
