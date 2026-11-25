#!/usr/bin/env python3
"""Audit 100 stratified TouchScale raw tactile episodes; no model fitting."""
from __future__ import annotations
import csv, hashlib, io, json, os, tarfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
CONFIG = Path("configs/touchscale_signal_sample_100.tsv")
OUT = Path("artifacts/touchscale_signal_100")
THRESHOLD_N = 3.0

def load_selection():
    rows = []
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        eid, shard, family = line.split("|")
        rows.append({"episode_id": eid, "shard": shard, "family": family})
    if len(rows) != 100 or len({r["episode_id"] for r in rows}) != 100:
        raise ValueError(f"Expected 100 unique episodes, got {len(rows)}")
    return rows

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def contiguous_segments(mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [{"start": int(g[0]), "end": int(g[-1]), "frames": int(len(g))} for g in groups]

def split_half_structure(X, active):
    idx = np.flatnonzero(active)
    if len(idx) < 40:
        return {"status": "insufficient_active_frames", "active_frames": int(len(idx))}
    mid = len(idx) // 2
    A = X[idx[:mid]]
    B = X[idx[mid:]]
    valid = (A.std(axis=0) > 1e-8) & (B.std(axis=0) > 1e-8)
    n_taxels = int(valid.sum())
    if n_taxels < 10:
        return {"status": "too_few_variable_taxels", "active_frames": int(len(idx)),
                "variable_taxels_both_halves": n_taxels}
    A, B = A[:, valid], B[:, valid]
    C1, C2 = np.corrcoef(A, rowvar=False), np.corrcoef(B, rowvar=False)
    iu = np.triu_indices(n_taxels, k=1)
    x, y = C1[iu], C2[iu]
    ok = np.isfinite(x) & np.isfinite(y)
    value = None
    if ok.sum() >= 20 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
        value = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return {"status": "computed" if value is not None else "undefined",
            "active_frames": int(len(idx)), "variable_taxels_both_halves": n_taxels,
            "valid_edges": int(ok.sum()), "split_half_edge_correlation_pearson": value}

def audit_hand(group):
    pad_keys = sorted((k for k in group.keys() if k.startswith("pad_")),
                      key=lambda x: int(x.split("_")[1]))
    if not pad_keys:
        return {"status": "no_pad_arrays"}
    arrays = [np.asarray(group[k][()]) for k in pad_keys]
    lengths = [a.shape[0] for a in arrays]
    if len(set(lengths)) != 1:
        return {"status": "pad_time_length_mismatch",
                "pad_shapes": {k: list(group[k].shape) for k in pad_keys}}
    T = lengths[0]
    X0 = np.concatenate([a.reshape(T, -1).astype(np.float32) for a in arrays], axis=1)
    finite = np.isfinite(X0)
    X = np.where(finite, X0, 0.0)
    force = X.sum(axis=1)
    active = force > THRESHOLD_N
    ts = np.asarray(group["t_epoch"][()]).reshape(-1) if "t_epoch" in group else np.array([])
    dt = np.diff(ts.astype(np.float64)) if len(ts) == T else np.array([])
    rate = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None
    return {
        "status": "ok", "frames": int(T), "taxels": int(X.shape[1]), "pads": len(pad_keys),
        "timestamps_length": int(len(ts)),
        "timestamps_monotonic": bool(np.all(dt > 0)) if len(dt) else None,
        "timestamp_rate_hz": rate,
        "finite_fraction": float(finite.mean()),
        "active_fraction_force_gt_3N": float(active.mean()),
        "active_frames": int(active.sum()),
        "active_segment_count": len(contiguous_segments(active)),
        "active_segments": contiguous_segments(active)[:20],
        "split_half_structure": split_half_structure(X, active),
    }

def audit_episode(blob):
    result = {"hands": {}}
    with h5py.File(io.BytesIO(blob), "r") as f:
        for hand in ("left", "right"):
            path = f"obs/tactile/raw/{hand}"
            result["hands"][hand] = audit_hand(f[path]) if path in f else {"status": "missing_hand_group"}
        if "time/timestamp_ns" in f:
            ts = np.asarray(f["time/timestamp_ns"][()]).reshape(-1)
            dt = np.diff(ts.astype(np.float64)) * 1e-9
            result["episode_timeline"] = {
                "frames": int(len(ts)), "monotonic": bool(np.all(dt > 0)),
                "median_rate_hz": float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None
            }
    return result

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)
    selected = load_selection()
    by_shard = defaultdict(list)
    for row in selected:
        by_shard[row["shard"]].append(row)
    details, rows, shard_hashes = {}, [], {}
    for shard, shard_rows in sorted(by_shard.items()):
        local = hf_hub_download(repo_id=REPO, filename=shard, repo_type="dataset", token=token)
        shard_hashes[shard] = sha256(local)
        wanted = {r["episode_id"] + ".episode.h5": r for r in shard_rows}
        found = set()
        with tarfile.open(local, "r:*") as tar:
            for member in tar:
                base = Path(member.name).name
                if base not in wanted:
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    continue
                meta = wanted[base]
                eid = meta["episode_id"]
                audited = audit_episode(stream.read())
                details[eid] = {**meta, **audited}
                found.add(eid)
                for hand, h in audited["hands"].items():
                    stability = h.get("split_half_structure", {})
                    rows.append({
                        "episode_id": eid, "family": meta["family"], "shard": shard, "hand": hand,
                        "status": h.get("status"), "frames": h.get("frames"), "taxels": h.get("taxels"),
                        "pads": h.get("pads"), "timestamp_rate_hz": h.get("timestamp_rate_hz"),
                        "timestamps_monotonic": h.get("timestamps_monotonic"),
                        "finite_fraction": h.get("finite_fraction"),
                        "active_fraction_force_gt_3N": h.get("active_fraction_force_gt_3N"),
                        "active_frames": h.get("active_frames"),
                        "active_segment_count": h.get("active_segment_count"),
                        "split_half_status": stability.get("status"),
                        "variable_taxels_both_halves": stability.get("variable_taxels_both_halves"),
                        "split_half_edge_corr": stability.get("split_half_edge_correlation_pearson"),
                    })
        for row in shard_rows:
            if row["episode_id"] not in found:
                details[row["episode_id"]] = {**row, "status": "episode_h5_not_found_in_shard"}
    (OUT / "signal_100_detail.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "signal_100_summary.csv", index=False)
    (OUT / "source_shards_sha256.json").write_text(json.dumps(shard_hashes, indent=2), encoding="utf-8")
    status_counts = pd.Series([v.get("status", "unknown") for v in details.values()]).value_counts().to_dict()
    print(json.dumps({"selected": len(selected), "found": sum(v.get("status") != "episode_h5_not_found_in_shard" for v in details.values()),
                      "shards": len(by_shard), "hand_rows": len(rows), "episode_status": status_counts,
                      "artifact_dir": str(OUT)}, indent=2))

if __name__ == "__main__":
    main()
