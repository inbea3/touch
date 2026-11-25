#!/usr/bin/env python3
"""Extract interpretable time-varying tactile spatial features from TouchScale.

This is a representation/QC stage only: no model fitting and no adaptation claims.
"""
from __future__ import annotations
import io, json, os, shutil, tarfile, hashlib, sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_touchscale_signal_300 import REPO, SHARD_CONFIG, choose_sample

OUT = Path("artifacts/touchscale_dynamic_structure")
N_BINS = 10
CONTACT_THRESHOLD_N = 3.0
EPS = 1e-10
MIN_BIN_FRAMES = 5
N_PADS = 15


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits for two probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    if p.sum() <= EPS or q.sum() <= EPS:
        return None
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def distribution_metrics(share):
    """Metrics for one 15-pad normalized load distribution."""
    x = np.asarray(share, dtype=np.float64)
    if x.shape != (N_PADS,) or not np.isfinite(x).all() or x.sum() <= EPS:
        return {"entropy_norm": None, "hhi": None, "top1_share": None,
                "top3_share": None, "dominant_pad_index": None}
    x = x / x.sum()
    nz = x > 0
    entropy = -float(np.sum(x[nz] * np.log(x[nz])))
    return {
        "entropy_norm": entropy / np.log(N_PADS),
        "hhi": float(np.sum(x*x)),
        "top1_share": float(np.max(x)),
        "top3_share": float(np.sort(x)[-3:].sum()),
        "dominant_pad_index": int(np.argmax(x))
    }


def make_time_bins(timestamps, n_bins=N_BINS):
    """Return normalized-time bin IDs; rejects non-finite/non-monotonic clocks."""
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(t) < n_bins * MIN_BIN_FRAMES:
        raise ValueError("too_few_timestamps")
    if not np.isfinite(t).all():
        raise ValueError("non_finite_timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("non_monotonic_timestamps")
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    ids = np.searchsorted(edges, t, side="right") - 1
    ids = np.clip(ids, 0, n_bins - 1)
    return ids, edges


def extract_hand(g, hand, meta):
    pad_keys = sorted([k for k in g if k.startswith("pad_")],
                      key=lambda k: int(k.split("_")[1]))
    if len(pad_keys) != N_PADS:
        return {"status": "pad_count_not_15", "pad_count": len(pad_keys)}, []
    arrays = [np.asarray(g[k][()], dtype=np.float32) for k in pad_keys]
    lengths = [a.shape[0] for a in arrays]
    if len(set(lengths)) != 1:
        return {"status": "pad_length_mismatch"}, []
    T = lengths[0]
    P = np.column_stack([
        np.maximum(a.reshape(T, -1), 0).sum(axis=1) for a in arrays
    ]).astype(np.float64)
    if not np.isfinite(P).all():
        return {"status": "non_finite_pad_load"}, []
    if "t_epoch" not in g:
        return {"status": "missing_t_epoch"}, []
    ts = np.asarray(g["t_epoch"][()], dtype=np.float64).reshape(-1)
    if len(ts) != T:
        return {"status": "timestamp_length_mismatch", "frames": T,
                "timestamp_count": len(ts)}, []
    try:
        bin_ids, edges = make_time_bins(ts)
    except ValueError as e:
        return {"status": str(e), "frames": T}, []

    total = P.sum(axis=1)
    active = total > CONTACT_THRESHOLD_N
    rows = []
    previous_share = None
    for b in range(N_BINS):
        ix = np.flatnonzero(bin_ids == b)
        if len(ix) < MIN_BIN_FRAMES:
            return {"status": "insufficient_frames_in_time_bin",
                    "frames": T, "bin": b, "bin_frames": len(ix)}, []
        Pb = P[ix]
        active_b = active[ix]
        active_loads = Pb[active_b]
        if len(active_loads):
            frame_totals = active_loads.sum(axis=1)
            shares_by_frame = active_loads / np.maximum(frame_totals[:, None], EPS)
            share = shares_by_frame.mean(axis=0)
            dm = distribution_metrics(share)
            mean_active_total = float(frame_totals.mean())
            mean_active_pad_load = active_loads.mean(axis=0)
            active_pad_share = share
        else:
            share = None
            dm = distribution_metrics(np.zeros(N_PADS))
            mean_active_total = None
            mean_active_pad_load = np.full(N_PADS, np.nan)
            active_pad_share = None

        row = {
            "episode_id": meta["episode_id"], "scene": meta["scene"],
            "task_name": meta["task_name"], "actor_id": meta["actor_id"],
            "collection_date": meta["collection_date"], "hand": hand,
            "time_bin": b, "time_start_s": float(edges[b]),
            "time_end_s": float(edges[b+1]), "bin_frames": int(len(ix)),
            "active_frames": int(active_b.sum()),
            "active_fraction": float(active_b.mean()),
            "mean_total_pad_load_N": float(Pb.sum(axis=1).mean()),
            "mean_active_total_load_N": mean_active_total,
            **dm
        }
        for j in range(N_PADS):
            row[f"pad_load_mean_{j:02d}_N"] = float(Pb[:, j].mean())
            row[f"pad_load_p95_{j:02d}_N"] = float(np.quantile(Pb[:, j], .95))
            row[f"pad_share_active_{j:02d}"] = (
                float(active_pad_share[j]) if active_pad_share is not None else np.nan
            )
            row[f"pad_load_active_mean_{j:02d}_N"] = (
                float(mean_active_pad_load[j]) if np.isfinite(mean_active_pad_load[j]) else np.nan
            )

        if active_pad_share is not None and previous_share is not None:
            row["adjacent_bin_pad_share_l1"] = float(np.abs(active_pad_share-previous_share).sum())
            row["adjacent_bin_pad_share_js_bits"] = js_divergence(active_pad_share, previous_share)
            denom = np.linalg.norm(active_pad_share)*np.linalg.norm(previous_share)
            row["adjacent_bin_pad_share_cosine"] = (
                float(np.dot(active_pad_share, previous_share)/denom) if denom > EPS else None
            )
        else:
            row["adjacent_bin_pad_share_l1"] = None
            row["adjacent_bin_pad_share_js_bits"] = None
            row["adjacent_bin_pad_share_cosine"] = None
        if active_pad_share is not None:
            previous_share = active_pad_share
        rows.append(row)

    active_bins = [r for r in rows if r["active_frames"] > 0]
    summary = {
        "status": "ok", "frames": int(T),
        "duration_s": float(ts[-1]-ts[0]),
        "effective_rate_hz": float((T-1)/(ts[-1]-ts[0])),
        "active_frames": int(active.sum()),
        "active_fraction": float(active.mean()),
        "active_bins": int(len(active_bins)),
        "valid_dynamic_transitions": int(sum(
            r["adjacent_bin_pad_share_js_bits"] is not None for r in rows
        )),
        "median_bin_entropy_norm": float(np.nanmedian([
            r["entropy_norm"] if r["entropy_norm"] is not None else np.nan for r in rows
        ])) if active_bins else None,
        "median_adjacent_bin_js_bits": float(np.nanmedian([
            r["adjacent_bin_pad_share_js_bits"]
            for r in rows if r["adjacent_bin_pad_share_js_bits"] is not None
        ])) if any(r["adjacent_bin_pad_share_js_bits"] is not None for r in rows) else None,
        "median_adjacent_bin_l1": float(np.nanmedian([
            r["adjacent_bin_pad_share_l1"]
            for r in rows if r["adjacent_bin_pad_share_l1"] is not None
        ])) if any(r["adjacent_bin_pad_share_l1"] is not None for r in rows) else None
    }
    return summary, rows


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "hf_cache"
    shards = [x.strip() for x in SHARD_CONFIG.read_text().splitlines() if x.strip()]
    meta_path = hf_hub_download(repo_id=REPO, filename="metadata/episodes.parquet",
        repo_type="dataset", token=token, cache_dir=str(cache))
    metadata = pd.read_parquet(meta_path)
    shutil.rmtree(cache, ignore_errors=True)
    sample, scene_quotas = choose_sample(metadata, shards)
    sample.to_csv(OUT / "sample_selection.csv", index=False)

    by_shard = defaultdict(list)
    for row in sample.to_dict("records"):
        by_shard[row["shard"]].append(row)

    episode_summaries, bin_rows, details, hashes = [], [], {}, {}
    for shard, rows in sorted(by_shard.items()):
        local = hf_hub_download(repo_id=REPO, filename=shard, repo_type="dataset",
            token=token, cache_dir=str(cache))
        h = hashlib.sha256()
        with open(local, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        hashes[shard] = h.hexdigest()
        with tarfile.open(local, "r:*") as tar:
            wanted = {r["episode_id"] + ".episode.h5" for r in rows}
            members = {Path(m.name).name: m for m in tar
                       if m.isfile() and Path(m.name).name in wanted}
            for row in rows:
                eid = str(row["episode_id"])
                member = members.get(eid + ".episode.h5")
                if member is None:
                    details[eid] = {"status": "h5_missing", "shard": shard}
                    continue
                with h5py.File(io.BytesIO(tar.extractfile(member).read()), "r") as h5:
                    hands = {}
                    for hand in ("left", "right"):
                        path = f"obs/tactile/raw/{hand}"
                        if path in h5:
                            result, rows_out = extract_hand(h5[path], hand, row)
                        else:
                            result, rows_out = {"status": "missing_hand"}, []
                        hands[hand] = result
                        bin_rows.extend(rows_out)
                        episode_summaries.append({
                            "episode_id": eid, "scene": row["scene"],
                            "task_name": row["task_name"], "actor_id": row["actor_id"],
                            "collection_date": row["collection_date"], "hand": hand,
                            **result
                        })
                details[eid] = {"status": "ok", "shard": shard, "hands": hands}
        shutil.rmtree(cache, ignore_errors=True)

    episode_df = pd.DataFrame(episode_summaries)
    bins_df = pd.DataFrame(bin_rows)
    episode_df.to_csv(OUT / "episode_hand_dynamic_summary.csv", index=False)
    bins_df.to_csv(OUT / "time_bin_dynamic_features.csv", index=False)
    (OUT / "detail.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (OUT / "shard_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    valid = episode_df[episode_df.status == "ok"]
    transitions = bins_df[bins_df.adjacent_bin_pad_share_js_bits.notna()] if len(bins_df) else bins_df
    result = {
        "selected_episodes": int(len(sample)),
        "unique_episodes": int(sample.episode_id.nunique()),
        "scenes": int(sample.scene.nunique()), "actors": int(sample.actor_id.nunique()),
        "dates": int(sample.collection_date.nunique()), "shards": int(len(by_shard)),
        "hand_records": int(len(episode_df)),
        "successful_hand_records": int(len(valid)),
        "failed_status_counts": {str(k): int(v) for k, v in
            episode_df[episode_df.status != "ok"].status.value_counts().items()},
        "time_bins_per_hand": N_BINS,
        "feature_rows": int(len(bins_df)),
        "valid_adjacent_bin_transitions": int(len(transitions)),
        "median_active_fraction": float(valid.active_fraction.median()) if len(valid) else None,
        "median_active_bins": float(valid.active_bins.median()) if len(valid) else None,
        "median_bin_entropy_norm": float(valid.median_bin_entropy_norm.median()) if len(valid) else None,
        "median_adjacent_bin_js_bits": float(valid.median_adjacent_bin_js_bits.median()) if len(valid) else None,
        "median_adjacent_bin_l1": float(valid.median_adjacent_bin_l1.median()) if len(valid) else None,
        "sampling": "same deterministic 300-episode scene-proportional sample via choose_sample",
        "representation": "15 pad-level loads and active-contact normalized pad shares over 10 equal-duration bins",
        "model_training": False
    }
    (OUT / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
