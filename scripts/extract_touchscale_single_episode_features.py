#!/usr/bin/env python3
"""Extract the established 10-bin tactile structure representation for one real episode.
Pilot scope: exactly one metadata-selected episode; no bulk shard download."""
from __future__ import annotations
import io, json, os, tarfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import fsspec
from huggingface_hub import hf_hub_download, hf_hub_url

from extract_touchscale_dynamic_structure import extract_hand, CONTACT_THRESHOLD_N, N_BINS

REPO = "2077AIDataFoundation/TouchScale"
TASK = "Use a glass rod to stir the buffer solution in the beaker until the particles are mostly dissolved"
OUT = Path("artifacts/touchscale_single_episode_features")

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = hf_hub_download(repo_id=REPO, filename="metadata/episodes.parquet",
                                    repo_type="dataset", token=token)
    metadata = pd.read_parquet(metadata_path)
    candidates = metadata[metadata.task_name == TASK].copy()
    if candidates.empty:
        raise RuntimeError("Pilot task family absent from TouchScale metadata")
    row = candidates.sort_values(["quality_score", "episode_id"], ascending=[False, True]).iloc[0]
    meta = {
        "episode_id": str(row.episode_id), "scene": str(row.scene),
        "task_name": str(row.task_name), "actor_id": str(row.actor_id),
        "collection_date": str(row.collection_date),
    }
    shard = str(row.shard)
    url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
    fs = fsspec.filesystem("http", block_size=8*1024*1024,
                           headers={"Authorization": f"Bearer {token}"})
    started = time.time()
    with fs.open(url, "rb") as remote:
        remote_size = int(remote.size)
        with tarfile.open(fileobj=remote, mode="r:") as tar:
            wanted = meta["episode_id"] + ".episode.h5"
            member = next((m for m in tar if m.isfile() and Path(m.name).name == wanted), None)
            if member is None:
                raise RuntimeError(f"Episode member {wanted} not found in {shard}")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError("Could not read HDF5 tar member")
            blob = source.read()
    hand_summaries, feature_rows, threshold_qc = [], [], []
    with h5py.File(io.BytesIO(blob), "r") as h5:
        for hand in ("left", "right"):
            key = f"obs/tactile/raw/{hand}"
            if key not in h5:
                hand_summaries.append({"hand": hand, "status": "missing_hand"})
                continue
            summary, rows = extract_hand(h5[key], hand, meta)
            hand_summaries.append({"hand": hand, **summary})
            feature_rows.extend(rows)
            # Threshold sensitivity diagnostic, using the same pad-load definition.
            g = h5[key]
            pads = sorted((k for k in g if k.startswith("pad_")), key=lambda k:int(k.split("_")[1]))
            if len(pads) == 15:
                arrays = [np.asarray(g[k][()], dtype=np.float32) for k in pads]
                if len({a.shape[0] for a in arrays}) == 1:
                    T = arrays[0].shape[0]
                    loads = np.column_stack([np.maximum(a.reshape(T,-1),0).sum(axis=1) for a in arrays])
                    total = loads.sum(axis=1)
                    ts = np.asarray(g["t_epoch"][()]).reshape(-1) if "t_epoch" in g else np.array([])
                    for threshold in (0.1, 0.5, 1.0, 3.0, 5.0):
                        threshold_qc.append({
                            "hand": hand, "threshold": threshold,
                            "active_frame_fraction": float((total > threshold).mean()),
                            "median_total_load": float(np.median(total)),
                            "p95_total_load": float(np.quantile(total,.95)),
                            "timestamp_count": int(len(ts)),
                            "frame_count": int(T),
                            "timestamp_monotonic": bool(len(ts)==T and np.isfinite(ts).all() and np.all(np.diff(ts)>0)),
                        })
    features = pd.DataFrame(feature_rows)
    features.to_csv(OUT/"time_bin_dynamic_features.csv", index=False)
    pd.DataFrame(hand_summaries).to_csv(OUT/"episode_hand_summary.csv", index=False)
    pd.DataFrame(threshold_qc).to_csv(OUT/"contact_threshold_sensitivity.csv", index=False)
    result = {
        "status": "success" if len(features) == 2*N_BINS and all(x.get("status")=="ok" for x in hand_summaries) else "partial",
        "episode": meta, "shard": shard, "shard_size_bytes": remote_size,
        "h5_member_bytes": len(blob), "range_read_seconds": round(time.time()-started,2),
        "hands": hand_summaries, "feature_rows": int(len(features)),
        "time_bins": N_BINS, "contact_threshold_N": CONTACT_THRESHOLD_N,
        "representation": "existing project extractor: 15 pad-wise positive load sums, 10 equal-duration bins, active-contact load distribution, normalized entropy/HHI/top shares and adjacent-bin JS/L1/cosine changes",
        "important_limit": "One episode is a feature-pipeline smoke test only; it does not validate between-episode repeatability or few-shot adaptation."
    }
    (OUT/"run_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"] != "success":
        raise RuntimeError("Single-episode feature extraction did not produce valid 10-bin features for both hands")

if __name__ == "__main__":
    main()
