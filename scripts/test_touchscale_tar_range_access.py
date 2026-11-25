#!/usr/bin/env python3
"""One-episode TouchScale raw tactile read and HDF5 schema smoke test."""
from __future__ import annotations
import io, json, os, tarfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from huggingface_hub import hf_hub_download, hf_hub_url
import fsspec

REPO = "2077AIDataFoundation/TouchScale"
TASK = "Use a glass rod to stir the buffer solution in the beaker until the particles are mostly dissolved"
OUT = Path("artifacts/touchscale_range_access")

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
        raise RuntimeError("Pilot task family not found in metadata")
    row = candidates.sort_values(["quality_score", "episode_id"], ascending=[False, True]).iloc[0]
    episode_id, shard = str(row.episode_id), str(row.shard)
    url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
    fs = fsspec.filesystem("http", block_size=8 * 1024 * 1024,
                           headers={"Authorization": f"Bearer {token}"})
    started = time.time()
    with fs.open(url, "rb") as remote:
        remote_size = remote.size
        with tarfile.open(fileobj=remote, mode="r:") as tar:
            target = episode_id + ".episode.h5"
            member = next((m for m in tar if Path(m.name).name == target), None)
            if member is None:
                raise RuntimeError(f"Target {target} not found in shard {shard}")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError("Tar member could not be opened")
            blob = stream.read()
            with h5py.File(io.BytesIO(blob), "r") as h5:
                hands = {}
                for hand in ("left", "right"):
                    key = f"obs/tactile/raw/{hand}"
                    if key not in h5:
                        hands[hand] = {"status": "missing"}
                        continue
                    group = h5[key]
                    pads = sorted(k for k in group if k.startswith("pad_"))
                    pad_details = {}
                    for pad in pads:
                        ds = group[pad]
                        arr = np.asarray(ds)
                        pad_details[pad] = {
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "finite_fraction": float(np.isfinite(arr).mean()) if arr.size else None,
                            "min": float(np.nanmin(arr)) if arr.size else None,
                            "max": float(np.nanmax(arr)) if arr.size else None,
                            "mean": float(np.nanmean(arr)) if arr.size else None,
                            "std": float(np.nanstd(arr)) if arr.size else None,
                        }
                    hands[hand] = {
                        "status": "ok" if pads else "no_pad_arrays",
                        "pad_count": len(pads),
                        "has_t_epoch": "t_epoch" in group,
                        "pads": pad_details,
                    }
    result = {
        "status": "success",
        "episode_id": episode_id,
        "task_name": str(row.task_name),
        "actor_id": str(row.actor_id),
        "scene": str(row.scene),
        "shard": shard,
        "shard_size_bytes": int(remote_size),
        "h5_member_bytes": len(blob),
        "elapsed_seconds": round(time.time() - started, 2),
        "hands": hands,
        "method": "single HDF5 member read from seekable HTTP tar stream using 8 MiB range blocks",
    }
    (OUT/"range_access_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not any(v.get("status") == "ok" for v in hands.values()):
        raise RuntimeError("HDF5 opened, but no raw tactile pad arrays were found")

if __name__ == "__main__":
    main()
