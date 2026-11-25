#!/usr/bin/env python3
"""Prove remote HTTP-range access to one selected TouchScale tar member before bulk extraction.
This test avoids downloading complete multi-gigabyte tar shards."""
from __future__ import annotations
import io, json, os, tarfile, time
from pathlib import Path
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
    candidates = candidates.sort_values(["quality_score", "episode_id"], ascending=[False, True])
    row = candidates.iloc[0]
    episode_id = str(row.episode_id)
    shard = str(row.shard)
    url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
    fs = fsspec.filesystem("http", block_size=8 * 1024 * 1024,
                           headers={"Authorization": f"Bearer {token}"})
    started = time.time()
    with fs.open(url, "rb") as remote:
        remote_size = remote.size
        with tarfile.open(fileobj=remote, mode="r:") as tar:
            target = episode_id + ".episode.h5"
            member = None
            for info in tar:
                if Path(info.name).name == target:
                    member = info
                    break
            if member is None:
                raise RuntimeError(f"Target {target} not found in {shard}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError("Tar member could not be opened")
            blob = extracted.read()
            with h5py.File(io.BytesIO(blob), "r") as h5:
                hand_info = {}
                for hand in ("left", "right"):
                    p = f"obs/tactile/raw/{hand}"
                    if p in h5:
                        group = h5[p]
                        pads = sorted([k for k in group if k.startswith("pad_")])
                        hand_info[hand] = {
                            "pad_count": len(pads),
                            "first_pad_shape": list(group[pads[0]].shape) if pads else None,
                            "has_t_epoch": "t_epoch" in group
                        }
                    else:
                        hand_info[hand] = {"missing": True}
    elapsed = time.time() - started
    cache = getattr(remote, "cache", None)
    cached_bytes = getattr(cache, "total_bytes", None)
    result = {
        "status": "success",
        "episode_id": episode_id,
        "task_name": TASK,
        "actor_id": str(row.actor_id),
        "shard": shard,
        "shard_size_bytes": int(remote_size),
        "target_h5_bytes": len(blob),
        "range_block_size_bytes": 8 * 1024 * 1024,
        "cache_bytes_reported": cached_bytes,
        "elapsed_seconds": round(elapsed, 2),
        "hand_groups": hand_info,
        "method": "fsspec HTTPFile with range-backed seekable tar access; no full shard download requested"
    }
    (OUT/"range_access_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
