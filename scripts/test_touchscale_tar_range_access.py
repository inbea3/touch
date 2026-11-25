#!/usr/bin/env python3
"""Small three-episode TouchScale tactile extraction test using HTTP range reads."""
from __future__ import annotations
import io, json, os, tarfile, time
from pathlib import Path
import pandas as pd
import h5py
from huggingface_hub import hf_hub_download, hf_hub_url
import fsspec

REPO = "2077AIDataFoundation/TouchScale"
TASK = "Use a glass rod to stir the buffer solution in the beaker until the particles are mostly dissolved"
N_EPISODES = 3
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
    # Select three episodes from one shard to test one remote object only.
    shard_counts = candidates.groupby("shard").episode_id.nunique().sort_values(ascending=False)
    shard = str(shard_counts.index[0])
    selected = candidates[candidates.shard.astype(str) == shard].sort_values(
        ["quality_score", "episode_id"], ascending=[False, True]
    ).head(N_EPISODES)
    if len(selected) < N_EPISODES:
        raise RuntimeError(f"Only {len(selected)} episodes available in selected shard {shard}")
    url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
    fs = fsspec.filesystem("http", block_size=8 * 1024 * 1024,
                           headers={"Authorization": f"Bearer {token}"})
    started = time.time()
    details = []
    with fs.open(url, "rb") as remote:
        remote_size = remote.size
        with tarfile.open(fileobj=remote, mode="r:") as tar:
            wanted = {str(eid) + ".episode.h5" for eid in selected.episode_id}
            found = set()
            for member in tar:
                if Path(member.name).name not in wanted:
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    continue
                blob = stream.read()
                with h5py.File(io.BytesIO(blob), "r") as h5:
                    hands = {}
                    for hand in ("left", "right"):
                        key = f"obs/tactile/raw/{hand}"
                        if key not in h5:
                            hands[hand] = {"status": "missing"}
                            continue
                        g = h5[key]
                        pads = sorted([k for k in g if k.startswith("pad_")])
                        hands[hand] = {
                            "status": "ok" if pads else "no_pad_arrays",
                            "pad_count": len(pads),
                            "first_pad_shape": list(g[pads[0]].shape) if pads else None,
                            "has_t_epoch": "t_epoch" in g,
                        }
                details.append({
                    "episode_id": Path(member.name).name.replace(".episode.h5", ""),
                    "h5_bytes": len(blob),
                    "hands": hands,
                })
                found.add(Path(member.name).name)
                if len(found) == len(wanted):
                    break
    elapsed = time.time() - started
    missing = sorted(wanted - found)
    result = {
        "status": "success" if not missing and len(details) == N_EPISODES else "incomplete",
        "task_name": TASK,
        "shard": shard,
        "shard_size_bytes": int(remote_size),
        "requested_episodes": N_EPISODES,
        "extracted_episodes": len(details),
        "missing_members": missing,
        "elapsed_seconds": round(elapsed, 2),
        "episodes": details,
        "method": "fsspec HTTPFile seekable tar stream with 8 MiB range blocks; only three HDF5 members extracted",
    }
    (OUT/"range_access_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        raise RuntimeError("Range-read pilot did not extract all requested episodes")

if __name__ == "__main__":
    main()
