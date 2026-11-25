#!/usr/bin/env python3
"""Preflight TouchScale pilot extraction: verify selected episodes and shard download footprint."""
from __future__ import annotations
import json, os, shutil
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
TASKS = [
    "Cut a mixed portion of fruit and vegetables that can be picked up with utensils",
    "Use a glass rod to stir the buffer solution in the beaker until the particles are mostly dissolved",
    "Use a plastic test-tube clamp to place the wide-mouth plastic test tube into the deep-hole rack",
    "Use a wooden test-tube clamp to place the narrow-mouth plastic test tube into the twelve-hole rack",
]
OUT = Path("artifacts/touchscale_pilot_preflight")

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = hf_hub_download(repo_id=REPO, filename="metadata/episodes.parquet",
                                repo_type="dataset", token=token)
    episodes = pd.read_parquet(meta_path)
    selected = episodes[episodes["task_name"].isin(TASKS)].copy()
    if selected.empty:
        raise RuntimeError("No episodes found for the four pre-registered pilot task families")
    missing = sorted(set(TASKS) - set(selected["task_name"]))
    if missing:
        raise RuntimeError(f"Pilot task families missing from full metadata: {missing}")
    if "shard" not in selected.columns:
        raise RuntimeError("episodes metadata has no shard column")
    shards = sorted(selected["shard"].dropna().astype(str).unique())
    api = HfApi(token=token)
    info = api.dataset_info(REPO, files_metadata=True)
    sizes = {}
    for sibling in info.siblings:
        if sibling.rfilename in shards:
            size = getattr(sibling, "size", None)
            if size is None and getattr(sibling, "lfs", None):
                size = getattr(sibling.lfs, "size", None)
            sizes[sibling.rfilename] = int(size) if size is not None else None
    absent = sorted(set(shards) - set(sizes))
    known = [v for v in sizes.values() if v is not None]
    total = sum(known)
    summary = {
        "task_families": int(selected["task_name"].nunique()),
        "episodes": int(selected["episode_id"].nunique()),
        "participants": int(selected["actor_id"].nunique()),
        "scenes": int(selected["scene"].nunique()),
        "unique_shards": len(shards),
        "known_shard_sizes": len(known),
        "missing_size_metadata": absent,
        "total_known_shard_bytes": total,
        "total_known_shard_gib": round(total / (1024**3), 3),
        "largest_shards_gib": sorted(
            [{"shard": k, "gib": round(v/(1024**3),3)} for k,v in sizes.items() if v is not None],
            key=lambda x: x["gib"], reverse=True
        )[:15],
        "runner_free_disk_gib": round(shutil.disk_usage(".").free/(1024**3),3),
        "download_strategy": "Do not download the full selected shard set unless the total footprint safely fits the runner disk with extraction overhead.",
    }
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"shard": k, "size_bytes": v, "size_gib": (round(v/(1024**3),3) if v is not None else None)}
                  for k,v in sorted(sizes.items())]).to_csv(OUT/"shard_sizes.csv", index=False)
    selected[["episode_id","task_id","task_name","scene","actor_id","collection_date",
              "duration_s","num_frames","quality_score","shard"]].to_csv(OUT/"pilot_episode_manifest.csv", index=False)
    (OUT/"selected_shards.txt").write_text("\n".join(shards)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
