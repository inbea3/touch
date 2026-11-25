#!/usr/bin/env python3
"""Small cross-participant TouchScale feature repeatability pilot (3 episodes, one task).
Uses metadata-selected episodes and remote range reads; no bulk download."""
from __future__ import annotations
import io, json, os, tarfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import h5py, fsspec
from huggingface_hub import hf_hub_download, hf_hub_url

from extract_touchscale_dynamic_structure import extract_hand, N_BINS

REPO = "2077AIDataFoundation/TouchScale"
TASK = "Use a glass rod to stir the buffer solution in the beaker until the particles are mostly dissolved"
OUT = Path("artifacts/touchscale_three_episode_pilot")

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = hf_hub_download(repo_id=REPO, filename="metadata/episodes.parquet",
                                repo_type="dataset", token=token)
    df = pd.read_parquet(meta_path)
    candidates = df[df.task_name == TASK].copy()
    if candidates.empty:
        raise RuntimeError("Pilot task missing from metadata")
    # Pick one high-quality episode per participant first, then a third episode.
    candidates = candidates.sort_values(["quality_score", "episode_id"], ascending=[False, True])
    actor_best = candidates.drop_duplicates("actor_id")
    selected = actor_best.head(2).copy()
    if len(selected) < 2:
        raise RuntimeError("Need at least two participants for cross-participant pilot")
    remaining = candidates[~candidates.episode_id.isin(selected.episode_id)]
    if remaining.empty:
        raise RuntimeError("No third episode available")
    selected = pd.concat([selected, remaining.head(1)], ignore_index=True)
    selected = selected.drop_duplicates("episode_id")
    if len(selected) != 3 or selected.actor_id.nunique() < 2:
        raise RuntimeError("Selection did not produce three episodes across at least two actors")
    selected.to_csv(OUT/"selected_episode_manifest.csv", index=False)

    by_shard = defaultdict(list)
    for row in selected.to_dict("records"):
        by_shard[str(row["shard"])].append(row)
    summaries, features, details = [], [], {}
    started_all = time.time()
    for shard, rows in sorted(by_shard.items()):
        url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
        fs = fsspec.filesystem("http", block_size=8*1024*1024,
                               headers={"Authorization": f"Bearer {token}"})
        shard_started = time.time()
        with fs.open(url, "rb") as remote:
            shard_size = int(remote.size)
            with tarfile.open(fileobj=remote, mode="r:") as tar:
                wanted = {str(r["episode_id"])+".episode.h5" for r in rows}
                found = set()
                for member in tar:
                    base = Path(member.name).name
                    if not member.isfile() or base not in wanted:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    blob = source.read()
                    row = next(r for r in rows if str(r["episode_id"])+".episode.h5" == base)
                    meta = {k: str(row[k]) for k in ("episode_id","scene","task_name","actor_id","collection_date")}
                    hand_results = {}
                    with h5py.File(io.BytesIO(blob), "r") as h5:
                        for hand in ("left","right"):
                            key = f"obs/tactile/raw/{hand}"
                            if key not in h5:
                                result, out_rows = {"status":"missing_hand"}, []
                            else:
                                result, out_rows = extract_hand(h5[key], hand, meta)
                            hand_results[hand] = result
                            features.extend(out_rows)
                            summaries.append({"episode_id":meta["episode_id"],"actor_id":meta["actor_id"],
                                "scene":meta["scene"],"task_name":meta["task_name"],"hand":hand,**result})
                    details[meta["episode_id"]] = {"shard":shard,"h5_bytes":len(blob),"hands":hand_results}
                    found.add(base)
                    if found == wanted:
                        break
                missing = sorted(wanted-found)
                if missing:
                    raise RuntimeError(f"Missing episode members in {shard}: {missing}")
        details["_shard_"+shard] = {"size_bytes":shard_size,"range_read_seconds":round(time.time()-shard_started,2)}
    summary_df = pd.DataFrame(summaries)
    feature_df = pd.DataFrame(features)
    summary_df.to_csv(OUT/"episode_hand_dynamic_summary.csv",index=False)
    feature_df.to_csv(OUT/"time_bin_dynamic_features.csv",index=False)
    (OUT/"details.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    valid = summary_df[summary_df.status=="ok"]
    result = {
        "status":"success" if len(valid)==6 and len(feature_df)==3*2*N_BINS else "partial",
        "task_name":TASK,"episodes":int(selected.episode_id.nunique()),
        "participants":int(selected.actor_id.nunique()),"shards":int(len(by_shard)),
        "successful_hand_records":int(len(valid)),"feature_rows":int(len(feature_df)),
        "expected_feature_rows":3*2*N_BINS,
        "median_active_fraction_by_hand":valid.groupby("hand").active_fraction.median().to_dict() if len(valid) else {},
        "median_entropy_by_hand":valid.groupby("hand").median_bin_entropy_norm.median().to_dict() if len(valid) else {},
        "median_adjacent_js_by_hand":valid.groupby("hand").median_adjacent_bin_js_bits.median().to_dict() if len(valid) else {},
        "elapsed_seconds":round(time.time()-started_all,2),
        "scope_limit":"Three-episode exploratory repeatability check only; not a statistically powered generalization or meta-learning evaluation."
    }
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="success":
        raise RuntimeError("Not all 3 episodes yielded valid features for both hands")

if __name__=="__main__":
    main()
