#!/usr/bin/env python3
"""Balanced six-episode, three-participant TouchScale tactile structure pilot.
Exploratory extraction only; no full-dataset download."""
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
TASK = "Operate the lever switch once"
EPISODES_PER_ACTOR = 2
MIN_ACTORS = 3
OUT = Path("artifacts/touchscale_six_episode_pilot")

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
        raise RuntimeError(f"Task family not found: {TASK}")
    actor_counts = candidates.groupby("actor_id").episode_id.nunique()
    eligible_actors = sorted(actor_counts[actor_counts >= EPISODES_PER_ACTOR].index.tolist())
    if len(eligible_actors) < MIN_ACTORS:
        raise RuntimeError(f"Need {MIN_ACTORS} actors with >= {EPISODES_PER_ACTOR} episodes; found {eligible_actors}")
    # Select three actors with the most repeated observations, then two best-quality episodes per actor.
    eligible_actors = sorted(eligible_actors, key=lambda a: (-int(actor_counts[a]), str(a)))[:MIN_ACTORS]
    selected_parts = []
    for actor in eligible_actors:
        part = candidates[candidates.actor_id == actor].sort_values(
            ["quality_score", "episode_id"], ascending=[False, True]
        ).head(EPISODES_PER_ACTOR)
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    if selected.episode_id.nunique() != MIN_ACTORS * EPISODES_PER_ACTOR:
        raise RuntimeError("Episode selection is not unique")
    if selected.actor_id.nunique() != MIN_ACTORS:
        raise RuntimeError("Participant balance check failed")
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
                            summaries.append({
                                "episode_id":meta["episode_id"], "actor_id":meta["actor_id"],
                                "collection_date":meta["collection_date"], "scene":meta["scene"],
                                "task_name":meta["task_name"], "hand":hand, **result
                            })
                    details[meta["episode_id"]] = {
                        "actor_id":meta["actor_id"], "collection_date":meta["collection_date"],
                        "shard":shard, "h5_bytes":len(blob), "hands":hand_results
                    }
                    found.add(base)
                    if found == wanted:
                        break
                missing = sorted(wanted-found)
                if missing:
                    raise RuntimeError(f"Missing episode members in {shard}: {missing}")
        details["_shard_"+shard] = {
            "size_bytes":shard_size, "range_read_seconds":round(time.time()-shard_started,2)
        }

    summary_df = pd.DataFrame(summaries)
    feature_df = pd.DataFrame(features)
    summary_df.to_csv(OUT/"episode_hand_dynamic_summary.csv", index=False)
    feature_df.to_csv(OUT/"time_bin_dynamic_features.csv", index=False)
    (OUT/"details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    valid = summary_df[summary_df.status == "ok"]
    per_actor = valid.groupby("actor_id").agg(
        episodes=("episode_id","nunique"),
        median_active_fraction=("active_fraction","median"),
        median_entropy=("median_bin_entropy_norm","median"),
        median_adjacent_js=("median_adjacent_bin_js_bits","median")
    ).reset_index()
    per_actor.to_csv(OUT/"participant_feature_summary.csv", index=False)
    dates_by_actor = selected.groupby("actor_id").collection_date.nunique().to_dict()
    result = {
        "status":"success" if len(valid)==MIN_ACTORS*EPISODES_PER_ACTOR*2 and len(feature_df)==MIN_ACTORS*EPISODES_PER_ACTOR*2*N_BINS else "partial",
        "task_name":TASK, "episodes":int(selected.episode_id.nunique()),
        "participants":int(selected.actor_id.nunique()), "episodes_per_participant":EPISODES_PER_ACTOR,
        "shards":int(len(by_shard)), "successful_hand_records":int(len(valid)),
        "feature_rows":int(len(feature_df)),
        "expected_feature_rows":MIN_ACTORS*EPISODES_PER_ACTOR*2*N_BINS,
        "collection_dates_by_participant":{str(k):int(v) for k,v in dates_by_actor.items()},
        "participant_date_confound_warning":"Each participant's selected records are from one collection date; participant effects cannot be separated from session/date effects in this pilot.",
        "elapsed_seconds":round(time.time()-started_all,2),
        "scope_limit":"Six-episode exploratory feature stability pilot only; not a powered cross-participant generalization or meta-learning evaluation."
    }
    (OUT/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        raise RuntimeError("Six-episode pilot did not produce all expected hand/bin feature rows")

if __name__ == "__main__":
    main()
