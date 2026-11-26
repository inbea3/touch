#!/usr/bin/env python3
"""Nine-episode TouchScale contact and sampling-rate quality audit.

Selects three episodes per participant for one repeated task family. Extracts
features at the current 3 N threshold and independently audits threshold
sensitivity at 0.1, 0.5, 1, 3 and 5 in the stored load units.
"""
from __future__ import annotations
import io, json, os, tarfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import h5py, fsspec
from huggingface_hub import hf_hub_download, hf_hub_url
from extract_touchscale_dynamic_structure import extract_hand, N_BINS, N_PADS

REPO = "2077AIDataFoundation/TouchScale"
TASK = "Operate the lever switch once"
EPISODES_PER_ACTOR = 3
OUT = Path("artifacts/touchscale_nine_episode_qc")

def threshold_audit(group, hand, episode_id, actor_id):
    pads = sorted((k for k in group if k.startswith("pad_")),
                  key=lambda k: int(k.split("_")[1]))
    if len(pads) != N_PADS:
        return []
    arrays = [np.asarray(group[k][()], dtype=np.float32) for k in pads]
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        return []
    t = lengths.pop()
    loads = np.column_stack([
        np.maximum(a.reshape(t, -1), 0).sum(axis=1) for a in arrays
    ])
    if not np.isfinite(loads).all():
        return []
    total = loads.sum(axis=1)
    rows = []
    for threshold in (0.1, 0.5, 1.0, 3.0, 5.0):
        active = total > threshold
        rows.append({
            "episode_id": episode_id, "actor_id": actor_id, "hand": hand,
            "threshold": threshold, "active_fraction": float(active.mean()),
            "active_frames": int(active.sum()), "frame_count": int(t),
            "median_total_load": float(np.median(total)),
            "p95_total_load": float(np.quantile(total, .95)),
            "max_total_load": float(np.max(total)),
            "all_zero_load": bool(np.all(total <= 0)),
        })
    return rows

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
        raise RuntimeError(f"Task family absent: {TASK}")
    counts = candidates.groupby("actor_id").episode_id.nunique()
    actors = sorted(counts[counts >= EPISODES_PER_ACTOR].index.tolist())
    if len(actors) < 3:
        raise RuntimeError(f"Expected >=3 actors with >=3 episodes, found {actors}")
    # Preserve the three actors used in the six-episode pilot; choose top-quality
    # records within each actor so the extension is directly comparable.
    prior_actors = ["actor_01", "actor_02", "actor_23"]
    actors = [a for a in prior_actors if a in actors]
    if len(actors) != 3:
        raise RuntimeError(f"Prior pilot actor set unavailable: {actors}")
    selected = pd.concat([
        candidates[candidates.actor_id == actor].sort_values(
            ["quality_score", "episode_id"], ascending=[False, True]
        ).head(EPISODES_PER_ACTOR)
        for actor in actors
    ], ignore_index=True)
    if selected.episode_id.nunique() != 9:
        raise RuntimeError("Nine unique episodes were not selected")
    selected.to_csv(OUT/"selected_episode_manifest.csv", index=False)

    by_shard = defaultdict(list)
    for row in selected.to_dict("records"):
        by_shard[str(row["shard"])].append(row)
    hand_summaries, features, threshold_rows, details = [], [], [], {}
    started = time.time()
    for shard, rows in sorted(by_shard.items()):
        url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
        fs = fsspec.filesystem("http", block_size=8*1024*1024,
                               headers={"Authorization": f"Bearer {token}"})
        shard_start = time.time()
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
                    eid, actor = str(row["episode_id"]), str(row["actor_id"])
                    meta = {k: str(row[k]) for k in ("episode_id","scene","task_name","actor_id","collection_date")}
                    hands = {}
                    with h5py.File(io.BytesIO(blob), "r") as h5:
                        for hand in ("left","right"):
                            key = f"obs/tactile/raw/{hand}"
                            if key not in h5:
                                result, out_rows = {"status":"missing_hand"}, []
                            else:
                                result, out_rows = extract_hand(h5[key], hand, meta)
                                threshold_rows.extend(threshold_audit(h5[key], hand, eid, actor))
                            hands[hand] = result
                            features.extend(out_rows)
                            hand_summaries.append({
                                "episode_id":eid,"actor_id":actor,
                                "collection_date":meta["collection_date"],
                                "hand":hand,**result
                            })
                    details[eid] = {"actor_id":actor,"shard":shard,
                                    "h5_bytes":len(blob),"hands":hands}
                    found.add(base)
                    if found == wanted:
                        break
                missing = sorted(wanted-found)
                if missing:
                    raise RuntimeError(f"Missing HDF5 members in {shard}: {missing}")
        details["_shard_"+shard] = {
            "size_bytes":shard_size,
            "range_read_seconds":round(time.time()-shard_start,2)
        }

    hs = pd.DataFrame(hand_summaries)
    ft = pd.DataFrame(features)
    th = pd.DataFrame(threshold_rows)
    hs.to_csv(OUT/"episode_hand_summary.csv", index=False)
    ft.to_csv(OUT/"time_bin_dynamic_features.csv", index=False)
    th.to_csv(OUT/"threshold_sensitivity.csv", index=False)
    (OUT/"details.json").write_text(json.dumps(details,indent=2),encoding="utf-8")

    actor_summary = hs.groupby(["actor_id","hand"]).agg(
        episodes=("episode_id","nunique"),
        median_active_fraction=("active_fraction","median"),
        zero_contact_episodes=("active_fraction",lambda x:int((x==0).sum())),
        median_rate_hz=("effective_rate_hz","median"),
        median_entropy=("median_bin_entropy_norm","median")
    ).reset_index()
    actor_summary.to_csv(OUT/"actor_hand_summary.csv",index=False)
    threshold_summary = th.groupby(["actor_id","hand","threshold"]).agg(
        median_active_fraction=("active_fraction","median"),
        zero_contact_episodes=("all_zero_load","sum"),
        median_p95_load=("p95_total_load","median")
    ).reset_index()
    threshold_summary.to_csv(OUT/"threshold_summary.csv",index=False)

    valid = hs[hs.status=="ok"]
    result = {
        "status":"success" if len(valid)==18 and len(ft)==18*N_BINS and len(th)==18*5 else "partial",
        "task_name":TASK,"episodes":int(selected.episode_id.nunique()),
        "participants":int(selected.actor_id.nunique()),
        "episodes_per_participant":EPISODES_PER_ACTOR,
        "shards":int(len(by_shard)),
        "successful_hand_records":int(len(valid)),
        "feature_rows":int(len(ft)),"expected_feature_rows":18*N_BINS,
        "threshold_rows":int(len(th)),"expected_threshold_rows":18*5,
        "median_sampling_rate_by_actor":hs.groupby("actor_id").effective_rate_hz.median().to_dict(),
        "elapsed_seconds":round(time.time()-started,2),
        "scope_limit":"QC pilot only; not an independent participant/date generalization test or meta-learning evaluation."
    }
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="success":
        raise RuntimeError("Nine-episode QC pilot produced incomplete feature or threshold records")

if __name__=="__main__":
    main()
