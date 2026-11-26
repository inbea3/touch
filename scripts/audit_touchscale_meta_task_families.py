#!/usr/bin/env python3
"""Audit TouchScale task families for valid cross-participant few-shot episodes.

Metadata-only: no HDF5 shards are downloaded. Ranks task families by number
of participants with repeated episodes and reports collection-date confounds.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
OUT = Path("artifacts/touchscale_meta_task_audit")

def main():
    token=os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True,exist_ok=True)
    p=hf_hub_download(repo_id=REPO,filename="metadata/episodes.parquet",
                     repo_type="dataset",token=token)
    df=pd.read_parquet(p)
    required={"episode_id","task_name","actor_id","shard","collection_date","quality_score"}
    missing=required-set(df.columns)
    if missing:
        raise RuntimeError(f"Metadata columns missing: {sorted(missing)}")
    df=df.drop_duplicates("episode_id").copy()
    df["actor_id"]=df.actor_id.astype(str)
    df["task_name"]=df.task_name.astype(str)
    # Per task/participant support and independent session coverage.
    ap=df.groupby(["task_name","actor_id"]).agg(
        episodes=("episode_id","nunique"),
        dates=("collection_date","nunique"),
        shards=("shard","nunique"),
        median_quality=("quality_score","median")
    ).reset_index()
    task=ap.groupby("task_name").agg(
        participants=("actor_id","nunique"),
        total_episodes=("episodes","sum"),
        actors_with_2plus=("episodes",lambda x:int((x>=2).sum())),
        actors_with_3plus=("episodes",lambda x:int((x>=3).sum())),
        actors_with_multiple_dates=("dates",lambda x:int((x>=2).sum())),
        min_episodes_per_actor=("episodes","min"),
        median_episodes_per_actor=("episodes","median")
    ).reset_index()
    task["eligible_3actors_x2episodes"]=(task.participants>=3)&(task.actors_with_2plus>=3)
    task["eligible_3actors_x3episodes"]=(task.participants>=3)&(task.actors_with_3plus>=3)
    task=task.sort_values(
        ["eligible_3actors_x3episodes","eligible_3actors_x2episodes",
         "actors_with_multiple_dates","participants","total_episodes"],
        ascending=False
    )
    ap.to_csv(OUT/"task_actor_coverage.csv",index=False)
    task.to_csv(OUT/"task_family_rankings.csv",index=False)
    # Focus on families that could form meta-learning tasks, retain all actors.
    candidates=task[task.eligible_3actors_x2episodes].head(30)
    if len(candidates):
        ap[ap.task_name.isin(candidates.task_name)].to_csv(
            OUT/"eligible_task_actor_detail.csv",index=False)
    result={
        "status":"success","episodes":int(df.episode_id.nunique()),
        "task_families":int(df.task_name.nunique()),
        "participants":int(df.actor_id.nunique()),
        "eligible_families_3actors_2episodes":int(task.eligible_3actors_x2episodes.sum()),
        "eligible_families_3actors_3episodes":int(task.eligible_3actors_x3episodes.sum()),
        "top_task_families":task.head(15).to_dict("records"),
        "note":"Metadata-only screening. Eligibility does not prove raw tactile signal quality or independence; follow-up HDF5 QC is required."
    }
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
