#!/usr/bin/env python3
"""Build metadata-only LOTO/LOAO split-feasibility manifests for TouchScale.

No raw tactile shards or video are downloaded. The grouping unit is the
(scene, task_name) pair because task_id is not globally reliable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
OUT = Path("artifacts/touchscale_split_feasibility")
SHOTS = (1, 2, 5)
SEEDS = (17, 29, 43)


def task_key(scene: object, task_name: object) -> str:
    raw = f"{str(scene).strip()}\x1f{str(task_name).strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_split(ids: list[str], shots: int, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    support = [ids[int(i)] for i in order[:shots]]
    query = [ids[int(i)] for i in order[shots:]]
    return support, query


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)

    metadata_path = hf_hub_download(
        repo_id=REPO,
        filename="metadata/episodes.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(metadata_path)
    required = {
        "episode_id", "scene", "task_name", "actor_id",
        "collection_date", "shard", "quality_score",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")

    df = df.copy()
    df["episode_id"] = df["episode_id"].astype(str)
    for col in ["scene", "task_name", "actor_id", "collection_date", "shard"]:
        df[col] = df[col].map(safe_str)
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    if df["episode_id"].duplicated().any():
        dup = int(df["episode_id"].duplicated().sum())
        raise RuntimeError(f"Duplicate episode_id rows found: {dup}")

    invalid = df[
        (df["scene"] == "") | (df["task_name"] == "") |
        (df["actor_id"] == "")
    ].copy()
    invalid.to_csv(OUT / "invalid_metadata_rows.csv", index=False)
    df = df[
        (df["scene"] != "") & (df["task_name"] != "") &
        (df["actor_id"] != "")
    ].copy()
    df["task_key"] = [
        task_key(scene, name) for scene, name in zip(df["scene"], df["task_name"])
    ]

    # Task-template census and actor-by-task support.
    actor_task = (
        df.groupby(["task_key", "scene", "task_name", "actor_id"], dropna=False)
        .agg(
            episodes=("episode_id", "nunique"),
            dates=("collection_date", "nunique"),
            shards=("shard", "nunique"),
            median_quality=("quality_score", "median"),
            episode_ids=("episode_id", lambda s: ";".join(sorted(s.astype(str)))),
        )
        .reset_index()
    )
    task_census = (
        actor_task.groupby(["task_key", "scene", "task_name"], dropna=False)
        .agg(
            actors=("actor_id", "nunique"),
            episodes=("episodes", "sum"),
            actors_2plus=("episodes", lambda s: int((s >= 2).sum())),
            actors_3plus=("episodes", lambda s: int((s >= 3).sum())),
            actors_6plus=("episodes", lambda s: int((s >= 6).sum())),
            median_actor_episodes=("episodes", "median"),
            max_actor_episodes=("episodes", "max"),
            median_quality=("median_quality", "median"),
        )
        .reset_index()
    )
    actor_task.to_csv(OUT / "actor_task_coverage.csv", index=False)
    task_census.to_csv(OUT / "task_template_census.csv", index=False)

    # LOTO: hold out the entire (scene, task_name) template.
    loto_rows = []
    for row in task_census.itertuples(index=False):
        g = df[df["task_key"] == row.task_key].sort_values("episode_id")
        ids = g["episode_id"].tolist()
        train_meta_tasks = int(
            actor_task.loc[actor_task["task_key"] != row.task_key]
            .query("episodes >= 2")["task_key"].nunique()
        )
        for shot in SHOTS:
            eligible = len(ids) >= shot + 1
            if not eligible:
                continue
            for seed in SEEDS:
                support, query = make_split(ids, shot, seed + shot * 1009)
                loto_rows.append({
                    "task_key": row.task_key,
                    "scene": row.scene,
                    "task_name": row.task_name,
                    "shots": shot,
                    "seed": seed,
                    "support_episode_ids": ";".join(support),
                    "query_episode_ids": ";".join(query),
                    "support_n": len(support),
                    "query_n": len(query),
                    "task_actors": int(row.actors),
                    "train_meta_task_count": train_meta_tasks,
                    "split_type": "LOTO_task_template",
                    "actor_disjoint_support_query": False,
                })
    loto = pd.DataFrame(loto_rows)
    loto.to_csv(OUT / "loto_support_query_manifest.csv", index=False)

    # LOAO: hold out one actor globally; evaluate only shared task templates.
    # For a test actor-task pair, another actor must have >=2 episodes of that
    # same template, and the held-out actor must have shot+query episodes.
    loao_rows = []
    for actor in sorted(df["actor_id"].unique()):
        held_actor = df[df["actor_id"] == actor]
        held_groups = (
            held_actor.groupby(["task_key", "scene", "task_name"])
            .agg(
                episodes=("episode_id", "nunique"),
                episode_ids=("episode_id", lambda s: ";".join(sorted(s.astype(str)))),
            )
            .reset_index()
        )
        for row in held_groups.itertuples(index=False):
            train_same_task = actor_task[
                (actor_task["task_key"] == row.task_key) &
                (actor_task["actor_id"] != actor) &
                (actor_task["episodes"] >= 2)
            ]
            if train_same_task.empty:
                continue
            ids = row.episode_ids.split(";")
            train_ids = actor_task[
                (actor_task["actor_id"] != actor) &
                (actor_task["episodes"] >= 2)
            ]
            train_task_count = int(train_ids["task_key"].nunique())
            train_actor_task_count = int(len(train_ids))
            train_same_task_ids = ";".join(train_same_task["episode_ids"].tolist())
            for shot in SHOTS:
                if len(ids) < shot + 1:
                    continue
                for seed in SEEDS:
                    support, query = make_split(ids, shot, seed + shot * 1009)
                    loao_rows.append({
                        "heldout_actor": actor,
                        "task_key": row.task_key,
                        "scene": row.scene,
                        "task_name": row.task_name,
                        "shots": shot,
                        "seed": seed,
                        "support_episode_ids": ";".join(support),
                        "query_episode_ids": ";".join(query),
                        "support_n": len(support),
                        "query_n": len(query),
                        "train_reference_actors": int(train_same_task["actor_id"].nunique()),
                        "train_reference_episodes": int(train_same_task["episodes"].sum()),
                        "train_reference_episode_ids": train_same_task_ids,
                        "train_meta_task_count": train_task_count,
                        "train_actor_task_group_count": train_actor_task_count,
                        "split_type": "LOAO_actor_task",
                        "heldout_actor_excluded_from_training": True,
                    })
    loao = pd.DataFrame(loao_rows)
    loao.to_csv(OUT / "loao_support_query_manifest.csv", index=False)

    def shot_summary(frame: pd.DataFrame, key: str) -> dict:
        result = {}
        for shot in SHOTS:
            part = frame[frame["shots"] == shot] if not frame.empty else frame
            result[str(shot)] = {
                "manifest_rows": int(len(part)),
                "unique_test_units": int(part[key].nunique()) if len(part) else 0,
                "unique_seeds": int(part["seed"].nunique()) if len(part) else 0,
            }
        return result

    actor_summary = (
        loao.groupby(["heldout_actor", "shots"])
        .agg(
            eligible_task_templates=("task_key", "nunique"),
            manifest_rows=("task_key", "size"),
        )
        .reset_index()
    )
    actor_summary.to_csv(OUT / "loao_actor_summary.csv", index=False)

    summary = {
        "status": "success",
        "dataset": REPO,
        "episodes": int(df["episode_id"].nunique()),
        "task_templates_scene_task": int(task_census["task_key"].nunique()),
        "actors": int(df["actor_id"].nunique()),
        "invalid_metadata_rows": int(len(invalid)),
        "task_templates_by_actor_count": {
            str(int(k)): int(v)
            for k, v in task_census["actors"].value_counts().sort_index().items()
        },
        "task_templates_with_at_least_two_actors_and_each_actor_episodes": {
            str(k): int(((task_census["actors"] >= 2) &
                         (task_census[f"actors_{k}plus"] >= 2)).sum())
            for k in (2, 3, 6)
        },
        "loto": shot_summary(loto, "task_key"),
        "loao": shot_summary(loao, "heldout_actor"),
        "loao_heldout_actors": sorted(loao["heldout_actor"].unique().tolist())
        if not loao.empty else [],
        "notes": [
            "LOTO holds out the entire (scene, task_name) template; actor overlap across other tasks is allowed and must not be called unseen-actor evaluation.",
            "LOAO holds out all episodes from one actor; only shared task templates with another actor having at least two episodes are eligible.",
            "Support and Query are split by complete Episode ID, without replacement.",
            "This is metadata-only feasibility. It does not certify raw tactile signal quality or physical object-instance identity.",
            "task_id is not used as a global grouping key because the full metadata audit found collisions."
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
