#!/usr/bin/env python3
"""Audit TouchScale episode/task/object metadata without downloading videos or raw tactile shards.

Requires an accepted Hugging Face dataset agreement and HF_TOKEN.
Outputs metadata-only CSV/JSON files under artifacts/touchscale_full_metadata_identity/.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
OUT = Path("artifacts/touchscale_full_metadata_identity")


def norm(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def as_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    # Metadata may be serialized as a Python/NumPy-like string array.
    quoted = text.split("'")[1::2]
    if quoted:
        return [x.strip() for x in quoted if x.strip()]
    try:
        import ast
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except Exception:
        pass
    return [text]


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Set HF_TOKEN to an accepted Hugging Face Read token.")
    OUT.mkdir(parents=True, exist_ok=True)

    episodes_path = hf_hub_download(
        repo_id=REPO, filename="metadata/episodes.parquet",
        repo_type="dataset", token=os.environ["HF_TOKEN"]
    )
    objects_path = hf_hub_download(
        repo_id=REPO, filename="metadata/objects.parquet",
        repo_type="dataset", token=os.environ["HF_TOKEN"]
    )
    episodes = pd.read_parquet(episodes_path)
    objects = pd.read_parquet(objects_path)

    required = {"episode_id", "task_id", "task_name", "scene", "items",
                "actor_id", "collection_date"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise RuntimeError(f"episodes.parquet missing expected fields: {missing}")

    episodes["items_parsed"] = episodes["items"].apply(as_items)
    episodes["item_count"] = episodes["items_parsed"].str.len()
    episodes["task_id_text"] = episodes["task_id"].astype("string")
    episodes["task_name_norm"] = episodes["task_name"].map(norm)

    id_audit = (
        episodes.groupby("task_id_text", dropna=False)
        .agg(episodes=("episode_id", "nunique"),
             task_names=("task_name", "nunique"),
             scenes=("scene", "nunique"),
             actors=("actor_id", "nunique"),
             dates=("collection_date", "nunique"))
        .reset_index()
    )
    id_audit["multiple_task_names"] = id_audit["task_names"] > 1
    id_audit["multiple_scenes"] = id_audit["scenes"] > 1

    scene_id_audit = (
        episodes.groupby(["scene", "task_id_text"], dropna=False)
        .agg(episodes=("episode_id", "nunique"),
             task_names=("task_name", "nunique"),
             actors=("actor_id", "nunique"),
             dates=("collection_date", "nunique"))
        .reset_index()
    )
    scene_id_audit["multiple_task_names"] = scene_id_audit["task_names"] > 1

    task_name_audit = (
        episodes.groupby("task_name", dropna=False)
        .agg(episodes=("episode_id", "nunique"),
             task_ids=("task_id_text", "nunique"),
             scenes=("scene", "nunique"),
             actors=("actor_id", "nunique"),
             dates=("collection_date", "nunique"))
        .reset_index()
    )
    repeated_task_names = task_name_audit[task_name_audit["episodes"] > 1].copy()

    object_required = {"object_id", "object_name"}
    object_missing = sorted(object_required - set(objects.columns))
    if object_missing:
        raise RuntimeError(f"objects.parquet missing expected fields: {object_missing}")
    objects["object_name_norm"] = objects["object_name"].map(norm)
    object_name_to_id = dict(zip(objects["object_name_norm"], objects["object_id"].astype(str)))

    mentions = []
    for row in episodes[["episode_id", "task_id_text", "task_name", "scene",
                         "actor_id", "collection_date", "items_parsed"]].to_dict("records"):
        for item in row["items_parsed"]:
            key = norm(item)
            mentions.append({
                "episode_id": row["episode_id"],
                "task_id": row["task_id_text"],
                "task_name": row["task_name"],
                "scene": row["scene"],
                "actor_id": row["actor_id"],
                "collection_date": row["collection_date"],
                "item_label": item,
                "item_label_norm": key,
                "exact_object_catalogue_match": key in object_name_to_id,
                "candidate_object_id": object_name_to_id.get(key)
            })
    item_mentions = pd.DataFrame(mentions)
    if item_mentions.empty:
        item_mentions = pd.DataFrame(columns=[
            "episode_id", "task_id", "task_name", "scene", "actor_id",
            "collection_date", "item_label", "item_label_norm",
            "exact_object_catalogue_match", "candidate_object_id"
        ])

    episode_export = episodes.copy()
    episode_export["items_parsed"] = episode_export["items_parsed"].apply(
        lambda xs: " | ".join(xs)
    )
    keep = [c for c in [
        "episode_id", "task_id", "task_name", "scene", "actor_id",
        "collection_date", "items", "items_parsed", "item_count",
        "shard", "duration_s", "num_frames", "quality_score"
    ] if c in episode_export.columns]
    episode_export[keep].to_csv(OUT / "episodes_metadata_audit.csv", index=False)
    objects.to_csv(OUT / "object_mesh_catalogue.csv", index=False)
    id_audit.to_csv(OUT / "task_id_global_audit.csv", index=False)
    scene_id_audit.to_csv(OUT / "scene_task_id_audit.csv", index=False)
    repeated_task_names.to_csv(OUT / "repeated_task_names.csv", index=False)
    item_mentions.to_csv(OUT / "episode_item_object_catalogue_candidates.csv", index=False)

    summary = {
        "repo_id": REPO,
        "episode_rows": int(len(episodes)),
        "unique_episode_ids": int(episodes["episode_id"].nunique()),
        "unique_task_ids": int(episodes["task_id"].nunique(dropna=True)),
        "unique_task_names": int(episodes["task_name"].nunique(dropna=True)),
        "unique_scenes": int(episodes["scene"].nunique(dropna=True)),
        "unique_actors": int(episodes["actor_id"].nunique(dropna=True)),
        "object_mesh_rows": int(len(objects)),
        "object_mesh_unique_ids": int(objects["object_id"].nunique()),
        "episode_rows_with_empty_items": int((episodes["item_count"] == 0).sum()),
        "episode_rows_with_nonempty_items": int((episodes["item_count"] > 0).sum()),
        "task_ids_with_multiple_task_names": int(id_audit["multiple_task_names"].sum()),
        "task_ids_with_multiple_scenes": int(id_audit["multiple_scenes"].sum()),
        "scene_task_id_pairs_with_multiple_names": int(scene_id_audit["multiple_task_names"].sum()),
        "repeated_exact_task_names": int(len(repeated_task_names)),
        "item_mentions": int(len(item_mentions)),
        "exact_item_label_matches_to_mesh_catalogue": int(item_mentions["exact_object_catalogue_match"].sum()),
        "important_caveat": (
            "objects.parquet describes 3D mesh objects; episode items are free-text involved-object lists. "
            "An exact name match is only a candidate association, not proof that the mesh is the physical "
            "instance touched in that episode. Do not use it as an instance-level split key."
        ),
        "no_video_or_raw_tactile_data_downloaded": True
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
