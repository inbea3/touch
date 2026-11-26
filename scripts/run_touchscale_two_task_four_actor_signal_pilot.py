#!/usr/bin/env python3
"""Eight-episode, two-task, four-actor TouchScale signal/structure pilot.

Downloads only the selected HDF5 members via ranged reads from the required
WebDataset shards. No video is extracted or retained; no model is trained.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, hf_hub_url

from extract_touchscale_dynamic_structure import extract_hand, N_BINS

REPO = "2077AIDataFoundation/TouchScale"
OUT = Path("artifacts/touchscale_two_task_four_actor_signal_pilot")
SEEDS = (17, 29, 43)

TASKS = [
    {
        "scene": "Laboratory",
        "task_name": "Transfer a large amount of powder into the beaker with a lab scoop (basic scooping and tapping)",
        "actors": ["actor_06", "actor_12"],
    },
    {
        "scene": "Lab Apparatus & Assembly Workbench",
        "task_name": "Use a trial tool to insert the rod into the rack hole",
        "actors": ["actor_07", "actor_19"],
    },
]


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return None
    a, b = a[ok], b[ok]
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else None


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True, exist_ok=True)

    metadata_path = hf_hub_download(
        repo_id=REPO, filename="metadata/episodes.parquet",
        repo_type="dataset", token=token
    )
    metadata = pd.read_parquet(metadata_path)
    required = {"episode_id", "scene", "task_name", "actor_id", "collection_date", "shard", "quality_score"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise RuntimeError(f"Metadata missing required fields: {missing}")

    selected_rows = []
    for task in TASKS:
        candidates = metadata[
            (metadata.scene.astype(str) == task["scene"]) &
            (metadata.task_name.astype(str) == task["task_name"])
        ].copy()
        for actor in task["actors"]:
            actor_rows = candidates[candidates.actor_id.astype(str) == actor].copy()
            if len(actor_rows) < 2:
                raise RuntimeError(
                    f"Need >=2 episodes for {task['task_name']} / {actor}; found {len(actor_rows)}"
                )
            actor_rows["quality_score"] = pd.to_numeric(actor_rows.quality_score, errors="coerce")
            chosen = actor_rows.sort_values(
                ["quality_score", "episode_id"], ascending=[False, True]
            ).head(2)
            for row in chosen.to_dict("records"):
                selected_rows.append({
                    "episode_id": str(row["episode_id"]),
                    "scene": task["scene"],
                    "task_name": task["task_name"],
                    "actor_id": actor,
                    "collection_date": str(row["collection_date"]),
                    "quality_score": float(row["quality_score"]),
                    "shard": str(row["shard"]),
                })

    selected = pd.DataFrame(selected_rows).drop_duplicates("episode_id")
    if len(selected) != 8 or selected.actor_id.nunique() != 4 or selected.task_name.nunique() != 2:
        raise RuntimeError("Selection did not produce exactly 8 episodes / 4 actors / 2 tasks")
    selected.to_csv(OUT / "selected_episode_manifest.csv", index=False)

    by_shard = defaultdict(list)
    for row in selected.to_dict("records"):
        by_shard[row["shard"]].append(row)

    summaries, feature_rows = [], []
    shard_audit = []
    started_all = time.time()
    for shard, rows in sorted(by_shard.items()):
        url = hf_hub_url(repo_id=REPO, filename=shard, repo_type="dataset")
        fs = fsspec.filesystem(
            "http", block_size=8 * 1024 * 1024,
            headers={"Authorization": f"Bearer {token}"}
        )
        shard_started = time.time()
        wanted = {r["episode_id"] + ".episode.h5" for r in rows}
        found = set()
        with fs.open(url, "rb") as remote:
            shard_size = int(remote.size)
            with tarfile.open(fileobj=remote, mode="r:") as tar:
                for member in tar:
                    base = Path(member.name).name
                    if not member.isfile() or base not in wanted:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    blob = source.read()
                    row = next(r for r in rows if r["episode_id"] + ".episode.h5" == base)
                    episode_meta = {
                        k: row[k] for k in (
                            "episode_id", "scene", "task_name", "actor_id", "collection_date"
                        )
                    }
                    hand_results = {}
                    with h5py.File(io.BytesIO(blob), "r") as h5:
                        for hand in ("left", "right"):
                            key = f"obs/tactile/raw/{hand}"
                            if key not in h5:
                                result, rows_out = {"status": "missing_hand"}, []
                            else:
                                result, rows_out = extract_hand(h5[key], hand, episode_meta)
                            hand_results[hand] = result
                            feature_rows.extend(rows_out)
                            summaries.append({
                                **episode_meta, "hand": hand,
                                "quality_score": row["quality_score"],
                                "shard": shard, "h5_member_bytes": len(blob), **result
                            })
                    found.add(base)
                    if found == wanted:
                        break
        missing_members = sorted(wanted - found)
        if missing_members:
            raise RuntimeError(f"Missing HDF5 members in {shard}: {missing_members}")
        shard_audit.append({
            "shard": shard, "selected_episodes": len(rows),
            "shard_size_bytes": shard_size,
            "range_read_seconds": round(time.time() - shard_started, 2),
        })

    summary_df = pd.DataFrame(summaries)
    feature_df = pd.DataFrame(feature_rows)
    summary_df.to_csv(OUT / "episode_hand_dynamic_summary.csv", index=False)
    feature_df.to_csv(OUT / "time_bin_dynamic_features.csv", index=False)
    pd.DataFrame(shard_audit).to_csv(OUT / "shard_access_audit.csv", index=False)

    # Descriptive within-actor repeat and between-actor structure similarity.
    pad_share_cols = [f"pad_share_active_{i:02d}" for i in range(15)]
    repeatability = []
    for task_name, task_features in feature_df.groupby("task_name"):
        episode_vectors = {}
        episode_meta = {}
        for (episode_id, actor_id, hand), g in task_features.groupby(["episode_id", "actor_id", "hand"]):
            if not set(pad_share_cols).issubset(g.columns):
                continue
            vec = g[pad_share_cols].mean(axis=0, skipna=True).to_numpy(dtype=float)
            episode_vectors[(episode_id, actor_id, hand)] = vec
            episode_meta[episode_id] = actor_id
        actors = sorted(task_features.actor_id.unique())
        for hand in ("left", "right"):
            for actor in actors:
                eps = [k for k in episode_vectors if k[1] == actor and k[2] == hand]
                if len(eps) >= 2:
                    repeatability.append({
                        "task_name": task_name, "hand": hand, "comparison": "within_actor_repeat",
                        "actor_a": actor, "actor_b": actor,
                        "episode_a": eps[0][0], "episode_b": eps[1][0],
                        "cosine_similarity_mean_pad_share": cosine(
                            episode_vectors[eps[0]], episode_vectors[eps[1]]
                        ),
                    })
            actor_eps = {
                actor: [k for k in episode_vectors if k[1] == actor and k[2] == hand]
                for actor in actors
            }
            for i, actor_a in enumerate(actors):
                for actor_b in actors[i+1:]:
                    for ea in actor_eps[actor_a]:
                        for eb in actor_eps[actor_b]:
                            repeatability.append({
                                "task_name": task_name, "hand": hand, "comparison": "between_actor",
                                "actor_a": actor_a, "actor_b": actor_b,
                                "episode_a": ea[0], "episode_b": eb[0],
                                "cosine_similarity_mean_pad_share": cosine(
                                    episode_vectors[ea], episode_vectors[eb]
                                ),
                            })
    rep_df = pd.DataFrame(repeatability)
    rep_df.to_csv(OUT / "descriptive_structure_similarity.csv", index=False)

    valid = summary_df[summary_df.status == "ok"]
    result = {
        "status": "success" if len(valid) == 16 and len(feature_df) == 8 * 2 * N_BINS else "partial",
        "episodes": int(selected.episode_id.nunique()),
        "actors": int(selected.actor_id.nunique()),
        "task_templates": int(selected.task_name.nunique()),
        "shards": int(len(by_shard)),
        "successful_hand_records": int(len(valid)),
        "feature_rows": int(len(feature_df)),
        "expected_feature_rows": int(8 * 2 * N_BINS),
        "episode_counts_by_actor_task": selected.groupby(["task_name", "actor_id"]).size().rename("episodes").reset_index().to_dict("records"),
        "median_active_fraction_by_hand": valid.groupby("hand").active_fraction.median().to_dict() if len(valid) else {},
        "within_actor_repeat_pairs": int((rep_df.comparison == "within_actor_repeat").sum()) if len(rep_df) else 0,
        "between_actor_pairs": int((rep_df.comparison == "between_actor").sum()) if len(rep_df) else 0,
        "elapsed_seconds": round(time.time() - started_all, 2),
        "limits": [
            "Metadata-selected high-quality episodes only; quality_score is not a substitute for tactile QC.",
            "Two episodes per actor-task group provide only one descriptive within-actor repeat pair.",
            "Cosine similarity is descriptive, not an inferential reliability statistic.",
            "No model training, no video extraction, and no physical object-instance claim."
        ]
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "success":
        raise RuntimeError("The 8-episode signal/structure pilot was incomplete")


if __name__ == "__main__":
    main()
