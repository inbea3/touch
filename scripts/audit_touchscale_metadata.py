#!/usr/bin/env python3
"""Metadata-only audit for the gated TouchScale dataset.

Requires accepted dataset terms and HF_TOKEN (or an already authenticated
huggingface_hub login). Downloads only metadata/*.parquet and the two
metadata geometry JSON files; it never downloads raw episode shards.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "2077AIDataFoundation/TouchScale"
FILES = {
    "episodes": "metadata/episodes.parquet",
    "objects": "metadata/objects.parquet",
    "left_geometry": "metadata/glove_geometry_left.json",
    "right_geometry": "metadata/glove_geometry_right.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_items(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
            if isinstance(parsed, dict):
                return [str(x).strip() for x in parsed.values() if str(x).strip()]
        except Exception:
            pass
    # Preserve raw free text as one unresolved identity; do not split on
    # punctuation heuristically because that can silently create false IDs.
    return [text]


def geometry_summary(obj):
    def walk(x, path="$"):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            yield (path, len(x), x[:2])
    candidates = []
    for path, n, preview in walk(obj):
        if n > 0:
            candidates.append({"path": path, "length": n, "preview": preview})
    return candidates[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="artifacts/touchscale_metadata_audit")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    downloaded = {}
    for key, filename in FILES.items():
        path = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=args.revision,
            token=token,
        ))
        downloaded[key] = path

    episodes = pd.read_parquet(downloaded["episodes"])
    objects = pd.read_parquet(downloaded["objects"])
    required = ["episode_id", "task_id", "task_name", "scene", "items",
                "primary_hand", "duration_s", "actor_id", "glove_rate_hz",
                "tactile_contact_frac_left", "tactile_contact_frac_right", "shard"]
    missing = [c for c in required if c not in episodes.columns]
    if missing:
        raise RuntimeError(f"episodes.parquet missing expected fields: {missing}")

    episodes["parsed_items"] = episodes["items"].map(parse_items)
    episodes["n_items"] = episodes["parsed_items"].map(len)
    counts = Counter(item for row in episodes["parsed_items"] for item in row)
    item_rows = []
    for item, n in counts.most_common():
        subset = episodes[episodes["parsed_items"].map(lambda xs: item in xs)]
        item_rows.append({
            "raw_item_label": item,
            "episode_count": int(n),
            "unique_actors": int(subset["actor_id"].nunique(dropna=True)),
            "unique_tasks": int(subset["task_id"].nunique(dropna=True)),
            "unique_scenes": int(subset["scene"].nunique(dropna=True)),
            "multi_item_episode_count": int((subset["n_items"] > 1).sum()),
            "unique_episode_ids": int(subset["episode_id"].nunique(dropna=True)),
        })
    item_df = pd.DataFrame(item_rows)
    item_df.to_csv(out / "episode_item_counts.csv", index=False)

    eligible = {}
    for k in (1, 2, 5, 10):
        eligible[str(k)] = int((item_df["episode_count"] >= k + 1).sum()) if not item_df.empty else 0

    geoms = {}
    for side in ("left_geometry", "right_geometry"):
        with downloaded[side].open() as f:
            geoms[side] = geometry_summary(json.load(f))

    summary = {
        "dataset": REPO_ID,
        "revision": args.revision,
        "episode_rows": int(len(episodes)),
        "unique_episode_ids": int(episodes["episode_id"].nunique(dropna=True)),
        "duplicate_episode_id_rows": int(episodes["episode_id"].duplicated().sum()),
        "unique_task_ids": int(episodes["task_id"].nunique(dropna=True)),
        "unique_actor_ids": int(episodes["actor_id"].nunique(dropna=True)),
        "unique_scenes": int(episodes["scene"].nunique(dropna=True)),
        "unique_raw_item_labels": int(len(counts)),
        "episodes_with_no_item_label": int((episodes["n_items"] == 0).sum()),
        "episodes_with_multiple_item_labels": int((episodes["n_items"] > 1).sum()),
        "episodes_per_item_eligibility_naive": eligible,
        "episodes_columns": list(episodes.columns.drop("parsed_items")),
        "objects_rows": int(len(objects)),
        "objects_columns": list(objects.columns),
        "duration_s_quantiles": episodes["duration_s"].quantile([0, .25, .5, .75, 1]).to_dict(),
        "glove_rate_hz_counts": episodes["glove_rate_hz"].value_counts(dropna=False).to_dict(),
        "left_contact_fraction_quantiles": episodes["tactile_contact_frac_left"].quantile([0, .25, .5, .75, 1]).to_dict(),
        "right_contact_fraction_quantiles": episodes["tactile_contact_frac_right"].quantile([0, .25, .5, .75, 1]).to_dict(),
        "metadata_files": {k: {"path": str(v), "bytes": v.stat().st_size, "sha256": sha256(v)} for k, v in downloaded.items()},
        "geometry_structure_candidates": geoms,
        "warnings": [
            "Per-item episode counts above are naive counts from raw free-text items; they are not verified object identities.",
            "An episode may contain multiple objects; inspect instructions and labels before forming object-level tasks.",
            "Eligibility counts are upper bounds only and do not prove independent Support/Query episodes.",
            "The 193-row objects table describes reconstructed meshes and is not a complete canonical ID map for all episode item labels.",
            "No raw tactile shards or video files were downloaded."
        ],
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {out.resolve()}")


if __name__ == "__main__":
    main()
