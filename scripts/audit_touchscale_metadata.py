#!/usr/bin/env python3
"""Metadata-only TouchScale audit. Never downloads raw tactile/video shards."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
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
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def geometry_summary(obj):
    regions=obj.get("regions",[]) if isinstance(obj,dict) else []
    return {
        "top_level_keys": list(obj.keys()) if isinstance(obj,dict) else [],
        "region_count": len(regions),
        "regions": [
            {k:v for k,v in r.items() if k not in ("taxels","coordinates","positions")}
            for r in regions if isinstance(r,dict)
        ],
        "columns": obj.get("columns") if isinstance(obj,dict) else None,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="artifacts/touchscale_metadata_audit")
    ap.add_argument("--revision",default="main")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    token=os.environ.get("HF_TOKEN")
    files={}
    for key,filename in FILES.items():
        files[key]=Path(hf_hub_download(repo_id=REPO_ID,filename=filename,repo_type="dataset",revision=args.revision,token=token))
    ep=pd.read_parquet(files["episodes"]); objects=pd.read_parquet(files["objects"])
    required=["episode_id","task_id","task_name","scene","items","instruction","primary_hand","duration_s","num_frames","collection_date","actor_id","glove_rate_hz","tactile_contact_frac_left","tactile_contact_frac_right","quality_score","shard"]
    missing=[c for c in required if c not in ep.columns]
    if missing: raise RuntimeError(f"episodes.parquet missing fields: {missing}")

    # IMPORTANT: items is free text describing objects involved, often concatenated
    # without a reliable delimiter. Preserve it as an opaque episode-level bundle.
    ep["items_raw"]=ep["items"].fillna("").astype(str).str.strip()
    ep["items_bundle_norm"]=ep["items_raw"].str.lower().str.replace(r"\\s+"," ",regex=True)
    bundle=ep[ep["items_bundle_norm"]!=""].groupby("items_bundle_norm",dropna=False).agg(
        episode_count=("episode_id","nunique"),
        unique_actors=("actor_id","nunique"),
        unique_tasks=("task_id","nunique"),
        unique_scenes=("scene","nunique"),
        unique_dates=("collection_date","nunique"),
        unique_shards=("shard","nunique"),
        median_quality=("quality_score","median"),
        raw_label_examples=("items_raw",lambda x:" || ".join(list(dict.fromkeys(x.astype(str)))[:3])),
        task_examples=("task_name",lambda x:" || ".join(list(dict.fromkeys(x.astype(str)))[:3])),
        instruction_examples=("instruction",lambda x:" || ".join(list(dict.fromkeys(x.astype(str)))[:2])),
    ).reset_index().sort_values("episode_count",ascending=False)
    bundle.to_csv(out/"episode_item_bundle_counts.csv",index=False)

    # Export episode metadata for manual object-identity review; no tactile arrays/video.
    review_cols=[c for c in required if c in ep.columns]
    ep[review_cols].to_csv(out/"episode_metadata_review.csv",index=False)
    objects.to_csv(out/"mesh_object_registry.csv",index=False)

    geometries={}
    for side in ("left_geometry","right_geometry"):
        obj=json.loads(files[side].read_text())
        geometries[side]=geometry_summary(obj)
        shutil.copy2(files[side],out/(side+".json"))

    unique_ids=int(ep["episode_id"].nunique(dropna=True))
    summary={
        "dataset":REPO_ID,"revision":args.revision,
        "episode_rows":int(len(ep)),"unique_episode_ids":unique_ids,
        "duplicate_episode_id_rows":int(ep["episode_id"].duplicated().sum()),
        "unique_task_ids":int(ep["task_id"].nunique(dropna=True)),
        "unique_actor_ids":int(ep["actor_id"].nunique(dropna=True)),
        "unique_scenes":int(ep["scene"].nunique(dropna=True)),
        "unique_exact_item_bundles":int(ep.loc[ep["items_bundle_norm"]!="","items_bundle_norm"].nunique()),
        "episodes_missing_items":int((ep["items_bundle_norm"]=="").sum()),
        "episodes_with_repeated_item_bundle":int((ep["items_bundle_norm"].value_counts()>1).sum()),
        "bundle_episode_count_quantiles":bundle["episode_count"].quantile([0,.25,.5,.75,1]).to_dict() if len(bundle) else {},
        "episode_duration_s_quantiles":ep["duration_s"].quantile([0,.25,.5,.75,1]).to_dict(),
        "tactile_sampling_rate_counts":ep["glove_rate_hz"].round(1).value_counts(dropna=False).to_dict(),
        "left_contact_fraction_quantiles":ep["tactile_contact_frac_left"].quantile([0,.25,.5,.75,1]).to_dict(),
        "right_contact_fraction_quantiles":ep["tactile_contact_frac_right"].quantile([0,.25,.5,.75,1]).to_dict(),
        "quality_score_quantiles":ep["quality_score"].quantile([0,.25,.5,.75,1]).to_dict(),
        "objects_mesh_registry_rows":int(len(objects)),
        "objects_mesh_registry_columns":list(objects.columns),
        "episode_columns":list(ep.columns.drop(["items_raw","items_bundle_norm"])),
        "geometry":geometries,
        "metadata_file_integrity":{k:{"bytes":v.stat().st_size,"sha256":sha256(v)} for k,v in files.items()},
        "warnings":[
            "The items field is free-text episode-level object involvement, not a canonical object-ID list.",
            "Repeated exact item bundles are NOT proof that the same individual object instance or object category recurs.",
            "The 193-row mesh registry is partial and object_name does not always match episode items.",
            "No 1/2/5/10-shot object eligibility is reported: this requires a validated object identity map and independent episode review.",
            "No raw tactile shards or video files were downloaded."
        ]
    }
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    print(f"Wrote {out.resolve()}")

if __name__=="__main__": main()
