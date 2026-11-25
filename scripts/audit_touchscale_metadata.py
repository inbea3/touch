#!/usr/bin/env python3
"""Metadata-only TouchScale audit. Never downloads raw tactile/video shards."""
from __future__ import annotations
import argparse, base64, hashlib, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "2077AIDataFoundation/TouchScale"
FILES = {
    "episodes": "metadata/episodes.parquet",
    "objects": "metadata/objects.parquet",
    "left_geometry": "metadata/glove_geometry_left.json",
    "right_geometry": "metadata/glove_geometry_right.json",
}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def item_list(value):
    if value is None or (isinstance(value,float) and pd.isna(value)): return []
    if isinstance(value,np.ndarray): value=value.tolist()
    if isinstance(value,(list,tuple,set)): return [str(x).strip() for x in value if str(x).strip()]
    # Do not guess-split stringified arrays. Preserve as one unresolved value.
    s=str(value).strip()
    return [s] if s else []

def geometry_audit(path,out,side):
    obj=json.loads(Path(path).read_text())
    raw=base64.b64decode(obj["data_base64"])
    arr=np.frombuffer(raw,dtype=np.dtype(obj["dtype"]))
    n=int(obj["point_count"]); stride=int(obj["point_stride"])
    if arr.size != n*stride: raise ValueError(f"{side}: geometry payload size mismatch")
    mat=arr.reshape(n,stride)
    cols=obj["columns"]
    ix,iy,iz=[cols.index(c) for c in ("x","y","z")]
    xyz=mat[:,[ix,iy,iz]]
    regions=obj["regions"]
    region_rows=[]
    start=0
    for r in regions:
        count=int(r["rows"])*int(r["cols"])
        region_rows.append({"region_id":r["id"],"region":r["name"],"rows":r["rows"],"cols":r["cols"],"taxel_count":count,"start_index":start,"end_index_exclusive":start+count})
        start+=count
    if start!=n: raise ValueError(f"{side}: region taxel total {start} != point_count {n}")
    region_df=pd.DataFrame(region_rows)
    region_df.to_csv(out/f"{side}_geometry_regions.csv",index=False)
    geom={
        "schema":obj.get("schema"),"side":obj.get("side"),"point_count":n,"point_stride":stride,
        "columns":cols,"region_count":len(regions),"region_taxel_total":start,
        "xyz_finite":bool(np.isfinite(xyz).all()),
        "xyz_duplicate_rows":int(pd.DataFrame(xyz).duplicated().sum()),
        "xyz_min":xyz.min(axis=0).tolist(),"xyz_max":xyz.max(axis=0).tolist(),
        "xyz_std":xyz.std(axis=0).tolist(),
        "regions":region_rows,
        "note":obj.get("note")
    }
    # Preserve the official file for reproducibility.
    shutil.copy2(path,out/f"{side}_geometry.json")
    return geom

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

    ep["item_list"]=ep["items"].map(item_list)
    ep["n_items"]=ep["item_list"].map(len)
    ep["items_json"]=ep["item_list"].map(lambda x:json.dumps(x,ensure_ascii=False))
    ep["items_bundle_norm"]=ep["item_list"].map(lambda x="": " | ".join(y.lower().strip() for y in x))
    # Individual raw item mentions; these are object labels, not guaranteed unique physical instances.
    rows=[]
    all_items=sorted({item for xs in ep["item_list"] for item in xs})
    for item in all_items:
        sub=ep[ep["item_list"].map(lambda xs:item in xs)]
        rows.append({
            "raw_item_label":item,"episode_count":int(sub["episode_id"].nunique()),
            "unique_actors":int(sub["actor_id"].nunique()),
            "unique_tasks":int(sub["task_id"].nunique()),
            "unique_scenes":int(sub["scene"].nunique()),
            "unique_dates":int(sub["collection_date"].nunique()),
            "unique_shards":int(sub["shard"].nunique()),
            "median_quality":float(sub["quality_score"].median()),
            "task_examples":" || ".join(list(dict.fromkeys(sub["task_name"].astype(str)))[:5]),
        })
    item_df=pd.DataFrame(rows).sort_values("episode_count",ascending=False)
    item_df.to_csv(out/"individual_item_label_counts.csv",index=False)

    # Exact full item bundles are useful for review but are not object identities.
    bundle=(ep[ep["n_items"]>0].groupby("items_bundle_norm")
        .agg(episode_count=("episode_id","nunique"),unique_actors=("actor_id","nunique"),
             unique_tasks=("task_id","nunique"),unique_scenes=("scene","nunique"),
             raw_examples=("items_json",lambda x:" || ".join(list(dict.fromkeys(x.astype(str)))[:2])))
        .reset_index().sort_values("episode_count",ascending=False))
    bundle.to_csv(out/"episode_item_bundle_counts.csv",index=False)

    review_cols=required+["items_json","n_items"]
    ep[review_cols].to_csv(out/"episode_metadata_review.csv",index=False)
    objects.to_csv(out/"mesh_object_registry.csv",index=False)

    geom={side:geometry_audit(files[f"{side}_geometry"],out,side) for side in ("left","right")}
    eligibility={}
    for k in (1,2,5,10):
        eligible=item_df[item_df["episode_count"]>=k+1]
        eligibility[str(k)]={
            "raw_item_labels_with_at_least_k_plus_1_episodes":int(len(eligible)),
            "not_confirmed_independent_object_instances":True
        }
    summary={
        "dataset":REPO_ID,"revision":args.revision,
        "episode_rows":int(len(ep)),"unique_episode_ids":int(ep["episode_id"].nunique()),
        "duplicate_episode_id_rows":int(ep["episode_id"].duplicated().sum()),
        "unique_task_ids":int(ep["task_id"].nunique()),"unique_actor_ids":int(ep["actor_id"].nunique()),
        "unique_scenes":int(ep["scene"].nunique()),
        "episodes_missing_item_labels":int((ep["n_items"]==0).sum()),
        "episodes_with_multiple_item_labels":int((ep["n_items"]>1).sum()),
        "unique_individual_item_labels":int(len(item_df)),
        "unique_exact_item_bundles":int(len(bundle)),
        "raw_item_label_episode_count_quantiles":item_df["episode_count"].quantile([0,.25,.5,.75,1]).to_dict() if len(item_df) else {},
        "raw_item_label_naive_k_eligibility":eligibility,
        "episode_duration_s_quantiles":ep["duration_s"].quantile([0,.25,.5,.75,1]).to_dict(),
        "tactile_sampling_rate_rounded_hz_counts":ep["glove_rate_hz"].round(1).value_counts(dropna=False).to_dict(),
        "left_contact_fraction_quantiles":ep["tactile_contact_frac_left"].quantile([0,.25,.5,.75,1]).to_dict(),
        "right_contact_fraction_quantiles":ep["tactile_contact_frac_right"].quantile([0,.25,.5,.75,1]).to_dict(),
        "quality_score_quantiles":ep["quality_score"].quantile([0,.25,.5,.75,1]).to_dict(),
        "objects_mesh_registry_rows":int(len(objects)),"objects_mesh_registry_columns":list(objects.columns),
        "geometry":geom,
        "metadata_file_integrity":{k:{"bytes":v.stat().st_size,"sha256":sha256(v)} for k,v in files.items()},
        "warnings":[
            "Raw item labels are involved-object names from episodes, not unique physical object-instance IDs.",
            "The same item label can refer to different instances; item co-occurrence bundles may contain props, containers, tools and target objects.",
            "Naive K eligibility counts are only label-frequency screens, not validated few-shot object tasks.",
            "The 193-row mesh registry is partial; object_name does not always match episode items.",
            "Episode-level repeated observations may share participant, task, scene, date or recording context; audit before Support/Query splitting.",
            "No raw tactile shards or video files were downloaded."
        ]
    }
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    print(f"Wrote {out.resolve()}")

if __name__=="__main__": main()
