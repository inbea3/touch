#!/usr/bin/env python3
"""TouchScale early-to-late pad-structure persistence feasibility audit (no model fitting)."""
from __future__ import annotations
import io, json, os, shutil, tarfile, hashlib
from collections import defaultdict
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_touchscale_signal_300 import REPO, SHARD_CONFIG, choose_sample

OUT=Path("artifacts/touchscale_structure_prediction")
EPS=1e-8

def corr_vec(X):
    # X: time x 15 pads
    if X.shape[0] < 20:
        return None
    X=np.log1p(np.maximum(X,0))
    sd=X.std(axis=0)
    keep=sd>EPS
    if keep.sum()<4:
        return None
    C=np.corrcoef(X[:,keep],rowvar=False)
    iu=np.triu_indices(C.shape[0],1)
    v=C[iu]
    return v[np.isfinite(v)]

def safe_corr(a,b):
    if a is None or b is None or len(a)!=len(b) or len(a)<6:
        return None
    if np.std(a)<EPS or np.std(b)<EPS:
        return None
    return float(np.corrcoef(a,b)[0,1])

def cosine(a,b):
    if a is None or b is None or len(a)!=len(b):
        return None
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>EPS else None

def summarize_hand(g):
    pads=sorted([k for k in g if k.startswith("pad_")],key=lambda k:int(k.split("_")[1]))
    if len(pads)!=15:
        return {"status":"pad_count_not_15","pads":len(pads)}
    arrays=[np.asarray(g[k][()],dtype=np.float32) for k in pads]
    lengths=[a.shape[0] for a in arrays]
    if len(set(lengths))!=1:
        return {"status":"pad_length_mismatch"}
    T=lengths[0]
    # One scalar force trace per pad; preserves pad-level spatial organization.
    P=np.column_stack([np.maximum(a.reshape(T,-1),0).sum(axis=1) for a in arrays])
    if not np.isfinite(P).all():
        return {"status":"non_finite"}
    n=max(20,int(T*.30))
    early=P[:n]
    late=P[T-n:]
    # Use one shared pad mask across early and late windows. Do not independently
    # filter pads in each window: differing variable-pad counts are not a reason
    # to discard an otherwise valid recording.
    Z=np.log1p(P)
    common=(Z[:n].std(axis=0)>EPS)&(Z[T-n:].std(axis=0)>EPS)
    if common.sum()<4:
        return {"status":"insufficient_common_variable_pads","frames":T,"variable_pads":int(common.sum())}
    Ce=np.corrcoef(Z[:n,common],rowvar=False)
    Cl=np.corrcoef(Z[T-n:,common],rowvar=False)
    iu=np.triu_indices(int(common.sum()),1)
    a,b=Ce[iu],Cl[iu]
    ok=np.isfinite(a)&np.isfinite(b)
    a,b=a[ok],b[ok]
    structure_corr=safe_corr(a,b)
    structure_cos=cosine(a,b)
    # Persistence baseline: predict late matrix with early matrix.
    # Report absolute edge error and normalized error; no learned model involved.
    edge_mae=float(np.mean(np.abs(a-b))) if len(a) else None
    edge_rmse=float(np.sqrt(np.mean((a-b)**2))) if len(a) else None
    # Distribution shift of mean pad loading.
    me=early.mean(axis=0); ml=late.mean(axis=0)
    load_cos=cosine(me,ml)
    # Rank overlap of the three most loaded pads.
    k=min(3,len(me))
    top_e=set(np.argsort(me)[-k:]); top_l=set(np.argsort(ml)[-k:])
    top_jaccard=len(top_e&top_l)/len(top_e|top_l) if top_e|top_l else None
    return {
      "status":"ok","frames":int(T),"window_frames":int(n),
      "common_variable_pads":int(common.sum()),"valid_edges":int(len(a)),
      "early_late_structure_corr":structure_corr,
      "early_late_structure_cosine":structure_cos,
      "persistence_edge_mae":edge_mae,"persistence_edge_rmse":edge_rmse,
      "pad_load_cosine":load_cos,"top3_pad_jaccard":top_jaccard,
      "early_active_fraction":float((P[:n].sum(axis=1)>3).mean()),
      "late_active_fraction":float((P[T-n:].sum(axis=1)>3).mean())
    }

def main():
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/"hf_cache"
    shards=[x.strip() for x in SHARD_CONFIG.read_text().splitlines() if x.strip()]
    meta_path=hf_hub_download(repo_id=REPO,filename="metadata/episodes.parquet",repo_type="dataset",token=token,cache_dir=str(cache))
    meta=pd.read_parquet(meta_path)
    sample,scene_quotas=choose_sample(meta,shards)
    sample.to_csv(OUT/"sample_selection.csv",index=False)
    by_shard=defaultdict(list)
    for row in sample.to_dict("records"):
        by_shard[row["shard"]].append(row)
    details={}; flat=[]; hashes={}
    for shard,rows in sorted(by_shard.items()):
        local=hf_hub_download(repo_id=REPO,filename=shard,repo_type="dataset",token=token,cache_dir=str(cache))
        h=hashlib.sha256()
        with open(local,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
        hashes[shard]=h.hexdigest()
        with tarfile.open(local,"r:*") as tar:
            wanted={r["episode_id"]+".episode.h5" for r in rows}
            members={Path(m.name).name:m for m in tar if m.isfile() and Path(m.name).name in wanted}
            for row in rows:
                eid=str(row["episode_id"]); member=members.get(eid+".episode.h5")
                if member is None:
                    details[eid]={"status":"h5_missing","shard":shard}
                    continue
                with h5py.File(io.BytesIO(tar.extractfile(member).read()),"r") as h5:
                    hands={}
                    for hand in ("left","right"):
                        path=f"obs/tactile/raw/{hand}"
                        hands[hand]=summarize_hand(h5[path]) if path in h5 else {"status":"missing_hand"}
                details[eid]={"status":"ok","shard":shard,"scene":row["scene"],
                  "task_name":row["task_name"],"actor_id":row["actor_id"],
                  "collection_date":row["collection_date"],"hands":hands}
                for hand,stat in hands.items():
                    flat.append({"episode_id":eid,"scene":row["scene"],"task_name":row["task_name"],
                      "actor_id":row["actor_id"],"collection_date":row["collection_date"],
                      "hand":hand,**stat})
        shutil.rmtree(cache,ignore_errors=True)
    (OUT/"detail.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    pd.DataFrame(flat).to_csv(OUT/"hand_structure_summary.csv",index=False)
    (OUT/"shard_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    df=pd.DataFrame(flat)
    ok=df[df.status=="ok"]
    result={"selected_episodes":int(len(sample)),"unique_episodes":int(sample.episode_id.nunique()),
      "episodes_found":int(sum(v.get("status")=="ok" for v in details.values())),
      "hand_records":int(len(df)),"successful_hand_records":int(len(ok)),
      "scenes":int(sample.scene.nunique()),"actors":int(sample.actor_id.nunique()),
      "dates":int(sample.collection_date.nunique()),"shards":int(len(by_shard)),
      "scene_quotas":scene_quotas,
      "median_early_late_structure_corr":float(ok.early_late_structure_corr.median()) if len(ok) else None,
      "median_persistence_edge_mae":float(ok.persistence_edge_mae.median()) if len(ok) else None,
      "median_pad_load_cosine":float(ok.pad_load_cosine.median()) if len(ok) else None,
      "median_top3_pad_jaccard":float(ok.top3_pad_jaccard.median()) if len(ok) else None}
    (OUT/"run_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
