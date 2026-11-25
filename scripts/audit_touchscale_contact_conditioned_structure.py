#!/usr/bin/env python3
"""Contact-conditioned early-to-late tactile pad-structure audit."""
from __future__ import annotations
import io, json, os, shutil, tarfile, hashlib, sys
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_touchscale_signal_300 import REPO, SHARD_CONFIG, choose_sample

OUT=Path("artifacts/touchscale_contact_conditioned_structure")
EPS=1e-8
CONTACT_THRESHOLD_N=3.0
WINDOW_FRACTION=0.30
MIN_ACTIVE_FRAMES_PER_WINDOW=20

def cosine(a,b):
    if a is None or b is None or len(a)!=len(b): return None
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den>EPS else None

def corr(a,b):
    if a is None or b is None or len(a)!=len(b) or len(a)<6: return None
    if np.std(a)<EPS or np.std(b)<EPS: return None
    return float(np.corrcoef(a,b)[0,1])

def structure_edges(Z, pad_mask):
    X=Z[:,pad_mask]
    if X.shape[1]<4 or X.shape[0]<MIN_ACTIVE_FRAMES_PER_WINDOW:
        return None
    C=np.corrcoef(X,rowvar=False)
    iu=np.triu_indices(X.shape[1],1)
    e=C[iu]
    return e if np.isfinite(e).all() else None

def audit_hand(g):
    pads=sorted([k for k in g if k.startswith("pad_")],key=lambda k:int(k.split("_")[1]))
    if len(pads)!=15:
        return {"status":"pad_count_not_15","pads":len(pads)}
    arrays=[np.asarray(g[k][()],dtype=np.float32) for k in pads]
    lengths=[a.shape[0] for a in arrays]
    if len(set(lengths))!=1:
        return {"status":"pad_length_mismatch"}
    T=lengths[0]
    # One nonnegative normal-force sum per native pad and frame.
    P=np.column_stack([np.maximum(a.reshape(T,-1),0).sum(axis=1) for a in arrays])
    if not np.isfinite(P).all():
        return {"status":"non_finite"}
    total=P.sum(axis=1)
    active=np.flatnonzero(total>CONTACT_THRESHOLD_N)
    n_active=len(active)
    n_window=max(MIN_ACTIVE_FRAMES_PER_WINDOW,int(np.floor(n_active*WINDOW_FRACTION)))
    if n_active < 2*MIN_ACTIVE_FRAMES_PER_WINDOW:
        return {"status":"insufficient_active_frames","frames":int(T),
                "active_frames":int(n_active),"active_fraction":float(n_active/T),
                "required_active_frames":2*MIN_ACTIVE_FRAMES_PER_WINDOW}
    early_idx=active[:n_window]
    late_idx=active[-n_window:]
    if early_idx[-1]>=late_idx[0]:
        return {"status":"overlapping_contact_windows","frames":int(T),"active_frames":int(n_active),
                "window_frames":int(n_window)}
    Z=np.log1p(P)
    early=Z[early_idx]; late=Z[late_idx]
    pad_mask=(early.std(axis=0)>EPS)&(late.std(axis=0)>EPS)
    if pad_mask.sum()<4:
        return {"status":"insufficient_common_variable_pads","frames":int(T),
                "active_frames":int(n_active),"window_frames":int(n_window),
                "common_variable_pads":int(pad_mask.sum())}
    e=structure_edges(early,pad_mask); l=structure_edges(late,pad_mask)
    if e is None or l is None or len(e)!=len(l):
        return {"status":"undefined_structure","frames":int(T),"active_frames":int(n_active),
                "window_frames":int(n_window),"common_variable_pads":int(pad_mask.sum())}
    early_load=early.mean(axis=0); late_load=late.mean(axis=0)
    k=3
    te=set(np.argsort(early_load)[-k:]); tl=set(np.argsort(late_load)[-k:])
    top3=len(te&tl)/len(te|tl)
    return {
      "status":"ok","frames":int(T),"active_frames":int(n_active),
      "active_fraction":float(n_active/T),"window_frames":int(n_window),
      "early_first_frame":int(early_idx[0]),"early_last_frame":int(early_idx[-1]),
      "late_first_frame":int(late_idx[0]),"late_last_frame":int(late_idx[-1]),
      "gap_between_windows_frames":int(late_idx[0]-early_idx[-1]-1),
      "common_variable_pads":int(pad_mask.sum()),"valid_edges":int(len(e)),
      "contact_early_late_structure_corr":corr(e,l),
      "contact_early_late_structure_cosine":cosine(e,l),
      "contact_persistence_edge_mae":float(np.mean(np.abs(e-l))),
      "contact_persistence_edge_rmse":float(np.sqrt(np.mean((e-l)**2))),
      "contact_pad_load_cosine":cosine(early_load,late_load),
      "contact_top3_pad_jaccard":float(top3)
    }

def main():
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/"hf_cache"
    shards=[x.strip() for x in SHARD_CONFIG.read_text().splitlines() if x.strip()]
    meta_path=hf_hub_download(repo_id=REPO,filename="metadata/episodes.parquet",
        repo_type="dataset",token=token,cache_dir=str(cache))
    meta=pd.read_parquet(meta_path)
    shutil.rmtree(cache,ignore_errors=True)
    sample,scene_quotas=choose_sample(meta,shards)
    sample.to_csv(OUT/"sample_selection.csv",index=False)
    by_shard=defaultdict(list)
    for row in sample.to_dict("records"): by_shard[row["shard"]].append(row)
    details={}; flat=[]; hashes={}
    for shard,rows in sorted(by_shard.items()):
        local=hf_hub_download(repo_id=REPO,filename=shard,repo_type="dataset",
            token=token,cache_dir=str(cache))
        h=hashlib.sha256()
        with open(local,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
        hashes[shard]=h.hexdigest()
        with tarfile.open(local,"r:*") as tar:
            wanted={r["episode_id"]+".episode.h5" for r in rows}
            members={Path(m.name).name:m for m in tar
                     if m.isfile() and Path(m.name).name in wanted}
            for row in rows:
                eid=str(row["episode_id"]); member=members.get(eid+".episode.h5")
                if member is None:
                    details[eid]={"status":"h5_missing","shard":shard}
                    continue
                with h5py.File(io.BytesIO(tar.extractfile(member).read()),"r") as h5:
                    hands={}
                    for hand in ("left","right"):
                        path=f"obs/tactile/raw/{hand}"
                        hands[hand]=audit_hand(h5[path]) if path in h5 else {"status":"missing_hand"}
                details[eid]={"status":"ok","shard":shard,"scene":row["scene"],
                    "task_name":row["task_name"],"actor_id":row["actor_id"],
                    "collection_date":row["collection_date"],"hands":hands}
                for hand,stat in hands.items():
                    flat.append({"episode_id":eid,"scene":row["scene"],"task_name":row["task_name"],
                        "actor_id":row["actor_id"],"collection_date":row["collection_date"],
                        "hand":hand,**stat})
        shutil.rmtree(cache,ignore_errors=True)
    (OUT/"detail.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    df=pd.DataFrame(flat)
    df.to_csv(OUT/"hand_contact_structure_summary.csv",index=False)
    (OUT/"shard_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    ok=df[df.status=="ok"]
    result={
      "selected_episodes":int(len(sample)),"unique_episodes":int(sample.episode_id.nunique()),
      "episodes_found":int(sum(v.get("status")=="ok" for v in details.values())),
      "hand_records":int(len(df)),"successful_hand_records":int(len(ok)),
      "status_counts":{str(k):int(v) for k,v in df.status.value_counts().items()},
      "scenes":int(sample.scene.nunique()),"actors":int(sample.actor_id.nunique()),
      "dates":int(sample.collection_date.nunique()),"shards":int(len(by_shard)),
      "scene_quotas":scene_quotas,
      "median_contact_early_late_structure_corr":float(ok.contact_early_late_structure_corr.median()) if len(ok) else None,
      "median_contact_persistence_edge_mae":float(ok.contact_persistence_edge_mae.median()) if len(ok) else None,
      "median_contact_pad_load_cosine":float(ok.contact_pad_load_cosine.median()) if len(ok) else None,
      "median_contact_top3_pad_jaccard":float(ok.contact_top3_pad_jaccard.median()) if len(ok) else None,
      "window_rule":{"threshold_N":CONTACT_THRESHOLD_N,"fraction_of_active_frames":WINDOW_FRACTION,
          "minimum_active_frames_per_window":MIN_ACTIVE_FRAMES_PER_WINDOW,
          "windows_nonoverlapping":True}
    }
    (OUT/"run_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
