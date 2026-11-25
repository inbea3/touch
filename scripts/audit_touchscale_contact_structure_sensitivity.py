#!/usr/bin/env python3
"""Sensitivity audit for contact-conditioned pad structure across thresholds/windows."""
from __future__ import annotations
import io,json,os,shutil,tarfile,hashlib
from collections import defaultdict
from pathlib import Path
import h5py,numpy as np,pandas as pd
from huggingface_hub import hf_hub_download
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_touchscale_signal_300 import REPO,SHARD_CONFIG,choose_sample

OUT=Path("artifacts/touchscale_contact_structure_sensitivity")
THRESHOLDS=(1.0,3.0,5.0)
FRACTIONS=(0.20,0.30,0.40)
MIN_WINDOW=20
EPS=1e-8

def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.dot(a,b)/d) if d>EPS else None

def pearson(a,b):
 if len(a)<6 or np.std(a)<EPS or np.std(b)<EPS:return None
 return float(np.corrcoef(a,b)[0,1])

def metric(P,threshold,fraction):
 T=len(P); active=np.flatnonzero(P.sum(axis=1)>threshold); na=len(active)
 nw=max(MIN_WINDOW,int(np.floor(na*fraction)))
 if na<2*MIN_WINDOW or nw*2>na:
  return {"status":"insufficient_active_frames","active_frames":na,"window_frames":nw}
 ei,li=active[:nw],active[-nw:]
 if ei[-1]>=li[0]:
  return {"status":"overlapping_windows","active_frames":na,"window_frames":nw}
 Z=np.log1p(P); E,L=Z[ei],Z[li]
 mask=(E.std(axis=0)>EPS)&(L.std(axis=0)>EPS)
 if mask.sum()<4:
  return {"status":"insufficient_common_variable_pads","active_frames":na,"window_frames":nw,"common_pads":int(mask.sum())}
 E,L=E[:,mask],L[:,mask]
 CE,CL=np.corrcoef(E,rowvar=False),np.corrcoef(L,rowvar=False)
 iu=np.triu_indices(mask.sum(),1); a,b=CE[iu],CL[iu]
 valid=np.isfinite(a)&np.isfinite(b);a,b=a[valid],b[valid]
 if len(a)<6:return {"status":"undefined_edges","active_frames":na,"window_frames":nw}
 em, lm=E.mean(0),L.mean(0)
 te=set(np.argsort(em)[-3:]);tl=set(np.argsort(lm)[-3:])
 return {"status":"ok","active_frames":na,"window_frames":nw,"common_pads":int(mask.sum()),
  "structure_corr":pearson(a,b),"structure_cosine":cos(a,b),
  "edge_mae":float(np.mean(np.abs(a-b))),"edge_rmse":float(np.sqrt(np.mean((a-b)**2))),
  "pad_load_cosine":cos(em,lm),"top3_jaccard":len(te&tl)/len(te|tl),
  "active_fraction":float(na/T)}

def audit_hand(g):
 pads=sorted([k for k in g if k.startswith("pad_")],key=lambda k:int(k.split("_")[1]))
 if len(pads)!=15:return {"status":"pad_count_not_15"}
 arr=[np.asarray(g[k][()],dtype=np.float32) for k in pads]
 if len({a.shape[0] for a in arr})!=1:return {"status":"pad_length_mismatch"}
 T=arr[0].shape[0]
 P=np.column_stack([np.maximum(a.reshape(T,-1),0).sum(1) for a in arr])
 if not np.isfinite(P).all():return {"status":"non_finite"}
 out=[]
 for th in THRESHOLDS:
  for fr in FRACTIONS:
   out.append({"threshold_N":th,"window_fraction":fr,**metric(P,th,fr)})
 return out

def main():
 token=os.environ.get("HF_TOKEN")
 if not token:raise RuntimeError("HF_TOKEN secret missing")
 OUT.mkdir(parents=True,exist_ok=True);cache=OUT/"hf_cache"
 shards=[x.strip() for x in SHARD_CONFIG.read_text().splitlines() if x.strip()]
 mp=hf_hub_download(repo_id=REPO,filename="metadata/episodes.parquet",repo_type="dataset",token=token,cache_dir=str(cache))
 meta=pd.read_parquet(mp);shutil.rmtree(cache,ignore_errors=True)
 sample,quotas=choose_sample(meta,shards);sample.to_csv(OUT/"sample_selection.csv",index=False)
 by=defaultdict(list)
 for r in sample.to_dict("records"):by[r["shard"]].append(r)
 rows=[];details={};hashes={}
 for shard,rs in sorted(by.items()):
  path=hf_hub_download(repo_id=REPO,filename=shard,repo_type="dataset",token=token,cache_dir=str(cache))
  h=hashlib.sha256()
  with open(path,"rb") as f:
   for b in iter(lambda:f.read(1<<20),b""):h.update(b)
  hashes[shard]=h.hexdigest()
  with tarfile.open(path,"r:*") as tar:
   wanted={r["episode_id"]+".episode.h5" for r in rs}
   members={Path(m.name).name:m for m in tar if m.isfile() and Path(m.name).name in wanted}
   for r in rs:
    eid=str(r["episode_id"]);m=members.get(eid+".episode.h5")
    if m is None:
     details[eid]={"status":"h5_missing"};continue
    with h5py.File(io.BytesIO(tar.extractfile(m).read()),"r") as f:
     for hand in ("left","right"):
      p=f"obs/tactile/raw/{hand}"
      if p not in f:
       rows.append({"episode_id":eid,"scene":r["scene"],"actor_id":r["actor_id"],"hand":hand,"threshold_N":None,"window_fraction":None,"status":"missing_hand"});continue
      stats=audit_hand(f[p])
      if isinstance(stats,dict):
       rows.append({"episode_id":eid,"scene":r["scene"],"actor_id":r["actor_id"],"hand":hand,**stats})
      else:
       for st in stats:
        rows.append({"episode_id":eid,"scene":r["scene"],"actor_id":r["actor_id"],"hand":hand,**st})
    details[eid]={"status":"ok","shard":shard}
  shutil.rmtree(cache,ignore_errors=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/"sensitivity_hand_summary.csv",index=False)
 (OUT/"detail.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
 (OUT/"shard_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
 result={"episodes_selected":len(sample),"episodes_found":sum(v.get("status")=="ok" for v in details.values()),
  "hand_condition_rows":len(df),"actors":int(sample.actor_id.nunique()),"scenes":int(sample.scene.nunique()),
  "thresholds_N":THRESHOLDS,"window_fractions":FRACTIONS,"minimum_frames_per_window":MIN_WINDOW,
  "status_counts":{str(k):int(v) for k,v in df.status.value_counts().items()}}
 (OUT/"run_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
