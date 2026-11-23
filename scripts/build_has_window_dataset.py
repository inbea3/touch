#!/usr/bin/env python3
"""Stage 22a: extract matched 5/10/20-frame pressure windows and subsequent future means."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
DATASET="TUM-ICS/Hide-and-Seek"; SPLITS=("train","validation","test"); C=44; WINDOWS=(5,10,20); DROP=(18,25)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def feat(x,t):
 tc=t-t.mean(); den=float(tc@tc)
 slope=((tc[:,None]*(x-x.mean(0))).sum(0)/den) if den>0 else np.zeros(C)
 return np.concatenate([x.mean(0),x.std(0),slope,x[-1]]).astype(np.float32)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--revision",default="4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf")
 p.add_argument("--split-manifest",default="results/has_object_grouped_split/episode_split_manifest.csv")
 p.add_argument("--object-manifest",default="results/has_object_grouped_split/object_split_manifest.csv")
 p.add_argument("--out",default="results/has_window_dataset")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(s):
  z=f"[{datetime.now(timezone.utc).isoformat()}] {s}";print(z,flush=True)
  with (out/"run.log").open("a") as f:f.write(z+"\n")
 manifests={}
 with open(a.split_manifest,newline="") as f:
  for r in csv.DictReader(f):manifests[(r["official_split"],int(r["episode_index"]))]=r
 objrows=list(csv.DictReader(open(a.object_manifest,newline="")))
 objsets={s:set(r["object_group"] for r in objrows if r["grouped_split"]==s) for s in ("train","validation","test")}
 if any(objsets[x]&objsets[y] for x,y in (("train","validation"),("train","test"),("validation","test"))):raise RuntimeError("object leakage")
 if len(manifests)!=4192:raise RuntimeError(f"manifest count {len(manifests)}")
 data={w:{"X":[],"Y":[],"index":[]} for w in WINDOWS}; total=0; seen=set()
 log(f"START dataset={DATASET} revision={a.revision} episodes={len(manifests)}")
 for sp in SPLITS:
  ds=load_dataset(DATASET,split=sp,streaming=True,revision=a.revision); active=None; st=None; closed=set()
  def finish(s):
   key=s["key"]; n=s["n"]; rec=s["meta"]
   if key not in seen:seen.add(key)
   for w in WINDOWS:
    nf=s["n"]-w
    if n<w+5:continue
    xx=np.asarray(s["early"][:w]); tt=np.asarray(s["times"][:w])
    mu=s["sum"][w]/nf
    # The target is the mean over all frames strictly after the observed window.
    data[w]["X"].append(feat(xx,tt));data[w]["Y"].append(mu.astype(np.float32))
    data[w]["index"].append({"array_row":len(data[w]["index"]),"window_frames":w,
      "grouped_split":rec["grouped_split"],"object_group":rec["object_group"],
      "official_split":key[0],"episode_index":key[1],"label_name":rec["label_name"],
      "n_frames":n,"future_frames":nf,"window_duration_seconds":float(tt[-1]-tt[0])})
  for ri,row in enumerate(ds):
   key=(sp,int(row["episode_index"]))
   if key not in manifests:continue
   if key!=active:
    if st is not None:finish(st);closed.add(active)
    if key in closed:raise RuntimeError(f"non-contiguous episode {key}")
    active=key;st={"key":key,"meta":manifests[key],"n":0,"early":[],"times":[],"last_t":None,
      "sum":{w:np.zeros(C) for w in WINDOWS}}
   x=np.asarray(row["obs_skin_force_left"],dtype=np.float64);ts=float(row["timestamp"])
   if x.shape!=(C,) or not np.isfinite(x).all():raise RuntimeError(f"invalid sensor row {key}")
   if st["last_t"] is not None and ts<=st["last_t"]:raise RuntimeError(f"timestamp order {key}")
   st["n"]+=1;total+=1;st["last_t"]=ts
   if st["n"]<=max(WINDOWS):st["early"].append(x.copy());st["times"].append(ts)
   for w in WINDOWS:
    if st["n"]>w:st["sum"][w]+=x
   if ri and ri%100000==0:log(f"PROGRESS split={sp} rows={ri} frames={total}")
  if st is not None:finish(st)
  log(f"FINISHED split={sp} frames={total}")
 if seen!=set(manifests):raise RuntimeError(f"manifest episodes unseen: {len(set(manifests)-seen)}")
 summaries={}
 keep=np.array([i for i in range(C) if i not in DROP]); cols=np.concatenate([keep,44+keep,88+keep,132+keep])
 for w in WINDOWS:
  X=np.stack(data[w]["X"]);Y=np.stack(data[w]["Y"]);idx=data[w]["index"]
  X=X[:,cols];Y=Y[:,keep]
  np.savez_compressed(out/f"window_{w}.npz",X=X,Y_future_mean=Y)
  with (out/f"window_{w}_index.csv").open("w",newline="") as f:
   wr=csv.DictWriter(f,fieldnames=list(idx[0]));wr.writeheader();wr.writerows(idx)
  summaries[str(w)]={"episodes":len(idx),"X_shape":list(X.shape),"Y_shape":list(Y.shape),
   "episodes_by_split":{s:sum(r["grouped_split"]==s for r in idx) for s in ("train","validation","test")},
   "objects_by_split":{s:len(set(r["object_group"] for r in idx if r["grouped_split"]==s)) for s in ("train","validation","test")}}
 config={"dataset":DATASET,"revision":a.revision,"split_seed":20261123,"windows":list(WINDOWS),
  "feature":"mean,population std,timestamp slope,last over first w frames; concatenated blocks then exclude channels 18,25",
  "target":"mean pressure over every frame strictly after first w frames; exclude channels 18,25",
  "input_dim":168,"target_dim":42,"min_future_frames":5,"raw_sequences_persisted":False}
 (out/"config.json").write_text(json.dumps(config,indent=2))
 (out/"summary.json").write_text(json.dumps({"frames_processed":total,"windows":summaries,"elapsed_seconds":time.time()-t0},indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n")
 log("DONE")
 files=sorted(x for x in out.iterdir() if x.is_file() and x.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(x)}  {x.name}" for x in files)+"\n")
 print(json.dumps(summaries,indent=2))
if __name__=="__main__":main()
