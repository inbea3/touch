#!/usr/bin/env python3
"""Stage 22B: matched 5/10/20-window persistence and residual Ridge benchmark."""
import argparse,csv,hashlib,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def metrics(y,p):
 e=p-y;sp=[]
 for a,b in zip(y,p):
  if np.std(a)>1e-12 and np.std(b)>1e-12:
   q=spearmanr(a,b).statistic
   if np.isfinite(q):sp.append(float(q))
 return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),
 "episode_spearman":float(np.mean(sp)) if sp else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",default="results/has_window_dataset");ap.add_argument("--out",default="results/has_window_benchmark")
 ap.add_argument("--alphas",default="0.1,1,10,100,1000");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);root=Path(a.data);t0=time.time()
 def log(s):
  z=f"[{datetime.now(timezone.utc).isoformat()}] {s}";print(z,flush=True)
  with (out/"run.log").open("a") as f:f.write(z+"\n")
 allres={}; common=None;paired={}
 for w in (5,10,20):
  d=np.load(root/f"window_{w}.npz");X=d["X"].astype(float);Y=d["Y_future_mean"].astype(float)
  rows=list(csv.DictReader(open(root/f"window_{w}_index.csv",encoding="utf-8")))
  if len(rows)!=len(X):raise RuntimeError("index mismatch")
  split=np.array([r["grouped_split"] for r in rows]);obj=np.array([r["object_group"] for r in rows])
  ids=np.array([r["official_split"]+":"+r["episode_index"] for r in rows])
  tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
  if len(set(obj[tr])&set(obj[va])) or len(set(obj[tr])&set(obj[te])) or len(set(obj[va])&set(obj[te])):raise RuntimeError("object leakage")
  last=X[:,126:168] # feature block layout: 42 mean, 42 sd, 42 slope, 42 last
  persistence=last
  trainmean=np.repeat(Y[tr].mean(0,keepdims=True),len(Y),axis=0)
  xm=X[tr].mean(0);xs=X[tr].std(0);xs[xs<1e-8]=1;Z=(X-xm)/xs
  residual=Y-last;ym=residual[tr].mean(0);ys=residual[tr].std(0);ys[ys<1e-8]=1;T=(residual-ym)/ys
  candidates=[]
  for alpha in [float(v) for v in a.alphas.split(",")]:
   m=Ridge(alpha=alpha,solver="lsqr");m.fit(Z[tr],T[tr]);pv=m.predict(Z[va])
   candidates.append((float(np.mean((pv-T[va])**2)),alpha,m))
  valmse,alpha,model=min(candidates,key=lambda q:q[0])
  pred=model.predict(Z)*ys+ym+last
  res={"eligible_episodes":len(rows),"split_counts":{s:int(np.sum(split==s)) for s in ("train","validation","test")},
   "selected_alpha":alpha,"validation_standardized_mse":valmse,
   "validation":{"persistence":metrics(Y[va],persistence[va]),"train_mean":metrics(Y[va],trainmean[va]),"residual_ridge":metrics(Y[va],pred[va])},
   "test_exploratory":{"persistence":metrics(Y[te],persistence[te]),"train_mean":metrics(Y[te],trainmean[te]),"residual_ridge":metrics(Y[te],pred[te])}}
  objmetrics=[]
  for ob in sorted(set(obj[te])):
   ix=te[obj[te]==ob]
   objmetrics.append({"object":ob,"n":len(ix),"persistence":metrics(Y[ix],persistence[ix]),"residual_ridge":metrics(Y[ix],pred[ix])})
  res["test_exploratory"]["per_object"]=objmetrics
  allres[str(w)]=res
  np.savez_compressed(out/f"window_{w}_test_predictions.npz",y_true=Y[te],persistence=persistence[te],ridge=pred[te],episode_id=ids[te])
  with (out/f"window_{w}_test_index.csv").open("w",newline="") as f:
   wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows([rows[i] for i in te])
  np.savez_compressed(out/f"window_{w}_ridge_model.npz",coef=model.coef_,intercept=model.intercept_,x_mean=xm,x_std=xs,residual_mean=ym,residual_std=ys)
  log(f"window={w} alpha={alpha} val_Ridge_MAE={res['validation']['residual_ridge']['mae']:.6f} val_persist_MAE={res['validation']['persistence']['mae']:.6f}")
  common=ids if common is None else np.intersect1d(common,ids)
  paired[str(w)]={"ids":ids,"split":split,"Y":Y,"pred":pred,"last":last}
 # Paired common-episode validation comparison across windows
 common_results={}
 for w in (5,10,20):
  q=paired[str(w)]; idx={v:i for i,v in enumerate(q["ids"])}
  valids=[v for v in common if v in idx and q["split"][idx[v]]=="validation"]
  ii=np.array([idx[v] for v in valids])
  common_results[str(w)]={"n_common_validation":len(ii),"persistence":metrics(q["Y"][ii],q["last"][ii]),
   "residual_ridge":metrics(q["Y"][ii],q["pred"][ii])}
 result={"windows":allres,"paired_common_validation":common_results,
 "interpretation":"Validation selects Ridge alpha; Test is exploratory only because earlier stages inspected these objects.",
 "elapsed_seconds":time.time()-t0}
 (out/"metrics.json").write_text(json.dumps(result,indent=2))
 (out/"config.json").write_text(json.dumps({"windows":[5,10,20],"models":["persistence","train_mean","residual_ridge"],"alpha_grid":[float(v) for v in a.alphas.split(",")],"normalization":"Train only","selection":"Validation only","test_usage":"exploratory fixed model only","input_dim":168,"target_dim":42},indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n")
 log("COMPLETE")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
