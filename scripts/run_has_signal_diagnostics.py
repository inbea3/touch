#!/usr/bin/env python3
"""Stage 21: diagnose signal, object effects, target degeneracy and baseline predictability."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mae(a,b):return float(np.mean(np.abs(a-b)))
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def corr_safe(a,b):
 if np.std(a)<1e-12 or np.std(b)<1e-12:return None
 return float(np.corrcoef(a,b)[0,1])
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_feature_dataset/episode_features_targets.npz")
 ap.add_argument("--index",default="results/has_feature_dataset/episode_index.csv")
 ap.add_argument("--out",default="results/has_signal_diagnostics")
 ap.add_argument("--alphas",default="0.1,1,10,100,1000")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(s):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {s}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 with np.load(a.data) as z:
  X=z["X_early"].astype(np.float64);Ym=z["Y_future_mean"].astype(np.float64);Ysd=z["Y_future_std"].astype(np.float64)
  R=z["Y_future_corr_upper"].astype(np.float64);ci=z["corr_i"];cj=z["corr_j"]
 rows=list(csv.DictReader(open(a.index,encoding="utf-8")))
 if len(rows)!=4192 or len(rows)!=len(X):raise RuntimeError("Feature/index mismatch")
 split=np.array([r["grouped_split"] for r in rows]);obj=np.array([r["object_group"] for r in rows]);label=np.array([r["label_name"] for r in rows])
 if len(set(obj[split=="train"])&set(obj[split=="validation"])) or len(set(obj[split=="train"])&set(obj[split=="test"])) or len(set(obj[split=="validation"])&set(obj[split=="test"])):raise RuntimeError("Object split leakage")
 ch=np.array([i for i in range(44) if i not in (18,25)])
 pairs=np.where(np.isin(ci,ch)&np.isin(cj,ch))[0]
 # Core targets
 targets={"future_mean_42":Ym[:,ch],"future_sd_42":Ysd[:,ch],"future_corr_861":R[:,pairs]}
 split_summary={}; channel_rows=[];object_rows=[];label_rows=[]
 for sp in ("train","validation","test"):
  ix=np.where(split==sp)[0]; split_summary[sp]={"episodes":len(ix),"objects":len(set(obj[ix]))}
  for name,Y in targets.items():
   yy=Y[ix]
   split_summary[sp][name]={"global_mean":float(yy.mean()),"global_std":float(yy.std()),
    "mean_per_target_std":float(yy.std(axis=0).mean()),"constant_target_fraction":float(np.mean(yy.std(axis=0)<1e-10))}
  # per-object target mean and within-object variability
  for ob in sorted(set(obj[ix])):
   oi=ix[obj[ix]==ob]
   rr={"split":sp,"object":ob,"episodes":len(oi)}
   for name,Y in targets.items():
    rr[name+"_mean"]=float(Y[oi].mean());rr[name+"_std"]=float(Y[oi].std())
   object_rows.append(rr)
  # per label summaries; label names are retained as metadata, not model inputs
  for lb in sorted(set(label[ix])):
   li=ix[label[ix]==lb]
   rr={"split":sp,"label":lb,"episodes":len(li),"objects":len(set(obj[li]))}
   for name,Y in targets.items():rr[name+"_mean"]=float(Y[li].mean())
   label_rows.append(rr)
 # Object ICC-like descriptive variance decomposition by target coordinate, computed separately per split.
 icc={}
 for sp in ("train","validation","test"):
  ix=np.where(split==sp)[0]; groups=sorted(set(obj[ix]))
  for name,Yall in targets.items():
   Y=Yall[ix]; overall=Y.mean(0); between=np.zeros(Y.shape[1]);within=np.zeros(Y.shape[1])
   for ob in groups:
    g=Y[obj[ix]==ob];between+=len(g)*(g.mean(0)-overall)**2;within+=((g-g.mean(0))**2).sum(0)
   denom=between+within
   icc.setdefault(sp,{})[name]={"between_object_variance_fraction":float(np.mean(np.divide(between,denom,out=np.zeros_like(between),where=denom>0))),
      "median_coordinate_fraction":float(np.median(np.divide(between,denom,out=np.zeros_like(between),where=denom>0)))}
 # Channel diagnostics: early-to-future association and future degeneracy.
 for sp in ("train","validation","test"):
  ix=np.where(split==sp)[0]
  for k,c in enumerate(ch):
   channel_rows.append({"split":sp,"channel":int(c),"future_mean":float(Ym[ix,c].mean()),
    "future_mean_std":float(Ym[ix,c].std()),"future_sd_mean":float(Ysd[ix,c].mean()),
    "future_sd_zero_fraction":float(np.mean(Ysd[ix,c]<1e-10)),
    "early_last_future_mean_corr":corr_safe(X[ix,132+c],Ym[ix,c])})
 # Validation-only predictive signal test. Fit feature scaler and target scaler on Train; select alpha on Validation.
 tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
 xcols=np.concatenate([ch,44+ch,88+ch,132+ch]);X42=X[:,xcols]
 xm=X42[tr].mean(0);xs=X42[tr].std(0);xs[xs<1e-8]=1;Z=(X42-xm)/xs
 prediction={}
 for name,Y in (("future_mean",Ym[:,ch]),("future_corr",R[:,pairs])):
  ym=Y[tr].mean(0);ys=Y[tr].std(0);ys[ys<1e-8]=1;T=(Y-ym)/ys
  base=np.repeat(ym[None,:],len(va),axis=0)
  candidates=[]
  for alpha in [float(v) for v in a.alphas.split(",")]:
   model=Ridge(alpha=alpha,solver="lsqr");model.fit(Z[tr],T[tr]);pv=model.predict(Z[va])
   candidates.append((float(np.mean((pv-T[va])**2)),alpha,model))
  valmse,alpha,model=min(candidates,key=lambda q:q[0])
  pred=model.predict(Z[va])*ys+ym
  prediction[name]={"selected_alpha_validation":alpha,"validation_standardized_mse":valmse,
   "validation_mae_ridge":mae(Y[va],pred),"validation_mae_trainmean":mae(Y[va],base),
   "validation_rmse_ridge":rmse(Y[va],pred),"validation_rmse_trainmean":rmse(Y[va],base),
   "validation_r2_uniform_average":float(r2_score(Y[va],pred,multioutput="uniform_average"))}
  # Exploratory locked-model test report; alpha is not reselected on Test.
  ptest=model.predict(Z[te])*ys+ym;bt=np.repeat(ym[None,:],len(te),axis=0)
  prediction[name]["test_exploratory_fixed_alpha"]={"mae_ridge":mae(Y[te],ptest),"mae_trainmean":mae(Y[te],bt),
   "rmse_ridge":rmse(Y[te],ptest),"rmse_trainmean":rmse(Y[te],bt)}
  log(f"{name}: validation alpha={alpha}, ridge MAE={prediction[name]['validation_mae_ridge']:.6f}, mean MAE={prediction[name]['validation_mae_trainmean']:.6f}")
 # Feature-target association summary; report absolute linear correlations only on train/validation.
 assoc={}
 for sp in ("train","validation"):
  ix=np.where(split==sp)[0]
  vals=[]
  for j in range(Ym.shape[1]):
   vals.append(corr_safe(X[ix,132+j],Ym[ix,j]))
  finite=[abs(v) for v in vals if v is not None]
  assoc[sp]={"matched_channel_last_to_future_mean_abs_corr_mean":float(np.mean(finite)) if finite else None,
   "matched_channel_last_to_future_mean_abs_corr_median":float(np.median(finite)) if finite else None}
 summary={"n_episodes":len(rows),"split_counts":{s:int(np.sum(split==s)) for s in ("train","validation","test")},
  "split_summary":split_summary,"object_variance_decomposition":icc,"validation_selected_prediction":prediction,
  "matched_channel_association":assoc,"interpretation_limits":["Object-level variance fraction is descriptive, not causal.","Test results are exploratory because earlier model development already inspected this fixed test set.","Label summaries are descriptive only; label is not used as model input.","No GNN or meta-learning model is trained in this stage."],
  "elapsed_seconds":time.time()-t0}
 (out/"split_summary.json").write_text(json.dumps(summary,indent=2))
 for name,data in [("object_target_summary.csv",object_rows),("label_target_summary.csv",label_rows),("channel_signal_summary.csv",channel_rows)]:
  with (out/name).open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (out/"config.json").write_text(json.dumps({"input":"Stage19 pinned feature artifact","channels_primary":ch.tolist(),"target_definitions":list(targets),"ridge_alpha_grid":[float(v) for v in a.alphas.split(",")],"scalers":"Train only","alpha_selection":"Validation only","test_usage":"exploratory fixed-alpha report only"},indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n")
 log(f"COMPLETE elapsed={summary['elapsed_seconds']:.1f}s")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
