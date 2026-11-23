#!/usr/bin/env python3
"""Stage 25: channel/object residual structure diagnostics; no neural model training."""
import argparse,csv,hashlib,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def stats(y,p):
 e=p-y
 return {"bias":np.mean(e,axis=0),"mae":np.mean(np.abs(e),axis=0),
         "rmse":np.sqrt(np.mean(e**2,axis=0)),"residual_sd":np.std(e,axis=0)}
def corr_summary(e):
 c=np.corrcoef(e,rowvar=False)
 c=np.nan_to_num(c,nan=0.0,posinf=0.0,neginf=0.0)
 np.fill_diagonal(c,0)
 vals=np.linalg.eigvalsh(c)
 off=c[np.triu_indices_from(c,1)]
 pairs=[]
 for i,j in zip(*np.triu_indices_from(c,1)):
  pairs.append((abs(float(c[i,j])),int(i),int(j),float(c[i,j])))
 pairs.sort(reverse=True)
 return c,{"mean_abs_offdiag_corr":float(np.mean(np.abs(off))),
 "median_abs_offdiag_corr":float(np.median(np.abs(off))),
 "max_abs_offdiag_corr":float(np.max(np.abs(off))),
 "eigenvalues_desc":vals[::-1].tolist(),
 "top_20_abs_pairs":[{"channel_i_retained":i,"channel_j_retained":j,"correlation":v} for _,i,j,v in pairs[:20]]}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_window_dataset")
 ap.add_argument("--out",default="results/has_stage25_residual_structure")
 ap.add_argument("--ridge-alpha",type=float,default=1000.0)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(s):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {s}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 root=Path(a.data)
 with np.load(root/"window_5.npz") as z:
  X=z["X"].astype(np.float64);Y=z["Y_future_mean"].astype(np.float64)
 rows=list(csv.DictReader(open(root/"window_5_index.csv",encoding="utf-8")))
 if X.shape!=(4192,168) or Y.shape!=(4192,42) or len(rows)!=4192:raise RuntimeError("Unexpected Stage22A shapes")
 split=np.array([r["grouped_split"] for r in rows]);obj=np.array([r["object_group"] for r in rows])
 tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
 groups={s:sorted(set(obj[split==s])) for s in ("train","validation","test")}
 if set(groups["train"])&set(groups["validation"]) or set(groups["train"])&set(groups["test"]) or set(groups["validation"])&set(groups["test"]):raise RuntimeError("Object leakage")
 # Feature schema: [mean(42), sd(42), slope(42), last(42)] after dropping source channels 18 and 25.
 P=X[:,126:168];D=Y-P
 xm=X[tr].mean(0);xs=X[tr].std(0);xs[xs<1e-8]=1
 dm=D[tr].mean(0);ds=D[tr].std(0);ds[ds<1e-8]=1
 Z=(X-xm)/xs;T=(D-dm)/ds
 model=Ridge(alpha=a.ridge_alpha).fit(Z[tr],T[tr])
 PR=P+model.predict(Z)*ds+dm
 config={"stage":"25","source_artifact_run":45193420854,"window":5,"episodes":4192,
 "input_dim":168,"target_dim":42,"retained_source_channel_indices":[i for i in range(44) if i not in (18,25)],
 "feature_blocks":["mean","population_sd","timestamp_slope","last"],"ridge_alpha":a.ridge_alpha,
 "ridge_alpha_origin":"fixed from Stage24B Validation selection; no retuning in Stage25",
 "scalers":"Train only","primary_split":"Validation","test":"exploratory only"}
 (out/"config.json").write_text(json.dumps(config,indent=2))
 channel_rows=[];object_rows=[];summary={}
 for sp,ix in (("validation",va),("test",te)):
  base=stats(Y[ix],P[ix]);ridge=stats(Y[ix],PR[ix])
  resid_base=P[ix]-Y[ix];resid_ridge=PR[ix]-Y[ix]
  for c,src in enumerate(config["retained_source_channel_indices"]):
   channel_rows.append({"split":sp,"retained_channel":c,"source_channel_index":src,
    "persistence_bias":float(base["bias"][c]),"persistence_mae":float(base["mae"][c]),
    "persistence_rmse":float(base["rmse"][c]),"persistence_residual_sd":float(base["residual_sd"][c]),
    "ridge_bias":float(ridge["bias"][c]),"ridge_mae":float(ridge["mae"][c]),
    "ridge_rmse":float(ridge["rmse"][c]),"ridge_residual_sd":float(ridge["residual_sd"][c]),
    "ridge_minus_persistence_mae":float(ridge["mae"][c]-base["mae"][c])})
  for ob in sorted(set(obj[ix])):
   jj=ix[obj[ix]==ob]
   bs=stats(Y[jj],P[jj]);rs=stats(Y[jj],PR[jj])
   object_rows.append({"split":sp,"object":ob,"n_episodes":len(jj),
    "persistence_mae":float(np.mean(bs["mae"])),"ridge_mae":float(np.mean(rs["mae"])),
    "persistence_rmse":float(np.mean(bs["rmse"])),"ridge_rmse":float(np.mean(rs["rmse"])),
    "persistence_bias_mean":float(np.mean(bs["bias"])),"ridge_bias_mean":float(np.mean(rs["bias"]))})
  cb,sb=corr_summary(resid_base);cr,sr=corr_summary(resid_ridge)
  np.save(out/f"{sp}_persistence_residual_corr.npy",cb);np.save(out/f"{sp}_ridge_residual_corr.npy",cr)
  summary[sp]={"n_episodes":len(ix),"persistence_mae":float(np.mean(base["mae"])),
   "ridge_mae":float(np.mean(ridge["mae"])),"persistence_rmse":float(np.mean(base["rmse"])),
   "ridge_rmse":float(np.mean(ridge["rmse"])),"persistence_residual_corr":sb,"ridge_residual_corr":sr,
   "channel_improved_by_ridge":int(np.sum(ridge["mae"]<base["mae"])),
   "channel_worsened_by_ridge":int(np.sum(ridge["mae"]>base["mae"]))}
  log(f"{sp}: persistence MAE={summary[sp]['persistence_mae']:.6f}; ridge MAE={summary[sp]['ridge_mae']:.6f}; ridge improved channels={summary[sp]['channel_improved_by_ridge']}/42")
 with (out/"channel_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(channel_rows[0]));w.writeheader();w.writerows(channel_rows)
 with (out/"object_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(object_rows[0]));w.writeheader();w.writerows(object_rows)
 # Channel-wise residual target distributions on Train only are descriptive diagnostics.
 summary["train_residual_target"]={"mean_abs_channel_mean":float(np.mean(np.abs(D[tr].mean(0)))) ,
 "median_channel_sd":float(np.median(D[tr].std(0))),
 "channels_sd_below_1e-6":int(np.sum(D[tr].std(0)<1e-6))}
 summary["interpretation_limits"]=["Residual correlation is predictive error covariance, not a physical adjacency graph.",
 "Channel indices are mapped back to the original 44-channel numbering after excluding source channels 18 and 25.",
 "Validation is the primary diagnostic split; Test outputs are exploratory only.",
 "Object groups, not episodes, are the comparison unit."]
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n")
 log(f"COMPLETE elapsed={time.time()-t0:.1f}s")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
