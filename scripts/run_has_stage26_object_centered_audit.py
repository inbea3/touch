#!/usr/bin/env python3
"""Stage 26: object-centered channel residual audit and leave-one-object-out stability."""
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
def corr(x):
 # rows=episodes, cols=channels; returns NaN for constant channels, handled downstream.
 if x.shape[0]<4:return np.full((x.shape[1],x.shape[1]),np.nan)
 c=np.corrcoef(x,rowvar=False)
 return c
def fisher_aggregate(mats):
 stack=np.stack(mats)
 out=np.full(stack.shape[1:],np.nan)
 for i in range(out.shape[0]):
  for j in range(out.shape[1]):
   vals=stack[:,i,j];vals=vals[np.isfinite(vals)]
   if len(vals):out[i,j]=np.tanh(np.mean(np.arctanh(np.clip(vals,-.999999,.999999))))
 return out
def pair_table(mat,channel_map):
 pairs=[]
 for i in range(mat.shape[0]):
  for j in range(i+1,mat.shape[1]):
   v=mat[i,j]
   if np.isfinite(v):pairs.append((abs(float(v)),i,j,float(v)))
 return [{"retained_i":i,"retained_j":j,"source_channel_i":channel_map[i],
          "source_channel_j":channel_map[j],"correlation":v}
         for _,i,j,v in sorted(pairs,reverse=True)]
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_window_dataset")
 ap.add_argument("--out",default="results/has_stage26_object_centered_audit")
 ap.add_argument("--ridge-alpha",type=float,default=1000)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(s):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {s}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 with np.load(Path(a.data)/"window_5.npz") as z:
  X=z["X"].astype(np.float64);Y=z["Y_future_mean"].astype(np.float64)
 rows=list(csv.DictReader(open(Path(a.data)/"window_5_index.csv",encoding="utf-8")))
 if X.shape!=(4192,168) or Y.shape!=(4192,42) or len(rows)!=4192:raise RuntimeError("Unexpected input shapes/index length")
 split=np.array([r["grouped_split"] for r in rows]);obj=np.array([r["object_group"] for r in rows])
 tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
 objects={s:sorted(set(obj[ix])) for s,ix in (("train",tr),("validation",va),("test",te))}
 if any(set(objects[x])&set(objects[y]) for x,y in (("train","validation"),("train","test"),("validation","test"))):raise RuntimeError("Object split leakage")
 channel_map=[i for i in range(44) if i not in (18,25)]
 if len(channel_map)!=42:raise RuntimeError("Channel mapping error")
 P=X[:,126:168];D=Y-P
 xm=X[tr].mean(0);xs=X[tr].std(0);xs[xs<1e-8]=1
 dm=D[tr].mean(0);ds=D[tr].std(0);ds[ds<1e-8]=1
 Z=(X-xm)/xs;T=(D-dm)/ds
 ridge=Ridge(alpha=a.ridge_alpha).fit(Z[tr],T[tr])
 PR=P+ridge.predict(Z)*ds+dm
 config={"stage":"26","source_dataset_run":45193420854,"prior_diagnostic_run":45211031230,
 "window":5,"input_dim":168,"target_dim":42,"channel_map_source_indices":channel_map,
 "ridge_alpha":a.ridge_alpha,"ridge_alpha_fixed_from_stage24b":True,"scalers_fit_on_train_only":True,
 "object_counts":{s:len(v) for s,v in objects.items()},
 "episode_counts":{s:int(np.sum(split==s)) for s in ("train","validation","test")},
 "correlation_method":"within each object, subtract that object's channel-wise residual mean before correlation; Fisher-z average object matrices equally",
 "primary_split":"Validation; six objects only","test_status":"exploratory"}
 (out/"config.json").write_text(json.dumps(config,indent=2))
 channel_rows=[];obj_channel_rows=[];obj_summary=[];corr_summaries={}
 for sp,ix in (("validation",va),("test",te)):
  matrices={"persistence":[],"ridge":[]};per_obj={}
  for ob in objects[sp]:
   jj=ix[obj[ix]==ob]
   obres={"persistence":P[jj]-Y[jj],"ridge":PR[jj]-Y[jj]}
   per_obj[ob]={}
   for model_name,E in obres.items():
    bias=E.mean(0);mae=np.abs(E).mean(0);rmse=np.sqrt((E**2).mean(0));sd=E.std(0)
    per_obj[ob][model_name]={"bias":bias,"mae":mae,"rmse":rmse,"sd":sd}
    for c,src in enumerate(channel_map):
     obj_channel_rows.append({"split":sp,"object":ob,"n_episodes":len(jj),"retained_channel":c,
      "source_channel_index":src,"model":model_name,"bias":float(bias[c]),"mae":float(mae[c]),
      "rmse":float(rmse[c]),"residual_sd":float(sd[c])})
    # Center within object before calculating correlation: removes constant object-level channel offsets.
    centered=E-E.mean(0,keepdims=True)
    matrices[model_name].append(corr(centered))
   bp=per_obj[ob]["persistence"];br=per_obj[ob]["ridge"]
   obj_summary.append({"split":sp,"object":ob,"n_episodes":len(jj),
    "persistence_mae":float(np.mean(bp["mae"])),"ridge_mae":float(np.mean(br["mae"])),
    "persistence_rmse":float(np.mean(bp["rmse"])),"ridge_rmse":float(np.mean(br["rmse"]))})
  for model_name,mats in matrices.items():
   agg=fisher_aggregate(mats);np.save(out/f"{sp}_{model_name}_within_object_corr_fisher.npy",agg)
   pairs=pair_table(agg,channel_map)
   corr_summaries[f"{sp}_{model_name}"]={"n_objects":len(mats),
    "mean_abs_offdiag":float(np.nanmean(np.abs(agg[np.triu_indices(42,1)]))),
    "median_abs_offdiag":float(np.nanmedian(np.abs(agg[np.triu_indices(42,1)]))),
    "max_abs_offdiag":float(np.nanmax(np.abs(agg[np.triu_indices(42,1)]))),
    "top_20_pairs":pairs[:20]}
  # Equal-object channel summaries, unlike pooled Stage25 metrics.
  for c,src in enumerate(channel_map):
   for model_name in ("persistence","ridge"):
    vals=[per_obj[o][model_name] for o in objects[sp]]
    channel_rows.append({"split":sp,"retained_channel":c,"source_channel_index":src,"model":model_name,
     "object_macro_bias":float(np.mean([v["bias"][c] for v in vals])),
     "object_macro_mae":float(np.mean([v["mae"][c] for v in vals])),
     "object_macro_rmse":float(np.mean([v["rmse"][c] for v in vals])),
     "object_macro_residual_sd":float(np.mean([v["sd"][c] for v in vals]))})
  # Leave-one-object-out pair stability, per model.
  for model_name,mats in matrices.items():
   loo=[]
   for omit in range(len(mats)):
    subset=[m for k,m in enumerate(mats) if k!=omit]
    agg=fisher_aggregate(subset);pairs=pair_table(agg,channel_map)
    loo.append({"omitted_object":objects[sp][omit],"top_10_pairs":pairs[:10]})
   corr_summaries[f"{sp}_{model_name}"]["leave_one_object_out"]=loo
   sets=[]
   for entry in loo:sets.append({(p["retained_i"],p["retained_j"]) for p in entry["top_10_pairs"]})
   freq={}
   for ss in sets:
    for pair in ss:freq[pair]=freq.get(pair,0)+1
   corr_summaries[f"{sp}_{model_name}"]["loo_top10_pair_frequency"]=[
    {"retained_i":i,"retained_j":j,"source_channel_i":channel_map[i],"source_channel_j":channel_map[j],"n_loo_runs":n}
    for (i,j),n in sorted(freq.items(),key=lambda kv:(-kv[1],kv[0]))]
 log("Validation per-object and object-centered correlation calculations complete")
 with (out/"channel_object_macro_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(channel_rows[0]));w.writeheader();w.writerows(channel_rows)
 with (out/"per_object_channel_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(obj_channel_rows[0]));w.writeheader();w.writerows(obj_channel_rows)
 with (out/"object_macro_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(obj_summary[0]));w.writeheader();w.writerows(obj_summary)
 summary={"stage":"26","correlation_summaries":corr_summaries,
 "object_macro_metrics":{sp:[r for r in obj_summary if r["split"]==sp] for sp in ("validation","test")},
 "channel_macro_metrics_file":"channel_object_macro_metrics.csv",
 "limitations":["Six Validation objects give low precision; leave-one-object-out is a stability diagnostic, not inferential confirmation.",
 "Within-object correlation is estimated from 65-255 episodes per object; finite-sample noise remains.",
 "Residual correlations are not physical adjacency. No sensor coordinates or validated physical layout are inferred.",
 "Test is exploratory only."]}
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n")
 log(f"COMPLETE elapsed={time.time()-t0:.1f}s")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps({"validation":corr_summaries,"objects":objects},indent=2))
if __name__=="__main__":main()
