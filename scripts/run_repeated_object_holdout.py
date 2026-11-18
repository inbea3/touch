#!/usr/bin/env python3
"""Repeated object-level holdout evaluation for pressure-only coactivation baselines.

The sensor subset and all model hyperparameters remain fixed across splits.
Each repeat independently permutes the 26 non-empty object IDs into 18/4/4.
"""
import argparse,csv,json,sys,tempfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_object_episode_splits import build as build_split
from run_object_coactivation_baselines import extract,stack,evaluate,stats

def write_csv(path,rows):
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def fit_models(data,seed,mlp_epochs,adapt_epochs,logistic_penalty,sensor_seed,nodes):
    train=sorted(r for r,d in data.items() if d["split"]=="meta_train")
    X,y=stack(data,train)
    scaler=StandardScaler().fit(X)
    Xs=scaler.transform(X).astype(np.float32)
    train_mean=float(np.mean([data[r]["y"].mean() for r in train]))
    mlp=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,batch_size=2048,
      learning_rate="constant",learning_rate_init=5e-4,max_iter=mlp_epochs,
      early_stopping=False,random_state=20261116,verbose=False).fit(Xs,y)
    logit=LogisticRegression(C=1,class_weight="balanced",max_iter=500,solver="lbfgs").fit(Xs,(y>=.7).astype(int))
    for d in data.values(): d["X"]=scaler.transform(d["X"]).astype(np.float32)
    return mlp,logit,train_mean

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--n-repeats",type=int,default=20)
    ap.add_argument("--split-seed",type=int,default=20261118)
    ap.add_argument("--sensor-seed",type=int,default=20261116)
    ap.add_argument("--nodes",type=int,default=96)
    ap.add_argument("--mlp-epochs",type=int,default=50)
    ap.add_argument("--adapt-epochs",type=int,default=5)
    ap.add_argument("--logistic-prior-penalty",type=float,default=1.0)
    a=ap.parse_args()
    if a.n_repeats<2: ap.error("--n-repeats must be at least 2")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    mat=loadmat(a.metadata,simplify_cells=True)
    split_rows=[];object_rows=[];assignment_rows=[];test_appearances={};sensor_indices=None
    with tempfile.TemporaryDirectory(prefix="stag_repeated_splits_") as td:
      for i in range(a.n_repeats):
        split_seed=a.split_seed+i
        split_dir=Path(td)/f"split_{i:03d}"
        build_split(a.metadata,split_dir,seed=split_seed,k_early=5)
        manifest=json.loads((split_dir/"object_split_manifest.json").read_text(encoding="utf-8"))
        for split_name,entries in manifest["object_splits"].items():
          for entry in entries:
            assignment_rows.append({"repeat":i+1,"split_seed":split_seed,"object_id":entry["object_id"],"object_name":entry["object_name"],"split":split_name,"recording_ids":";".join(map(str,entry["recording_ids"]))})
        data,indices=extract(mat,manifest,n_nodes=a.nodes,seed=a.sensor_seed)
        if sensor_indices is None: sensor_indices=indices
        elif indices!=sensor_indices: raise RuntimeError("Sensor subset changed across repeats")
        mlp,logit,train_mean=fit_models(data,split_seed,a.mlp_epochs,a.adapt_epochs,
          a.logistic_prior_penalty,a.sensor_seed,a.nodes)
        val=evaluate(data,"meta_validation",mlp,logit,train_mean,a.adapt_epochs,a.logistic_prior_penalty)
        test=evaluate(data,"meta_test",mlp,logit,train_mean,a.adapt_epochs,a.logistic_prior_penalty)
        for split_name,res in [("meta_validation",val),("meta_test",test)]:
          for row in res["object_level_metrics"]:
            flat={"repeat":i+1,"split_seed":split_seed,"split":split_name,
              "object_id":row["object_id"],"object_name":row["object_name"],"shots":row["shots"]}
            for model,groups in row["metrics"].items():
              for group,metrics in groups.items():
                for metric,value in metrics.items():
                  flat[f"{model}__{group}__{metric}"]=value
            object_rows.append(flat)
            if split_name=="meta_test":
              key=(row["object_id"],row["shots"])
              test_appearances.setdefault(key,[]).append(flat)
        for shots in (1,2):
          models=test["summary_by_shots"][str(shots)]["models"]
          for model,groups in models.items():
            for group,metrics in groups.items():
              for metric,agg in metrics.items():
                split_rows.append({"repeat":i+1,"split_seed":split_seed,"shots":shots,
                  "model":model,"metric_group":group,"metric":metric,
                  "object_mean":agg["mean"],"object_sd":agg["sd"],"n_test_objects":agg["n"]})
        print(f"[{i+1}/{a.n_repeats}] split_seed={split_seed} "
          f"test_objects={','.join(x['object_name'] for x in manifest['object_splits']['meta_test'])}")
    write_csv(a.output_dir/"repeated_split_assignments.csv",assignment_rows)
    write_csv(a.output_dir/"repeated_split_object_metrics.csv",object_rows)
    write_csv(a.output_dir/"repeated_split_run_metrics.csv",split_rows)
    pooled=[]
    for (oid,shots),rows in sorted(test_appearances.items()):
      for col in rows[0]:
        if "__" not in col: continue
        vals=[r[col] for r in rows if isinstance(r[col],(int,float)) and np.isfinite(r[col])]
        pooled.append({"object_id":oid,"object_name":rows[0]["object_name"],"shots":shots,
          "metric":col,"mean_across_test_appearances":float(np.mean(vals)) if vals else None,
          "sd_across_test_appearances":float(np.std(vals,ddof=1)) if len(vals)>1 else (0.0 if vals else None),
          "n_test_appearances":len(vals)})
    write_csv(a.output_dir/"repeated_split_object_stability.csv",pooled)
    run_summary={}
    for shots in (1,2):
      run_summary[str(shots)]={}
      keys=sorted({(r["model"],r["metric_group"],r["metric"]) for r in split_rows if r["shots"]==shots})
      for model,group,metric in keys:
        vals=[r["object_mean"] for r in split_rows if r["shots"]==shots and r["model"]==model and r["metric_group"]==group and r["metric"]==metric and isinstance(r["object_mean"],(int,float)) and np.isfinite(r["object_mean"])]
        run_summary[str(shots)][f"{model}__{group}__{metric}"]=stats(vals)
    summary={"protocol":"Repeated random object-level 18/4/4 holdout",
      "n_repeats":a.n_repeats,"split_seeds":[a.split_seed+i for i in range(a.n_repeats)],
      "sensor_subset_seed":a.sensor_seed,"n_nodes":a.nodes,"edges_per_recording":a.nodes*(a.nodes-1)//2,
      "fixed_sensor_indices":sensor_indices,"mlp":{"hidden_layers":[64,32],"epochs":a.mlp_epochs,
      "learning_rate_init":.0005,"alpha":.001,"adapt_epochs":a.adapt_epochs},
      "logistic":{"C":1,"class_weight":"balanced","support_prior_penalty":a.logistic_prior_penalty},
      "aggregation":"within each repeat average over held-out objects; summarize repeat-level object means; object appearances are reused across repeats and are not independent",
      "repeat_level_summary":run_summary,
      "assignment_file":"repeated_split_assignments.csv",
      "limitations":["Repeated splits reuse the same 26 objects; repeat-level variation is a robustness summary, not independent replication.",
       "Each repeat has only four held-out objects.",
       "Only the fixed sensor subset is evaluated; no full-array claim.",
       "No hyperparameter tuning is performed on test objects.",
       "The first-five-valid-frame input can span different durations."]}
    (a.output_dir/"repeated_split_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(run_summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
