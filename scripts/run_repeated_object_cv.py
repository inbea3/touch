#!/usr/bin/env python3
"""Repeated object-grouped K-fold CV for pressure-only co-activation baselines."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from run_object_coactivation_baselines import TAU,extract,stack,evaluate

def make_folds(object_ids,n_splits=5,n_repeats=5,seed=20261119):
    ids=sorted(map(int,object_ids))
    if len(ids)<n_splits or len(set(ids))!=len(ids) or n_splits<2 or n_repeats<1:
        raise ValueError("Invalid object IDs or fold/repeat count")
    repeats=[]
    for rep in range(n_repeats):
        perm=np.random.default_rng(seed+rep).permutation(ids)
        tests=np.array_split(perm,n_splits); folds=[]; seen=[]
        for j,a in enumerate(tests,1):
            test=sorted(map(int,a.tolist())); seen.extend(test); ts=set(test)
            folds.append({"repeat":rep+1,"fold":j,"seed":seed+rep,
                "train_object_ids":[x for x in ids if x not in ts],"test_object_ids":test})
        if len(seen)!=len(ids) or set(seen)!=set(ids): raise ValueError("Object not held out exactly once")
        sizes=[len(f["test_object_ids"]) for f in folds]
        if max(sizes)-min(sizes)>1: raise ValueError("Unbalanced folds")
        repeats.append({"repeat":rep+1,"seed":seed+rep,"folds":folds})
    return repeats

def all_object_manifest(fixed):
    entries=sorted([e for v in fixed["object_splits"].values() for e in v],key=lambda x:int(x["object_id"]))
    ids=[int(e["object_id"]) for e in entries]
    recs=[int(r) for e in entries for r in e["recording_ids"]]
    if len(ids)!=26 or len(set(ids))!=26 or len(recs)!=78 or len(set(recs))!=78:
        raise ValueError("Expected 26 objects and 78 unique recordings")
    if any(len(e["recording_ids"])!=3 for e in entries): raise ValueError("Expected 3 recordings per object")
    return {"object_splits":{"all_objects":entries}},ids

def fit_eval(data,fold,model_seed,epochs,adapt_epochs,penalty):
    tr=set(fold["train_object_ids"]); te=set(fold["test_object_ids"])
    if tr&te: raise ValueError("Object leakage")
    train=sorted(r for r,d in data.items() if int(d["object_id"]) in tr)
    if len(train)!=3*len(tr): raise ValueError("Training recording count mismatch")
    X,y=stack(data,train); scaler=StandardScaler().fit(X); Xs=scaler.transform(X).astype(np.float32)
    mean=float(np.mean([data[r]["y"].mean() for r in train]))
    mlp=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,batch_size=2048,
        learning_rate="constant",learning_rate_init=5e-4,max_iter=epochs,
        early_stopping=False,random_state=model_seed,verbose=False).fit(Xs,y)
    logit=LogisticRegression(C=1,class_weight="balanced",max_iter=500,
        solver="lbfgs",random_state=model_seed).fit(Xs,(y>=TAU).astype(int))
    fd={}
    for rid,d in data.items():
        q=dict(d); q["split"]="meta_train" if int(d["object_id"]) in tr else "meta_test"
        q["X"]=scaler.transform(d["X"]).astype(np.float32); fd[rid]=q
    result=evaluate(fd,"meta_test",mlp,logit,mean,adapt_epochs,penalty)
    return result,len(train),mean

def flatten(result,rep,fold):
    rows=[]
    for o in result["object_level_metrics"]:
        for model,groups in o["metrics"].items():
            for group,metrics in groups.items():
                for metric,value in metrics.items():
                    rows.append({"repeat":rep,"fold":fold,"object_id":o["object_id"],
                        "object_name":o["object_name"],"shots":o["shots"],"model":model,
                        "metric_group":group,"metric":metric,"value":value})
    return rows

def aggregate(rows):
    g={}
    for r in rows:
        v=r["value"]
        if v is None or not np.isfinite(float(v)): continue
        k=(r["object_id"],r["object_name"],r["shots"],r["model"],r["metric_group"],r["metric"])
        g.setdefault(k,[]).append(float(v))
    out=[]
    for k,vs in sorted(g.items()):
        oid,name,shots,model,group,metric=k
        out.append({"object_id":oid,"object_name":name,"shots":shots,"model":model,
            "metric_group":group,"metric":metric,"n_repeats":len(vs),
            "mean_across_repeats":float(np.mean(vs)),
            "sd_across_repeats":float(np.std(vs,ddof=1)) if len(vs)>1 else 0.0})
    return out

def boot(values,seed,n_boot):
    x=np.asarray([v for v in values if v is not None and np.isfinite(float(v))],float)
    if not len(x): return {"n_objects":0,"mean":None,"sd":None,"ci95_low":None,"ci95_high":None}
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),(n_boot,len(x)))
    m=x[idx].mean(axis=1)
    return {"n_objects":len(x),"mean":float(x.mean()),
        "sd":float(x.std(ddof=1)) if len(x)>1 else 0.0,"median":float(np.median(x)),
        "ci95_low":float(np.quantile(m,.025)),"ci95_high":float(np.quantile(m,.975))}

def summarize(rows,seed,n_boot):
    out={}
    keys=sorted({(r["shots"],r["model"],r["metric_group"],r["metric"]) for r in rows})
    for shots,model,group,metric in keys:
        vals=[r["mean_across_repeats"] for r in rows if (r["shots"],r["model"],r["metric_group"],r["metric"])==(shots,model,group,metric)]
        out.setdefault(str(shots),{}).setdefault(model,{}).setdefault(group,{})[metric]=boot(
            vals,seed+shots*1000+sum(map(ord,model+group+metric)),n_boot)
    return out

def paired(rows,seed,n_boot):
    d={(r["object_id"],r["shots"],r["model"],r["metric_group"],r["metric"]):r["mean_across_repeats"] for r in rows}
    out=[]
    for shots in (1,2):
        for a,b,label in [("constant_train","mlp_zero_shot","constant_minus_mlp"),
            ("mlp_zero_shot","mlp_support_adapted","zero_shot_minus_adapted"),
            ("constant_train","mlp_support_adapted","constant_minus_adapted")]:
            diffs=[]; rel=[]
            for oid in sorted({r["object_id"] for r in rows}):
                x=d.get((oid,shots,a,"regression","mae")); y=d.get((oid,shots,b,"regression","mae"))
                if x is None or y is None: continue
                diffs.append(x-y)
                if a=="constant_train" and abs(x)>1e-12: rel.append((x-y)/x)
            out.append({"shots":shots,"comparison":label,"absolute_mae_improvement":boot(diffs,seed+shots,n_boot),
                "relative_mae_reduction":boot(rel,seed+shots+99,n_boot) if rel else None,
                "objects_improved":int(np.sum(np.asarray(diffs)>0)),"n_objects":len(diffs)})
    return out

def write_csv(path,rows):
    if not rows: raise ValueError(f"Refusing empty CSV: {path}")
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,required=True);p.add_argument("--split-manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--folds",type=int,default=5);p.add_argument("--split-seed",type=int,default=20261119)
    p.add_argument("--sensor-seed",type=int,default=20261116);p.add_argument("--model-seed",type=int,default=20261116)
    p.add_argument("--nodes",type=int,default=96);p.add_argument("--mlp-epochs",type=int,default=50)
    p.add_argument("--adapt-epochs",type=int,default=5);p.add_argument("--logistic-prior-penalty",type=float,default=1.0)
    p.add_argument("--bootstrap-reps",type=int,default=5000);a=p.parse_args();t0=time.time()
    fixed=json.loads(a.split_manifest.read_text(encoding="utf-8")); manifest,ids=all_object_manifest(fixed)
    reps=make_folds(ids,a.folds,a.repeats,a.split_seed)
    data,sensors=extract(loadmat(a.metadata,simplify_cells=True),manifest,a.nodes,a.sensor_seed)
    if len(data)!=78: raise ValueError("Expected 78 recordings")
    long=[]; fold_summary=[]; repeat_summary=[]; fold_manifest=[]
    for rep in reps:
        this=[]
        for f in rep["folds"]:
            res,ntrain,trainmean=fit_eval(data,f,a.model_seed,a.mlp_epochs,a.adapt_epochs,a.logistic_prior_penalty)
            rows=flatten(res,rep["repeat"],f["fold"]);long.extend(rows);this.extend(rows)
            fold_manifest.append({**f,"n_train_recordings":ntrain,"n_test_recordings":3*len(f["test_object_ids"])})
            for shot,block in res["summary_by_shots"].items():
                for model,groups in block["models"].items():
                    for group,metrics in groups.items():
                        for metric,st in metrics.items():
                            fold_summary.append({"repeat":rep["repeat"],"fold":f["fold"],"shots":int(shot),
                                "n_objects":block["n_objects"],"model":model,"metric_group":group,"metric":metric,**st})
            print(f"Completed repeat {rep['repeat']}/{a.repeats}, fold {f['fold']}/{a.folds}",flush=True)
        ra=aggregate(this); rs=summarize(ra,a.split_seed+rep["repeat"]*100,a.bootstrap_reps)
        for shot,models in rs.items():
            for model,groups in models.items():
                for group,metrics in groups.items():
                    for metric,st in metrics.items():
                        repeat_summary.append({"repeat":rep["repeat"],"shots":int(shot),"model":model,
                            "metric_group":group,"metric":metric,**st})
    obj=aggregate(long); summary=summarize(obj,a.split_seed,a.bootstrap_reps)
    diffs=paired(obj,a.split_seed,a.bootstrap_reps); out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    audit={"each_object_held_out_once_per_repeat":True,"train_test_object_disjoint":True,
        "balanced_fold_sizes":True,"held_out_appearances_per_object":a.repeats,
        "n_objects":len(ids),"n_recordings":len(data)}
    split_doc={"design":f"repeated {a.folds}-fold grouped CV by object","split_seed":a.split_seed,
        "repeats":a.repeats,"folds_per_repeat":a.folds,"object_ids":ids,
        "folds":fold_manifest,"audit":audit,
        "rule":"All three recordings of an object stay in one fold; each object is test once per repeat."}
    (out/"repeated_cv_split_manifest.json").write_text(json.dumps(split_doc,ensure_ascii=False,indent=2),encoding="utf-8")
    result={"scope":"STAG pressure-only; no video or multimodal data",
        "design":"Repeated object-grouped outer CV; fixed hyperparameters, no held-out-fold tuning",
        "n_objects":len(ids),"n_recordings":len(data),"n_repeats":a.repeats,"folds_per_repeat":a.folds,
        "n_outer_fits":a.repeats*a.folds,"held_out_appearances_per_object":a.repeats,
        "nodes":a.nodes,"edges_per_recording":a.nodes*(a.nodes-1)//2,"sensor_seed":a.sensor_seed,
        "selected_sensor_flat_indices":sensors,"model_seed":a.model_seed,
        "mlp":{"hidden_layers":[64,32],"alpha":.001,"epochs":a.mlp_epochs,
            "learning_rate_init":.0005,"adapt_epochs":a.adapt_epochs},
        "logistic":{"C":1,"class_weight":"balanced","support_prior_penalty":a.logistic_prior_penalty},
        "inferential_unit":"26 objects after averaging each object's held-out appearances across repeats",
        "bootstrap":{"unit":"object","repetitions":a.bootstrap_reps,"interval":"95% percentile CI"},
        "summary_by_shots":summary,"paired_mae_differences":diffs,
        "limitations":["Fold training sets overlap; fold summaries are descriptive, not independent replicates.",
            "Only 26 distinct objects; bootstrap resamples objects, never sensor edges.",
            "Fixed 96/548 sensor subset; results do not establish full-array performance.",
            "Hyperparameters are frozen from the first baseline run; this is not nested CV.",
            "Support/Query permutations reuse recordings; aggregate by object."],
        "runtime_seconds":time.time()-t0}
    (out/"repeated_cv_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(out/"repeated_cv_object_repeat_metrics.csv",long);write_csv(out/"repeated_cv_object_metrics.csv",obj)
    write_csv(out/"repeated_cv_fold_summary.csv",fold_summary);write_csv(out/"repeated_cv_repeat_summary.csv",repeat_summary)
    summary_rows=[]
    for shots,models in summary.items():
        for model,groups in models.items():
            for group,metrics in groups.items():
                for metric,stat in metrics.items():
                    summary_rows.append({"shots":shots,"model":model,"metric_group":group,"metric":metric,**stat})
    write_csv(out/"repeated_cv_summary.csv",summary_rows)
    paired_rows=[]
    for item in diffs:
        absolute=item["absolute_mae_improvement"];relative=item["relative_mae_reduction"] or {}
        paired_rows.append({"shots":item["shots"],"comparison":item["comparison"],"n_objects":item["n_objects"],
            "positive_object_count":item["positive_object_count"],"absolute_improvement_mean":absolute.get("mean"),
            "absolute_improvement_sd":absolute.get("sd"),"absolute_ci95_low":absolute.get("ci95_low"),
            "absolute_ci95_high":absolute.get("ci95_high"),"relative_reduction_mean":relative.get("mean"),
            "relative_ci95_low":relative.get("ci95_low"),"relative_ci95_high":relative.get("ci95_high")})
    write_csv(out/"repeated_cv_paired_differences.csv",paired_rows)
    object_mae=[]
    for oid in sorted({r["object_id"] for r in obj}):
        for shots in (1,2):
            look={(r["model"],r["metric_group"],r["metric"]):r["mean_across_repeats"]
                for r in obj if r["object_id"]==oid and r["shots"]==shots}
            if not look: continue
            constant=look[("constant_train","regression","mae")]
            zero=look[("mlp_zero_shot","regression","mae")]
            adapted=look[("mlp_support_adapted","regression","mae")]
            name=next(r["object_name"] for r in obj if r["object_id"]==oid)
            object_mae.append({"object_id":oid,"object_name":name,"shots":shots,
                "constant_train_mae":constant,"mlp_zero_shot_mae":zero,"mlp_support_adapted_mae":adapted,
                "constant_minus_mlp_zero_shot":constant-zero,"constant_minus_mlp_adapted":constant-adapted,
                "zero_shot_minus_adapted":zero-adapted})
    write_csv(out/"repeated_cv_object_mae.csv",object_mae)
    print(json.dumps({"summary_by_shots":summary,"paired_mae_differences":diffs,
        "runtime_seconds":result["runtime_seconds"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
