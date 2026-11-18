#!/usr/bin/env python3
"""Run object-disjoint pressure-only co-activation prediction baselines."""
import argparse, copy, csv, itertools, json, warnings
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,roc_auc_score,average_precision_score,f1_score,precision_score,recall_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
TAU=.7
SEED=20261116

def corr(x):
    x=np.asarray(x,dtype=np.float64);x=x-x.mean(axis=0,keepdims=True)
    den=np.sqrt(np.outer(np.sum(x*x,axis=0),np.sum(x*x,axis=0)))
    with np.errstate(divide="ignore",invalid="ignore"): c=(x.T@x)/den
    c=np.where(np.isfinite(c),np.clip(c,-1,1),np.nan);np.fill_diagonal(c,np.nan);return c

def reg_metrics(y,p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho=spearmanr(y,p).statistic if np.std(y)>1e-12 and np.std(p)>1e-12 else np.nan
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),
            "spearman":float(rho) if np.isfinite(rho) else None}

def bin_metrics(y,s,cut):
    yb=(np.asarray(y)>=TAU).astype(int);s=np.asarray(s)
    try: auc=float(roc_auc_score(yb,s))
    except ValueError: auc=None
    try: ap=float(average_precision_score(yb,s))
    except ValueError: ap=None
    pred=(s>=cut).astype(int)
    return {"auroc":auc,"average_precision":ap,"f1":float(f1_score(yb,pred,zero_division=0)),
      "precision":float(precision_score(yb,pred,zero_division=0)),"recall":float(recall_score(yb,pred,zero_division=0)),
      "positive_prevalence":float(yb.mean())}

def extract(mat,manifest,n_nodes=96,seed=SEED,k=5):
    p=np.asarray(mat["pressure"]);rec=np.asarray(mat["recordingId"]).ravel().astype(int)
    obj=np.asarray(mat["objectId"]).ravel().astype(int);ts=np.asarray(mat["ts"]).ravel().astype(float)
    valid=np.asarray(mat["hasValidLabel"]).ravel().astype(bool);frame=np.asarray(mat["frame"]).ravel().astype(int)
    mask=np.nanstd(p.astype(np.float32),axis=0)>0
    if int(mask.sum())!=548: raise ValueError("Expected 548 active sensor pixels")
    active=np.flatnonzero(mask.ravel());chosen=np.sort(np.random.default_rng(seed).choice(548,n_nodes,replace=False))
    flat=active[chosen];tri=np.triu_indices(n_nodes,1);a,b=tri;yy,xx=np.divmod(flat,32)
    geom=np.column_stack([(xx[a]+xx[b])/62,(yy[a]+yy[b])/62,np.abs(xx[a]-xx[b])/31,
      np.abs(yy[a]-yy[b])/31,np.hypot(xx[a]-xx[b],yy[a]-yy[b])/(31*np.sqrt(2))]).astype(np.float32)
    owners={}
    for split,entries in manifest["object_splits"].items():
        for e in entries:
            for rid in e["recording_ids"]: owners[int(rid)]=(int(e["object_id"]),split,e["object_name"])
    data={}
    for rid,(oid,split,name) in owners.items():
        ix=np.flatnonzero(rec==rid)
        if len(np.unique(obj[ix]))!=1 or int(obj[ix][0])!=oid: raise ValueError(f"object mismatch {rid}")
        ix=ix[np.argsort(ts[ix],kind="stable")];v=np.flatnonzero(valid[ix]);ep=ix[v[:k]];fp=ix[v[v>v[k-1]]]
        if len(fp)<4: raise ValueError(f"insufficient future frames {rid}")
        x=np.clip((p[ep].reshape(k,-1)[:,flat].astype(np.float32)-500)/150,0,1)
        future=np.clip((p[fp].reshape(len(fp),-1)[:,flat].astype(np.float32)-500)/150,0,1)
        y=corr(future)[tri].astype(np.float32);h=corr(x)[tri].astype(np.float32);ok=np.isfinite(y);hg=np.isfinite(h)
        t=ts[ep];tc=t-t.mean();d=tc@tc
        slope=(tc[:,None]*(x-x.mean(axis=0))).sum(axis=0)/d if d>0 else np.zeros(n_nodes)
        z=np.column_stack([x.mean(0),x.std(0),x.min(0),x.max(0),np.ptp(x,axis=0),slope,x[-1],x[0]])
        za,zb=z[a],z[b];pair=np.concatenate([za+zb,np.abs(za-zb),za*zb],axis=1)
        X=np.concatenate([pair,geom,np.nan_to_num(h,nan=0)[:,None],hg.astype(np.float32)[:,None]],axis=1)
        data[rid]={"object_id":oid,"object_name":str(name),"split":split,"X":X[ok].astype(np.float32),
          "y":y[ok],"history":np.nan_to_num(h[ok],nan=0),"n_future":len(fp),
          "early_span":float(t[-1]-t[0]),"early_consecutive":bool(np.all(np.diff(frame[ep])==1)),
          "positive_fraction":float(np.mean(y[ok]>=TAU))}
    if len(data)!=78: raise ValueError(f"Expected 78 recordings, got {len(data)}")
    return data,flat.tolist()

def stack(data,rids):
    return np.concatenate([data[r]["X"] for r in rids]),np.concatenate([data[r]["y"] for r in rids])

def episodes(data,split):
    by={}
    for r,d in data.items():
        if d["split"]==split: by.setdefault(d["object_id"],[]).append(r)
    out=[]
    for oid,rs in sorted(by.items()):
        rs=sorted(rs)
        if len(rs)!=3: raise ValueError("Each object must have three recordings")
        for k in (1,2):
            for s in itertools.combinations(rs,k): out.append((oid,k,s,tuple(r for r in rs if r not in s)))
    return out

def adapt_logistic(model,X,y,penalty=1.0):
    yb=(y>=TAU).astype(float);w0=model.coef_[0].astype(float);b0=float(model.intercept_[0])
    n=len(yb);n1=max(int(yb.sum()),1);n0=max(n-int(yb.sum()),1)
    sw=np.where(yb>0,n/(2*n1),n/(2*n0));sw=sw/sw.mean()
    def loss(v):
        z=X@v[:-1]+v[-1];delta=v-np.r_[w0,b0]
        return np.mean(sw*(np.logaddexp(0,z)-yb*z))+.5*penalty*(delta@delta)
    opt=minimize(loss,np.r_[w0,b0],method="L-BFGS-B",options={"maxiter":100})
    return opt.x[:-1],float(opt.x[-1])

def stats(vals):
    a=np.asarray(vals,float);a=a[np.isfinite(a)]
    return {"n":int(len(a)),"mean":float(a.mean()) if len(a) else None,
      "sd":float(a.std(ddof=1)) if len(a)>1 else (0.0 if len(a) else None)}

def evaluate(data,split,mlp,logit,train_mean,adapt_epochs=5):
    eps=[]
    for oid,k,s,q in episodes(data,split):
        Xs,ys=stack(data,s);adapted=copy.deepcopy(mlp);adapted.verbose=False
        for _ in range(adapt_epochs): adapted.partial_fit(Xs,ys)
        w,b=adapt_logistic(logit,Xs,ys);sm=float(np.mean([data[r]["y"].mean() for r in s]));rows=[]
        for rid in q:
            d=data[rid];X,y,h=d["X"],d["y"],d["history"]
            preds={"constant_train":np.full_like(y,train_mean),"constant_support":np.full_like(y,sm),
              "historical_correlation":h,"mlp_zero_shot":mlp.predict(X),"mlp_support_adapted":adapted.predict(X)}
            row={}
            for name,p in preds.items(): row[name]={"regression":reg_metrics(y,p),"binary":bin_metrics(y,p,TAU)}
            for name,p in {"logistic_zero_shot":logit.predict_proba(X)[:,1],
                "logistic_support_adapted":1/(1+np.exp(-np.clip(X@w+b,-35,35)))}.items():
                row[name]={"binary":bin_metrics(y,p,.5)}
            rows.append(row)
        em={}
        for name in rows[0]:
            em[name]={}
            for g in rows[0][name]:
                em[name][g]={}
                for metric in rows[0][name][g]:
                    v=[r[name][g][metric] for r in rows if r[name][g][metric] is not None]
                    em[name][g][metric]=float(np.mean(v)) if v else None
        eps.append({"object_id":oid,"object_name":data[s[0]]["object_name"],"shots":k,"metrics":em})
    objects=[]
    for oid in sorted({e["object_id"] for e in eps}):
        for k in (1,2):
            group=[e for e in eps if e["object_id"]==oid and e["shots"]==k]
            if not group: continue
            row={"object_id":oid,"object_name":group[0]["object_name"],"shots":k,"metrics":{}}
            for name in group[0]["metrics"]:
                row["metrics"][name]={}
                for g in group[0]["metrics"][name]:
                    row["metrics"][name][g]={}
                    for metric in group[0]["metrics"][name][g]:
                        v=[e["metrics"][name][g][metric] for e in group if e["metrics"][name][g][metric] is not None]
                        row["metrics"][name][g][metric]=float(np.mean(v)) if v else None
            objects.append(row)
    summary={}
    for k in (1,2):
        subset=[r for r in objects if r["shots"]==k];summary[str(k)]={"n_objects":len(subset),"models":{}}
        for name in (subset[0]["metrics"] if subset else []):
            summary[str(k)]["models"][name]={}
            for g in subset[0]["metrics"][name]:
                summary[str(k)]["models"][name][g]={}
                for metric in subset[0]["metrics"][name][g]:
                    summary[str(k)]["models"][name][g][metric]=stats([r["metrics"][name][g][metric] for r in subset])
    return {"summary_by_shots":summary,"object_level_metrics":objects}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",type=Path,required=True);ap.add_argument("--split-manifest",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--nodes",type=int,default=96);ap.add_argument("--mlp-epochs",type=int,default=50)
    ap.add_argument("--adapt-epochs",type=int,default=5);ap.add_argument("--logistic-prior-penalty",type=float,default=1.0)
    a=ap.parse_args();manifest=json.loads(a.split_manifest.read_text(encoding="utf-8"))
    data,flat=extract(loadmat(a.metadata,simplify_cells=True),manifest,a.nodes,a.seed)
    train=sorted(r for r,d in data.items() if d["split"]=="meta_train");X,y=stack(data,train)
    scaler=StandardScaler().fit(X);Xs=scaler.transform(X).astype(np.float32)
    mean=float(np.mean([data[r]["y"].mean() for r in train]))
    mlp=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,batch_size=2048,
      learning_rate="constant",learning_rate_init=5e-4,max_iter=a.mlp_epochs,
      early_stopping=False,random_state=a.seed,verbose=False).fit(Xs,y)
    logit=LogisticRegression(C=1,class_weight="balanced",max_iter=500,solver="lbfgs").fit(Xs,(y>=TAU).astype(int))
    for d in data.values(): d["X"]=scaler.transform(d["X"]).astype(np.float32)
    val=evaluate(data,"meta_validation",mlp,logit,mean,a.adapt_epochs)
    test=evaluate(data,"meta_test",mlp,logit,mean,a.adapt_epochs)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    coverage=[{"recording_id":r,"object_id":d["object_id"],"object_name":d["object_name"],
      "split":d["split"],"n_edges":len(d["y"]),"n_future_frames":d["n_future"],
      "early_span_sec":d["early_span"],"early_consecutive":d["early_consecutive"],
      "positive_fraction":d["positive_fraction"]} for r,d in sorted(data.items())]
    with (a.output_dir/"baseline_recording_coverage.csv").open("w",newline="",encoding="utf-8-sig") as f:
      w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    result={"scope":"STAG pressure-only; no video or multimodal data",
      "target":"continuous future Pearson correlation after earliest five valid input frames",
      "auxiliary_target":"r >= 0.7","pressure_transform":"clip((pressure-500)/150,0,1)",
      "object_split_seed":manifest["seed"],"sensor_subset_seed":a.seed,"nodes":a.nodes,
      "edges_per_recording":a.nodes*(a.nodes-1)//2,"recordings":len(data),
      "train_objects":18,"train_recordings":len(train),"train_target_mean":mean,
      "selected_sensor_flat_indices":flat,"mlp":{"hidden_layers":[64,32],"epochs":a.mlp_epochs,"adapt_epochs":a.adapt_epochs},
      "logistic":{"C":1.0,"class_weight":"balanced","support_prior_penalty":a.logistic_prior_penalty},
      "validation":val,"test":test,
      "limitations":["Fixed 96/548 sensor subset; no full-array claim.",
      "Only 18 train and 4 test objects; object is the inferential unit.",
      "Support/Query permutations reuse recordings and are not independent tasks.",
      "18/78 earliest-five inputs are non-consecutive; this is conditional on valid-contact frames.",
      "Threshold positives are rare; classification metrics need prevalence-aware interpretation."]}
    (a.output_dir/"baseline_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    objrows=[]
    for split,res in [("validation",val),("test",test)]:
      for o in res["object_level_metrics"]:
       for model,groups in o["metrics"].items():
        for group,ms in groups.items():
         objrows.append({"split":split,"object_id":o["object_id"],"object_name":o["object_name"],
           "shots":o["shots"],"model":model,"metric_group":group,**ms})
    fields=list(dict.fromkeys(k for row in objrows for k in row))
    with (a.output_dir/"baseline_object_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(objrows)
    print(json.dumps({"validation":val["summary_by_shots"],"test":test["summary_by_shots"]},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
