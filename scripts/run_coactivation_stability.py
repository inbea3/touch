#!/usr/bin/env python3
"""Pressure-only co-activation estimator and label-stability audit."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import LedoitWolf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from touch_research.decomposition.stag import event_window

def vec(m,k,dtype):
    if k not in m: raise KeyError(f"metadata.mat 缺少字段: {k}")
    return np.asarray(m[k]).reshape(-1).astype(dtype)

def corr_pearson(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<3: raise ValueError("相关估计至少需要 3 个观测")
    sd=np.std(x,axis=0,ddof=1); good=np.isfinite(sd)&(sd>1e-12)
    out=np.full((x.shape[1],x.shape[1]),np.nan,dtype=np.float32)
    if good.sum()<2:return out
    z=x[:,good]-np.mean(x[:,good],axis=0,keepdims=True)
    z/=np.sqrt(np.sum(z*z,axis=0,keepdims=True))
    c=np.clip(z.T@z,-1,1); ix=np.flatnonzero(good)
    out[np.ix_(ix,ix)]=c.astype(np.float32);np.fill_diagonal(out,np.nan)
    return out

def corr_spearman(x):
    return corr_pearson(rankdata(np.asarray(x,dtype=np.float64),axis=0,method="average"))

def corr_ledoit_wolf(x):
    x=np.asarray(x,dtype=np.float64);sd=np.std(x,axis=0,ddof=1)
    good=np.isfinite(sd)&(sd>1e-12)
    out=np.full((x.shape[1],x.shape[1]),np.nan,dtype=np.float32)
    if good.sum()<2 or x.shape[0]<3:return out,None
    model=LedoitWolf(store_precision=False).fit(x[:,good]);cov=model.covariance_
    d=np.sqrt(np.maximum(np.diag(cov),0));den=np.outer(d,d)
    with np.errstate(divide="ignore",invalid="ignore"): c=cov/den
    ix=np.flatnonzero(good);out[np.ix_(ix,ix)]=np.clip(c,-1,1).astype(np.float32)
    np.fill_diagonal(out,np.nan)
    return out,float(model.shrinkage_)

def upper(a): return np.asarray(a)[np.triu_indices(np.asarray(a).shape[0],k=1)]

def compare(a,b,taus=(.5,.7,.9)):
    av,bv=upper(a),upper(b);ok=np.isfinite(av)&np.isfinite(bv)
    if ok.sum()<3:return {"common_edges":int(ok.sum()),"edge_spearman":None,"mean_abs_difference":None,"jaccard":{}}
    out={"common_edges":int(ok.sum()),"edge_spearman":float(spearmanr(av[ok],bv[ok]).statistic),
         "mean_abs_difference":float(np.mean(np.abs(av[ok]-bv[ok]))),"jaccard":{}}
    for tau in taus:
        aa,bb=av[ok]>=tau,bv[ok]>=tau;u=int(np.sum(aa|bb))
        out["jaccard"][str(tau)]=float(np.sum(aa&bb)/u) if u else 1.
    return out

def block_indices(n,length,rng):
    starts=rng.integers(0,n,size=int(np.ceil(n/length)))
    return np.concatenate([((s+np.arange(length))%n) for s in starts])[:n]

def bootstrap_time(x,node_ids,reps,block_len,seed):
    xx=np.asarray(x[:,node_ids],dtype=float);ref=corr_pearson(xx);rng=np.random.default_rng(seed)
    diffs,rhos,jacs=[],[],[]
    for _ in range(reps):
        cb=corr_pearson(xx[block_indices(len(xx),block_len,rng)])
        m=compare(ref,cb,taus=(.7,))
        if m["mean_abs_difference"] is not None:
            diffs.append(m["mean_abs_difference"]);rhos.append(m["edge_spearman"]);jacs.append(m["jaccard"]["0.7"])
    def med(a):return float(np.median(a)) if a else None
    return {"n_replicates":len(diffs),"block_length_frames":block_len,"node_subsample_n":len(node_ids),
            "median_abs_corr_difference":med(diffs),"median_edge_spearman":med(rhos),"median_jaccard":med(jacs)}

def summary(vals):
    a=np.asarray(vals,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return {"n_recordings":0}
    return {"n_recordings":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "sd":float(a.std(ddof=1)) if len(a)>1 else 0.,"q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",type=Path,default=Path("data/interim/metadata.mat"))
    ap.add_argument("--output-dir",type=Path,default=Path("results/coactivation_stability"))
    ap.add_argument("--k-early",type=int,default=5)
    ap.add_argument("--threshold",type=float,default=.5)
    ap.add_argument("--radius",type=int,default=80)
    ap.add_argument("--block-length",type=int,default=10)
    ap.add_argument("--bootstrap-reps",type=int,default=20)
    ap.add_argument("--bootstrap-nodes",type=int,default=96)
    ap.add_argument("--seed",type=int,default=20261116)
    a=ap.parse_args()
    m=loadmat(a.metadata,simplify_cells=True);p=np.asarray(m["pressure"]);n=len(p)
    rec=vec(m,"recordingId",np.int64);obj=vec(m,"objectId",np.int64);valid=vec(m,"hasValidLabel",bool)
    trans=vec(m,"isTransition",bool);grasp=vec(m,"isGrasp" if "isGrasp" in m else "isGrap",bool)
    direction=vec(m,"transitionDir",np.int64);names=[str(v) for v in np.asarray(m["objects"],dtype=object).reshape(-1)]
    if any(len(v)!=n for v in (rec,obj,valid,trans,grasp,direction)):raise ValueError("pressure 与 metadata 长度不一致")
    mask=np.nanstd(p.astype(np.float32),axis=0)>0
    if mask.sum()!=548:raise ValueError(f"预期 548 个有效传感器，实际 {mask.sum()}")
    node_ids=np.sort(np.random.default_rng(a.seed).choice(548,min(a.bootstrap_nodes,548),replace=False))
    rows,boots=[],[]
    for rid in np.unique(rec):
        ids=np.flatnonzero(rec==rid);classes=np.unique(obj[ids])
        if len(classes)!=1:raise ValueError(f"Recording {rid} 内 objectId 不唯一")
        cls=int(classes[0])
        if cls==0:continue
        x=np.clip((p[ids][:,mask].astype(np.float32)-500)/150,0,1)
        vi=np.flatnonzero(valid[ids])
        if len(vi)<=a.k_early+3:continue
        boundary=int(vi[a.k_early-1]);future=vi[vi>boundary];xf=x[future]
        pear=corr_pearson(xf);spear=corr_spearman(xf);lw,shrink=corr_ledoit_wolf(xf)
        sm=gaussian_filter1d(x.mean(axis=1,dtype=np.float64),3,mode="reflect")
        event_windows=[];expanded_ids=[];raw_parts=[];obj_parts=[];event_corrs=[];n_reliable=0
        for t in np.flatnonzero(trans[ids]):
            off=int(direction[ids][t]);q=t+off
            if off<=0 or q>=len(ids) or not grasp[ids][t] or q<=boundary:continue
            n_reliable+=1
            win=event_window(sm,int(t),int(q),radius=a.radius,threshold=a.threshold,normalization="recording")
            if win is None:continue
            lo=max(int(win[0]),boundary+1);hi=int(win[1])
            if hi<lo:continue
            fi=np.arange(lo,hi+1);raw=x[fi];objx=raw-x[t][None,:]
            expanded_ids.extend(fi.tolist());raw_parts.append(raw);obj_parts.append(objx)
            event_windows.append((int(t),int(q),lo,hi))
            if len(fi)>=5:event_corrs.append(corr_pearson(objx))
        expanded_n=len(expanded_ids);unique_ids=np.unique(expanded_ids) if expanded_ids else np.array([],dtype=int)
        unique_n=len(unique_ids);overlap=1-unique_n/expanded_n if expanded_n else None
        raw_exp=np.concatenate(raw_parts) if raw_parts else np.empty((0,x.shape[1]))
        obj_exp=np.concatenate(obj_parts) if obj_parts else np.empty((0,x.shape[1]))
        raw_unique=x[unique_ids] if unique_n else np.empty((0,x.shape[1]))
        c_raw_exp=corr_pearson(raw_exp) if len(raw_exp)>=3 else None
        c_raw_unique=corr_pearson(raw_unique) if len(raw_unique)>=3 else None
        c_obj_exp=corr_pearson(obj_exp) if len(obj_exp)>=3 else None
        if event_corrs:
            ez=np.stack([np.arctanh(np.clip(c,-.999,.999)) for c in event_corrs])
            finite=np.isfinite(ez);num=np.where(finite,ez,0).sum(axis=0);den=finite.sum(axis=0)
            mean_z=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
            c_event_balanced=np.tanh(mean_z)
        else:c_event_balanced=None
        m_overlap=compare(c_raw_exp,c_raw_unique) if c_raw_exp is not None and c_raw_unique is not None else {}
        m_balanced=compare(c_obj_exp,c_event_balanced) if c_obj_exp is not None and c_event_balanced is not None else {}
        m_ps=compare(pear,spear);m_pl=compare(pear,lw)
        row={"recording_id":int(rid),"object_id":cls,"object_name":names[cls],
             "n_valid_frames":int(len(vi)),"n_future_frames":int(len(future)),
             "n_reliable_events_after_input":n_reliable,"n_windows_after_input":len(event_windows),
             "n_event_matrices_min5frames":len(event_corrs),"expanded_event_frames":expanded_n,
             "unique_event_frames":unique_n,"duplicated_frame_occurrences":expanded_n-unique_n,
             "overlap_fraction_occurrences":overlap,
             "pearson_spearman_edge_rho":m_ps.get("edge_spearman"),
             "pearson_ledoitwolf_edge_rho":m_pl.get("edge_spearman"),
             "pearson_ledoitwolf_mean_abs_diff":m_pl.get("mean_abs_difference"),
             "ledoitwolf_shrinkage":shrink,
             "expanded_vs_unique_raw_edge_rho":m_overlap.get("edge_spearman"),
             "expanded_vs_unique_raw_jaccard_tau07":m_overlap.get("jaccard",{}).get("0.7"),
             "expanded_vs_event_balanced_obj_edge_rho":m_balanced.get("edge_spearman"),
             "expanded_vs_event_balanced_obj_jaccard_tau07":m_balanced.get("jaccard",{}).get("0.7")}
        for tau in (.5,.7,.9):
            vals=upper(pear);fin=np.isfinite(vals)
            row[f"future_pearson_positive_edge_fraction_tau{tau:g}"]=float(np.sum(fin&(vals>=tau))/fin.sum())
        rows.append(row)
        if a.bootstrap_reps>0:
            b=bootstrap_time(xf,node_ids,a.bootstrap_reps,a.block_length,a.seed+int(rid))
            b.update({"recording_id":int(rid),"object_id":cls});boots.append(b)
    fields=[k for k in rows[0] if k not in ("recording_id","object_id","object_name")]
    sums={k:summary([r[k] for r in rows]) for k in fields}
    bfields=[k for k in boots[0] if k not in ("recording_id","object_id")] if boots else []
    bsums={k:summary([r[k] for r in boots]) for k in bfields}
    result={"study_scope":"STAG pressure-only; no synchronized video or image annotations",
      "n_recordings":len(rows),"n_objects":len(set(r["object_id"] for r in rows)),
      "n_sensors":548,"n_undirected_edges":548*547//2,
      "protocol":{"early_input":f"first {a.k_early} hasValidLabel frames",
       "target":"later hasValidLabel frames in the same recording",
       "event_windows":"reliable official events with peak after early-input boundary; windows clipped to strictly future frames",
       "overlap":"expanded event occurrences vs unique raw frame union; baseline-subtracted pressure uses equal-event Fisher-z averaging as sensitivity comparison",
       "estimators":["Pearson","Spearman rank correlation","Ledoit-Wolf shrinkage correlation"],
       "bootstrap":f"circular moving-block bootstrap, block={a.block_length}, reps={a.bootstrap_reps}, fixed node subsample={len(node_ids)}",
       "dependence":"Recording is the summary unit; sensor edges within a recording are dependent."},
      "recording_level_summary":sums,"bootstrap_recording_level_summary":bsums,
      "recordings":rows,"bootstrap":boots}
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"coactivation_stability_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for filename,data in [("coactivation_stability_recording_metrics.csv",rows),("coactivation_stability_block_bootstrap.csv",boots)]:
        if data:
            with (out/filename).open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(f"recordings={len(rows)}, objects={result['n_objects']}, sensors=548, edges={result['n_undirected_edges']}")
    for k in ("overlap_fraction_occurrences","pearson_spearman_edge_rho","pearson_ledoitwolf_edge_rho",
              "expanded_vs_unique_raw_edge_rho","expanded_vs_unique_raw_jaccard_tau07",
              "expanded_vs_event_balanced_obj_edge_rho","expanded_vs_event_balanced_obj_jaccard_tau07"):
        print(k,sums.get(k))
    print("block bootstrap",bsums)
if __name__=="__main__":main()
