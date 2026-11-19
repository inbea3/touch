#!/usr/bin/env python3
"""Within-Recording event-cluster bootstrap for STAG pressure-only labels."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

def corr(x):
    x=np.asarray(x,dtype=float);sd=x.std(axis=0,ddof=1);good=np.isfinite(sd)&(sd>1e-12)
    out=np.full((x.shape[1],x.shape[1]),np.nan,dtype=np.float32)
    if good.sum()<2:return out
    z=x[:,good]-x[:,good].mean(axis=0,keepdims=True);z/=np.sqrt((z*z).sum(axis=0,keepdims=True))
    c=np.clip(z.T@z,-1,1);ix=np.flatnonzero(good);out[np.ix_(ix,ix)]=c;np.fill_diagonal(out,np.nan)
    return out

def window(y,t,q,radius=80,threshold=.5):
    span=float(y.max()-y.min())
    if span<=0:return None
    z=(y-y.min())/span
    if z[q]<threshold:return None
    lo=hi=q;ll=max(0,q-radius);hh=min(len(y)-1,q+radius)
    while lo>ll and z[lo-1]>=threshold:lo-=1
    while hi<hh and z[hi+1]>=threshold:hi+=1
    return lo,hi

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,default=Path("data/interim/metadata.mat"))
    p.add_argument("--output-dir",type=Path,default=Path("results/coactivation_stability"))
    p.add_argument("--k-early",type=int,default=5)
    p.add_argument("--min-event-frames",type=int,default=5)
    p.add_argument("--bootstrap-reps",type=int,default=100)
    p.add_argument("--bootstrap-nodes",type=int,default=64)
    p.add_argument("--seed",type=int,default=20261116)
    a=p.parse_args();m=loadmat(a.metadata,simplify_cells=True);raw=np.asarray(m["pressure"])
    rec=np.asarray(m["recordingId"]).reshape(-1).astype(int);obj=np.asarray(m["objectId"]).reshape(-1).astype(int)
    valid=np.asarray(m["hasValidLabel"]).reshape(-1).astype(bool);tr=np.asarray(m["isTransition"]).reshape(-1).astype(bool)
    gr=np.asarray(m["isGrasp"] if "isGrasp" in m else m["isGrap"]).reshape(-1).astype(bool)
    dr=np.asarray(m["transitionDir"]).reshape(-1).astype(int)
    names=[str(v) for v in np.asarray(m["objects"],dtype=object).reshape(-1)]
    mask=np.nanstd(raw.astype(np.float32),axis=0)>0
    if mask.sum()!=548:raise ValueError(f"expected 548 sensors, got {mask.sum()}")
    nodes=np.sort(np.random.default_rng(a.seed).choice(548,min(a.bootstrap_nodes,548),replace=False));rows=[]
    for rid in np.unique(rec):
        ids=np.flatnonzero(rec==rid);cls=int(obj[ids[0]])
        if cls==0:continue
        x=np.clip((raw[ids][:,mask].astype(np.float32)-500)/150,0,1);v=np.flatnonzero(valid[ids])
        if len(v)<=a.k_early+3:continue
        boundary=int(v[a.k_early-1]);sm=gaussian_filter1d(x.mean(axis=1,dtype=np.float64),3,mode="reflect");mats=[]
        for t in np.flatnonzero(tr[ids]):
            q=int(t+dr[ids][t])
            if dr[ids][t]<=0 or q>=len(ids) or q<=boundary or not gr[ids][t]:continue
            win=window(sm,int(t),q)
            if win is None:continue
            lo=max(win[0],boundary+1);hi=win[1]
            if hi-lo+1<a.min_event_frames:continue
            mats.append(corr(x[lo:hi+1,nodes]-x[t,nodes][None,:]))
        row={"recording_id":int(rid),"object_id":cls,"object_name":names[cls],"n_usable_events":len(mats),
             "bootstrap_reps":0,"node_subsample_n":len(nodes),"median_ci_width":None,"fraction_ci_excludes_zero":None}
        if len(mats)>=2:
            stack=np.stack([np.arctanh(np.clip(c,-.999,.999)) for c in mats]);finite=np.isfinite(stack);stack=np.where(finite,stack,0)
            rng=np.random.default_rng(a.seed+int(rid));boot=[]
            for _ in range(a.bootstrap_reps):
                take=rng.integers(0,len(mats),size=len(mats));den=finite[take].sum(axis=0);num=stack[take].sum(axis=0)
                mean=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);boot.append(np.tanh(mean))
            b=np.stack(boot)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                lo=np.nanpercentile(b,2.5,axis=0);hi=np.nanpercentile(b,97.5,axis=0)
            width=hi-lo;ex=(lo>0)|(hi<0);tri=np.triu_indices(len(nodes),1);w=width[tri];e=ex[tri];ok=np.isfinite(w)
            row.update({"bootstrap_reps":a.bootstrap_reps,
              "median_ci_width":float(np.nanmedian(w)),
              "fraction_ci_excludes_zero":float(e[ok].mean()) if ok.any() else None})
        rows.append(row)
    def summ(key):
        z=np.asarray([r[key] for r in rows if r[key] is not None],dtype=float)
        if not len(z):return {"n_recordings":0}
        return {"n_recordings":len(z),"mean":float(z.mean()),"median":float(np.median(z)),
                "q25":float(np.quantile(z,.25)),"q75":float(np.quantile(z,.75))}
    result={"scope":"pressure-only; no video or image data",
      "method":"within-Recording event-cluster bootstrap of event-level Fisher-z Pearson matrices",
      "minimum_event_frames":a.min_event_frames,"bootstrap_reps":a.bootstrap_reps,"node_subsample_n":len(nodes),
      "recording_level_summary":{"usable_events":summ("n_usable_events"),
        "median_95pct_interval_width":summ("median_ci_width"),
        "fraction_intervals_excluding_zero":summ("fraction_ci_excludes_zero")},"recordings":rows}
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"coactivation_event_bootstrap_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"coactivation_event_bootstrap_recording.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps(result["recording_level_summary"],ensure_ascii=False,indent=2))
if __name__=="__main__":main()
