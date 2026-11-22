#!/usr/bin/env python3
"""STAG Weight cross-grasp aggregation reliability analysis; pressure-only."""
import argparse, itertools, json, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr

METHODS=("equal_grasp_fisher_mean","equal_grasp_fisher_median","frame_weighted_fisher_mean","pooled_frame_pearson")
WINDOWS={"future_after_5":"future","all_valid_non_future":"all"}

def corr(x):
    with np.errstate(invalid="ignore",divide="ignore"): c=np.corrcoef(x,rowvar=False)
    c=np.clip(c,-1,1); np.fill_diagonal(c,np.nan); return c

def fisher(mats,weights=None,median=False):
    z=np.arctanh(np.clip(np.stack(mats),-.999999,.999999))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        if median: a=np.nanmedian(z,axis=0)
        elif weights is None: a=np.nanmean(z,axis=0)
        else:
            w=np.asarray(weights)[:,None,None]; ok=np.isfinite(z)
            a=np.divide(np.nansum(z*w,axis=0),np.sum(ok*w,axis=0),out=np.full(z.shape[1:],np.nan),where=np.sum(ok*w,axis=0)>0)
    out=np.tanh(a); np.fill_diagonal(out,np.nan); return out

def metrics(a,b):
    tri=np.triu_indices(a.shape[0],1); x,y=a[tri],b[tri]; ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: sr=pr=mad=np.nan
    else: sr=float(spearmanr(x,y).statistic); pr=float(pearsonr(x,y).statistic); mad=float(np.mean(np.abs(x-y)))
    d={"spearman":sr,"pearson":pr,"mean_abs_diff":mad,"n_edges":len(x)}
    for t in (.5,.7,.9):
        aa=x>=t; bb=y>=t; u=int(np.sum(aa|bb)); d[f"jaccard_r_ge_{t:.1f}"]=float(np.sum(aa&bb)/u) if u else np.nan
    return d

def main():
    p=argparse.ArgumentParser(); p.add_argument("--metadata",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--bootstrap-reps",type=int,default=5000); p.add_argument("--bootstrap-seed",type=int,default=20261122)
    p.add_argument("--sensor-seed",type=int,default=20261116); p.add_argument("--nodes",type=int,default=96); p.add_argument("--archive-sha256")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False)
    raw=loadmat(a.metadata,simplify_cells=True); pressure=np.asarray(raw["pressure"])
    oid=np.asarray(raw["objectId"]).ravel().astype(int); ts=np.asarray(raw["ts"]).ravel().astype(float)
    gi=np.asarray(raw["graspIndex"]).ravel().astype(int); valid=np.asarray(raw["graspValid"]).ravel().astype(bool)
    names=np.asarray(raw["objects"]).ravel(); rid=np.asarray(raw["recordingId"]).ravel()
    assert pressure.shape==(len(rid),32,32) and pressure.dtype==np.int16 and len(set(oid)-{0})==26
    active=np.flatnonzero(np.nanstd(pressure.astype(np.float32),axis=0).ravel()>0); assert len(active)==548
    flat=active[np.sort(np.random.default_rng(a.sensor_seed).choice(548,a.nodes,replace=False))]
    groups={}; matrices={}
    for obj in sorted(set(oid)-{0}):
        for g in range(6):
            ix=np.flatnonzero((oid==obj)&(gi==g)&valid); ix=ix[np.argsort(ts[ix],kind="stable")]
            x=np.clip((pressure[ix].reshape(len(ix),-1)[:,flat].astype(float)-500)/150,0,1)
            assert len(x)>=6; groups[obj,g]={"all":x,"future":x[5:]}; matrices[obj,g]=corr(x[5:])
    rows=[]
    for obj in sorted(set(oid)-{0}):
        for wn,wk in WINDOWS.items():
            for si,A in enumerate(x for x in itertools.combinations(range(6),3) if 0 in x):
                B=tuple(j for j in range(6) if j not in A)
                for method in METHODS:
                    def estimate(ids):
                        xs=[groups[obj,j][wk] for j in ids]; ms=[corr(x) for x in xs]
                        if method=="equal_grasp_fisher_mean": return fisher(ms)
                        if method=="equal_grasp_fisher_median": return fisher(ms,median=True)
                        if method=="frame_weighted_fisher_mean": return fisher(ms,[len(x) for x in xs])
                        return corr(np.concatenate(xs))
                    rows.append({"object_id":int(obj),"object_name":str(names[obj]),"window_protocol":wn,
                                 "split_index":si,"aggregation":method,**metrics(estimate(A),estimate(B))})
    pd.DataFrame(rows).to_csv(a.output_dir/"partition_metrics.csv",index=False)
    loo=[]
    for obj in sorted(set(oid)-{0}):
        ms=[matrices[obj,j] for j in range(6)]; full=fisher(ms)
        for j in range(6): loo.append({"object_id":int(obj),"omitted_grasp":j,**metrics(full,fisher([m for k,m in enumerate(ms) if k!=j]))})
    pd.DataFrame(loo).to_csv(a.output_dir/"leave_one_grasp_out.csv",index=False)
    np.savez_compressed(a.output_dir/"grasp_level_correlations.npz",correlations=np.stack([matrices[o,j] for o in sorted(set(oid)-{0}) for j in range(6)]))
    (a.output_dir/"config.json").write_text(json.dumps({"archive_sha256":a.archive_sha256,"sensor_seed":a.sensor_seed,"nodes":a.nodes,
      "transform":"clip((pressure-500)/150,0,1)","grasp_indices":list(range(6)),"bootstrap_reps":a.bootstrap_reps,
      "bootstrap_seed":a.bootstrap_seed,"outer_unit":"physical object","model_training":False},indent=2))
    print(f"completed {len(rows)} partition comparisons and {len(loo)} leave-one-grasp-out comparisons")
if __name__=="__main__": main()
