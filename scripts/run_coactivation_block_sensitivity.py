#!/usr/bin/env python3
"""Pressure-only moving-block bootstrap sensitivity across block lengths."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

def corr_pearson(x):
    x=np.asarray(x,dtype=np.float64)
    sd=np.std(x,axis=0,ddof=1);good=np.isfinite(sd)&(sd>1e-12)
    out=np.full((x.shape[1],x.shape[1]),np.nan,dtype=np.float32)
    if good.sum()<2:return out
    z=x[:,good]-np.mean(x[:,good],axis=0,keepdims=True)
    z/=np.sqrt(np.sum(z*z,axis=0,keepdims=True))
    c=np.clip(z.T@z,-1.,1.);ix=np.flatnonzero(good)
    out[np.ix_(ix,ix)]=c.astype(np.float32);np.fill_diagonal(out,np.nan)
    return out

def block_indices(n,block_len,rng):
    starts=rng.integers(0,n,size=int(np.ceil(n/block_len)))
    return np.concatenate([((s+np.arange(block_len))%n) for s in starts])[:n]

def compare(ref,boot):
    tri=np.triu_indices(ref.shape[0],k=1);a,b=ref[tri],boot[tri]
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3:return None
    a,b=a[ok],b[ok]
    rho=float(spearmanr(a,b).statistic);mad=float(np.mean(np.abs(a-b)))
    aa,bb=a>=.7,b>=.7;union=int(np.sum(aa|bb))
    jac=float(np.sum(aa&bb)/union) if union else 1.
    return mad,rho,jac

def describe(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return {"n":0}
    return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "sd":float(a.std(ddof=1)) if len(a)>1 else 0.,
            "q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def lag_autocorrelation(x,lags=(1,5,10,20,30,40,60,80)):
    z=np.asarray(x,dtype=np.float64).copy();z-=np.mean(z,axis=0,keepdims=True);rows=[]
    for lag in lags:
        if len(z)<=lag:continue
        a,b=z[:-lag],z[lag:];den=np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))
        ac=np.divide(np.sum(a*b,axis=0),den,out=np.full(z.shape[1],np.nan),where=den>0)
        rows.append({"lag_frames":int(lag),"mean_signed_acf":float(np.nanmean(ac)),
          "mean_abs_acf":float(np.nanmean(np.abs(ac))),"median_abs_acf":float(np.nanmedian(np.abs(ac)))})
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--block-lengths",type=int,nargs="+",default=[5,10,20])
    ap.add_argument("--bootstrap-reps",type=int,default=200)
    ap.add_argument("--bootstrap-nodes",type=int,default=96)
    ap.add_argument("--k-early",type=int,default=5)
    ap.add_argument("--seed",type=int,default=20261116)
    a=ap.parse_args()
    if a.bootstrap_reps<200:raise ValueError("本阶段正式敏感性分析要求至少 200 次 Bootstrap")
    if sorted(set(a.block_lengths))!=[5,10,20]:raise ValueError("本轮块长必须恰好为 5、10、20 帧")
    m=loadmat(a.metadata,simplify_cells=True);pressure=np.asarray(m["pressure"])
    rec=np.asarray(m["recordingId"]).reshape(-1).astype(np.int64)
    obj=np.asarray(m["objectId"]).reshape(-1).astype(np.int64)
    valid=np.asarray(m["hasValidLabel"]).reshape(-1).astype(bool)
    if any(len(v)!=len(pressure) for v in (rec,obj,valid)):raise ValueError("pressure 与元数据长度不一致")
    sensor_mask=np.nanstd(pressure.astype(np.float32),axis=0)>0
    if int(sensor_mask.sum())!=548:raise ValueError(f"预期 548 个有效传感器，实际 {int(sensor_mask.sum())}")
    node_rng=np.random.default_rng(a.seed)
    sensor_ids=np.sort(node_rng.choice(548,size=min(a.bootstrap_nodes,548),replace=False))
    rows=[];acf_rows=[];started=time.time()
    recording_ids=[int(r) for r in np.unique(rec)
      if np.unique(obj[rec==r]).size==1 and int(obj[rec==r][0])!=0]
    print(f"Loaded pressure-only metadata: {len(recording_ids)} recordings; {int(sensor_mask.sum())} sensors; {len(sensor_ids)} fixed bootstrap nodes",flush=True)
    for idx,rid in enumerate(recording_ids,1):
        ids=np.flatnonzero(rec==rid);cls=int(obj[ids[0]])
        x=np.clip((pressure[ids][:,sensor_mask].astype(np.float32)-500.)/150.,0.,1.)
        v=np.flatnonzero(valid[ids])
        if len(v)<=a.k_early+3:continue
        boundary=int(v[a.k_early-1]);future=v[v>boundary]
        xf=np.asarray(x[future][:,sensor_ids],dtype=np.float64)
        if len(xf)<3:continue
        reference=corr_pearson(xf)
        for acf in lag_autocorrelation(xf):
            acf_rows.append({"recording_id":rid,"object_id":cls,**acf})
        for block_len in a.block_lengths:
            rng=np.random.default_rng(a.seed+rid)
            vals={"mad":[],"rho":[],"jaccard":[]};checkpoints={50:{},100:{},200:{}}
            for rep in range(1,a.bootstrap_reps+1):
                sample=block_indices(len(xf),block_len,rng)
                mboot=corr_pearson(xf[sample]);metric=compare(reference,mboot)
                if metric is not None:
                    vals["mad"].append(metric[0]);vals["rho"].append(metric[1]);vals["jaccard"].append(metric[2])
                if rep in checkpoints:
                    for key,vv in vals.items():checkpoints[rep][key]=float(np.median(vv)) if vv else None
            row={"recording_id":rid,"object_id":cls,"block_length_frames":block_len,
                 "n_future_frames":int(len(xf)),"n_replicates":len(vals["mad"]),
                 "node_subsample_n":int(len(sensor_ids))}
            for key,vv in vals.items():
                row[f"{key}_median"]=float(np.median(vv)) if vv else None
                row[f"{key}_q25"]=float(np.quantile(vv,.25)) if vv else None
                row[f"{key}_q75"]=float(np.quantile(vv,.75)) if vv else None
                for cp in (50,100,200):row[f"{key}_median_first_{cp}"]=checkpoints[cp].get(key)
            rows.append(row)
        if idx%10==0 or idx==len(recording_ids):
            print(f"processed {idx}/{len(recording_ids)} recordings; elapsed={time.time()-started:.1f}s",flush=True)

    metrics=["mad_median","rho_median","jaccard_median"]
    recording_summary={};object_values={};object_summary={};convergence={}
    for block_len in a.block_lengths:
        group=[r for r in rows if r["block_length_frames"]==block_len]
        recording_summary[str(block_len)]={m:describe([r[m] for r in group if r[m] is not None]) for m in metrics}
        object_values[block_len]={}
        for oid in sorted({r["object_id"] for r in group}):
            rr=[r for r in group if r["object_id"]==oid]
            object_values[block_len][oid]={m:float(np.median([r[m] for r in rr if r[m] is not None])) for m in metrics}
        object_summary[str(block_len)]={m:describe([v[m] for v in object_values[block_len].values()]) for m in metrics}
        convergence[str(block_len)]={}
        for cp in (50,100,200):
            convergence[str(block_len)][str(cp)]={}
            for metric in ("mad","rho","jaccard"):
                by_obj={}
                for row in group:
                    val=row[f"{metric}_median_first_{cp}"]
                    if val is not None:by_obj.setdefault(row["object_id"],[]).append(val)
                objvals=[float(np.median(v)) for v in by_obj.values()]
                convergence[str(block_len)][str(cp)][f"{metric}_median"]=describe(objvals)

    acf_summary={}
    for lag in sorted({r["lag_frames"] for r in acf_rows}):
        gg=[r for r in acf_rows if r["lag_frames"]==lag]
        acf_summary[str(lag)]={metric:describe([r[metric] for r in gg])
          for metric in ("mean_signed_acf","mean_abs_acf","median_abs_acf")}
    deltas={}
    for block_len in (5,20):
        deltas[f"{block_len}_minus_10"]={}
        for metric in metrics:
            diffs=[object_values[block_len][oid][metric]-object_values[10][oid][metric]
              for oid in object_values[block_len] if oid in object_values[10]]
            deltas[f"{block_len}_minus_10"][metric]=describe(diffs)

    result={"scope":"STAG pressure-only; no synchronized video, image annotation, or multimodal data",
      "n_recordings":len({r["recording_id"] for r in rows}),"n_objects":len({r["object_id"] for r in rows}),
      "n_sensors":int(sensor_mask.sum()),"n_bootstrap_nodes":int(len(sensor_ids)),
      "n_undirected_edges_in_subsample":int(len(sensor_ids)*(len(sensor_ids)-1)//2),
      "settings":{"block_lengths_frames":a.block_lengths,"bootstrap_reps":a.bootstrap_reps,"k_early":a.k_early,
        "seed":a.seed,"sensor_node_sampling":"one fixed node sample shared by all block lengths and recordings",
        "recording_rng":"default_rng(seed + recording_id), preserving previous block=10 seed convention",
        "primary_summary_unit":"object; each object aggregates its three Recording medians",
        "edge_dependence":"sensor edges within a recording are dependent and are not independent replicates"},
      "recording_level_summary":recording_summary,"object_level_summary":object_summary,
      "object_level_paired_differences":deltas,"temporal_autocorrelation_recording_level_summary":acf_summary,
      "monte_carlo_convergence_first_50_100_200_object_level":convergence,"recordings":rows}
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"coactivation_block_sensitivity_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for filename,data in [("coactivation_block_sensitivity_recording.csv",rows),("coactivation_temporal_autocorrelation.csv",acf_rows)]:
        with (out/filename).open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    object_rows=[{"object_id":oid,"block_length_frames":bl,**vals}
      for bl in a.block_lengths for oid,vals in object_values[bl].items()]
    with (out/"coactivation_block_sensitivity_object.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(object_rows[0]));w.writeheader();w.writerows(object_rows)
    print(json.dumps({"object_level_summary":object_summary,"object_level_paired_differences":deltas,
      "temporal_autocorrelation_recording_level_summary":acf_summary,
      "monte_carlo_convergence_first_50_100_200_object_level":convergence},ensure_ascii=False,indent=2),flush=True)
    print(f"Saved to {out}; total elapsed={time.time()-started:.1f}s",flush=True)

if __name__=="__main__":main()
