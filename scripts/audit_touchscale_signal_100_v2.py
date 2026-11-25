#!/usr/bin/env python3
"""TouchScale 100-episode signal audit v2: metadata cross-check and timestamp QC."""
from __future__ import annotations
import hashlib, io, json, os, tarfile
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO="2077AIDataFoundation/TouchScale"
CONFIG=Path("configs/touchscale_signal_sample_100.tsv")
OUT=Path("artifacts/touchscale_signal_100_v2")
THRESHOLDS=(1.0,3.0,5.0)

def load_selection():
    rows=[]
    for line in CONFIG.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            eid,shard,family=line.split("|")
            rows.append({"episode_id":eid,"shard":shard,"family":family})
    if len(rows)!=100 or len({r["episode_id"] for r in rows})!=100:
        raise ValueError("Selection must contain 100 unique episodes")
    return rows

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def get_nested(obj, candidates):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in candidates: return v
        for v in obj.values():
            got=get_nested(v,candidates)
            if got is not None: return got
    elif isinstance(obj,list):
        for v in obj:
            got=get_nested(v,candidates)
            if got is not None: return got
    return None

def rate_stats(ts, scale=1.0):
    ts=np.asarray(ts,dtype=np.float64).reshape(-1)*scale
    if len(ts)<2: return {"n":len(ts),"monotonic":None}
    dt=np.diff(ts)
    positive=dt[dt>0]
    return {
        "n":int(len(ts)),"monotonic":bool(np.all(dt>0)),
        "median_rate_hz":float(1/np.median(positive)) if len(positive) else None,
        "p05_rate_hz":float(1/np.quantile(positive,.95)) if len(positive) else None,
        "p95_rate_hz":float(1/np.quantile(positive,.05)) if len(positive) else None,
        "max_gap_s":float(np.max(dt)) if len(dt) else None,
        "gaps_over_0_25s":int(np.sum(dt>.25)),
        "duplicate_or_reverse_steps":int(np.sum(dt<=0))
    }

def segments(mask):
    ix=np.flatnonzero(mask)
    if not len(ix): return []
    return [g for g in np.split(ix,np.where(np.diff(ix)>1)[0]+1) if len(g)]

def split_half(X,active):
    ix=np.flatnonzero(active)
    if len(ix)<40: return {"status":"insufficient_active_frames","active_frames":int(len(ix))}
    Aidx,Bidx=ix[:len(ix)//2],ix[len(ix)//2:]
    valid=(X[Aidx].std(axis=0)>1e-8)&(X[Bidx].std(axis=0)>1e-8)
    if valid.sum()<10: return {"status":"too_few_variable_taxels","variable_taxels":int(valid.sum())}
    A=X[Aidx][:,valid]; B=X[Bidx][:,valid]
    c1=np.corrcoef(A,rowvar=False); c2=np.corrcoef(B,rowvar=False)
    iu=np.triu_indices(int(valid.sum()),1); a,b=c1[iu],c2[iu]
    ok=np.isfinite(a)&np.isfinite(b)
    corr=float(np.corrcoef(a[ok],b[ok])[0,1]) if ok.sum()>=20 and a[ok].std()>0 and b[ok].std()>0 else None
    return {"status":"computed" if corr is not None else "undefined",
            "active_frames":int(len(ix)),"variable_taxels":int(valid.sum()),
            "valid_edges":int(ok.sum()),"split_half_edge_corr":corr}

def audit_hand(g, hand, epmeta):
    pads=sorted([k for k in g if k.startswith("pad_")],key=lambda k:int(k.split("_")[1]))
    if len(pads)!=15: pad_count=len(pads)
    if not pads: return {"status":"no_pad_arrays"}
    arrays=[np.asarray(g[k][()]) for k in pads]
    lengths=[a.shape[0] for a in arrays]
    if len(set(lengths))!=1: return {"status":"pad_length_mismatch","pad_shapes":{k:list(g[k].shape) for k in pads}}
    T=lengths[0]
    X0=np.concatenate([a.reshape(T,-1).astype(np.float64) for a in arrays],axis=1)
    finite=np.isfinite(X0); X=np.where(finite,X0,0.0)
    force_sum=X.sum(axis=1)
    ts=np.asarray(g["t_epoch"][()]).reshape(-1) if "t_epoch" in g else np.array([])
    # Dataset card documents per-hand t_epoch in Unix seconds.
    tstats=rate_stats(ts)
    meta_rate=get_nested(epmeta,{"glove_rate_hz"}) if epmeta else None
    meta_contact=get_nested(epmeta,{f"tactile_contact_frac_{hand}"}) if epmeta else None
    threshold={}
    for th in THRESHOLDS:
        active=force_sum>th
        segs=segments(active)
        threshold[str(int(th))]={
            "active_fraction":float(active.mean()),
            "active_frames":int(active.sum()),
            "segment_count":len(segs),
            "longest_segment_frames":int(max(map(len,segs),default=0))
        }
    active=force_sum>3.0
    s=split_half(X,active)
    return {
        "status":"ok","frames":int(T),"pads":len(pads),"taxels":int(X.shape[1]),
        "finite_fraction":float(finite.mean()),
        "force_sum_min_N":float(force_sum.min()),"force_sum_median_N":float(np.median(force_sum)),
        "force_sum_p95_N":float(np.quantile(force_sum,.95)),"force_sum_max_N":float(force_sum.max()),
        "timestamp":tstats,
        "metadata_glove_rate_hz":float(meta_rate) if meta_rate is not None else None,
        "rate_abs_error_hz":abs(tstats["median_rate_hz"]-float(meta_rate)) if meta_rate is not None and tstats.get("median_rate_hz") else None,
        "metadata_contact_fraction":float(meta_contact) if meta_contact is not None else None,
        "contact_fraction_at_3N":threshold["3"]["active_fraction"],
        "contact_fraction_abs_error":abs(threshold["3"]["active_fraction"]-float(meta_contact)) if meta_contact is not None else None,
        "threshold_sensitivity":threshold,
        "split_half_structure":s
    }

def main():
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True,exist_ok=True)
    selected=load_selection(); by_shard=defaultdict(list)
    for r in selected: by_shard[r["shard"]].append(r)
    detail={}; flat=[]; hashes={}; missing=[]
    for shard, rows in sorted(by_shard.items()):
        local=hf_hub_download(repo_id=REPO,filename=shard,repo_type="dataset",token=token)
        hashes[shard]=sha256(local)
        with tarfile.open(local,"r:*") as tar:
            members={Path(m.name).name:m for m in tar if m.isfile()}
            for r in rows:
                eid=r["episode_id"]; h5name=eid+".episode.h5"; jname=eid+".json"
                h5member=members.get(h5name); jmember=members.get(jname)
                if h5member is None:
                    # Save near matches to diagnose naming or shard assignment.
                    matches=[n for n in members if n.startswith(eid)]
                    detail[eid]={**r,"status":"h5_missing","near_match_members":matches[:10]}
                    missing.append(eid); continue
                epmeta={}
                if jmember is not None:
                    f=tar.extractfile(jmember)
                    try: epmeta=json.loads(f.read().decode("utf-8"))
                    except Exception as e: epmeta={"json_parse_error":str(e)}
                f=tar.extractfile(h5member)
                with h5py.File(io.BytesIO(f.read()),"r") as h:
                    hands={}
                    for hand in ("left","right"):
                        p=f"obs/tactile/raw/{hand}"
                        hands[hand]=audit_hand(h[p],hand,epmeta) if p in h else {"status":"missing_hand"}
                    timeline={}
                    if "time/timestamp_ns" in h:
                        timeline=rate_stats(np.asarray(h["time/timestamp_ns"][()]),scale=1e-9)
                    task={}
                    if "task" in h:
                        for k in ("name","task_id","success"):
                            if k in h["task"]:
                                v=np.asarray(h["task"][k][()])
                                task[k]=v.item() if v.ndim==0 else v.tolist()
                detail[eid]={**r,"status":"ok","episode_json_present":jmember is not None,
                             "metadata_keys":sorted(epmeta.keys()) if isinstance(epmeta,dict) else [],
                             "task":task,"episode_timeline":timeline,"hands":hands}
                for hand, v in hands.items():
                    if v.get("status")!="ok":
                        flat.append({"episode_id":eid,"family":r["family"],"hand":hand,"status":v.get("status")}); continue
                    flat.append({
                        "episode_id":eid,"family":r["family"],"hand":hand,"status":"ok",
                        "frames":v["frames"],"pads":v["pads"],"taxels":v["taxels"],
                        "finite_fraction":v["finite_fraction"],
                        "timestamp_rate_hz":v["timestamp"].get("median_rate_hz"),
                        "timestamp_monotonic":v["timestamp"].get("monotonic"),
                        "timestamp_max_gap_s":v["timestamp"].get("max_gap_s"),
                        "timestamp_gaps_over_0_25s":v["timestamp"].get("gaps_over_0_25s"),
                        "metadata_rate_hz":v["metadata_glove_rate_hz"],
                        "rate_abs_error_hz":v["rate_abs_error_hz"],
                        "metadata_contact_fraction":v["metadata_contact_fraction"],
                        "contact_fraction_3N":v["contact_fraction_at_3N"],
                        "contact_fraction_abs_error":v["contact_fraction_abs_error"],
                        "contact_fraction_1N":v["threshold_sensitivity"]["1"]["active_fraction"],
                        "contact_fraction_5N":v["threshold_sensitivity"]["5"]["active_fraction"],
                        "active_segments_3N":v["threshold_sensitivity"]["3"]["segment_count"],
                        "longest_active_segment_3N":v["threshold_sensitivity"]["3"]["longest_segment_frames"],
                        "split_half_edge_corr":v["split_half_structure"].get("split_half_edge_corr")
                    })
    (OUT/"detail.json").write_text(json.dumps(detail,indent=2),encoding="utf-8")
    pd.DataFrame(flat).to_csv(OUT/"hand_summary.csv",index=False)
    (OUT/"shard_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps({"requested":len(selected),"h5_found":sum(v.get("status")=="ok" for v in detail.values()),
                      "missing":missing,"hand_records":len(flat),"metadata_json_count":sum(v.get("episode_json_present",False) for v in detail.values()),
                      "output":str(OUT)},indent=2))

if __name__=="__main__": main()
