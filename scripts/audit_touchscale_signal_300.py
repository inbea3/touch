#!/usr/bin/env python3
"""TouchScale 300-episode stratified raw tactile QC audit (no model fitting)."""
from __future__ import annotations
import hashlib, io, json, os, shutil, tarfile
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO="2077AIDataFoundation/TouchScale"
SHARD_CONFIG=Path("configs/touchscale_signal_audit_300_shards.txt")
OUT=Path("artifacts/touchscale_signal_300")
N_SAMPLE=300
SEED=20261125
THRESHOLDS=(1.0,3.0,5.0)

def json_default(value):
    if isinstance(value,(bytes,bytearray)):
        return value.decode("utf-8",errors="replace")
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,np.ndarray):
        return value.tolist()
    return str(value)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def rate_stats(ts,scale=1.0):
    ts=np.asarray(ts,dtype=np.float64).reshape(-1)*scale
    if len(ts)<2: return {"n":int(len(ts)),"monotonic":None}
    dt=np.diff(ts); pos=dt[dt>0]
    return {
      "n":int(len(ts)),"monotonic":bool(np.all(dt>0)),
      "median_rate_hz":float(1/np.median(pos)) if len(pos) else None,
      "effective_rate_hz":float((len(ts)-1)/(ts[-1]-ts[0])) if ts[-1]>ts[0] else None,
      "dt_cv":float(np.std(pos)/np.mean(pos)) if len(pos) and np.mean(pos)>0 else None,
      "max_gap_s":float(np.max(dt)) if len(dt) else None,
      "gaps_over_0_25s":int(np.sum(dt>.25)),
      "duplicate_or_reverse_steps":int(np.sum(dt<=0))
    }

def get_nested(obj,keys):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in keys: return v
        for v in obj.values():
            got=get_nested(v,keys)
            if got is not None: return got
    elif isinstance(obj,list):
        for v in obj:
            got=get_nested(v,keys)
            if got is not None: return got
    return None

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

def audit_hand(g,hand,epmeta):
    pads=sorted([k for k in g if k.startswith("pad_")],key=lambda k:int(k.split("_")[1]))
    if not pads: return {"status":"no_pad_arrays"}
    arrays=[np.asarray(g[k][()]) for k in pads]
    lengths=[a.shape[0] for a in arrays]
    if len(set(lengths))!=1: return {"status":"pad_length_mismatch"}
    T=lengths[0]
    X0=np.concatenate([a.reshape(T,-1).astype(np.float64) for a in arrays],axis=1)
    finite=np.isfinite(X0); X=np.where(finite,X0,0.0)
    force_sum=X.sum(axis=1)
    ts=np.asarray(g["t_epoch"][()]).reshape(-1) if "t_epoch" in g else np.array([])
    tstats=rate_stats(ts)
    meta_rate=get_nested(epmeta,{"glove_rate_hz"}) if epmeta else None
    meta_contact=get_nested(epmeta,{f"tactile_contact_frac_{hand}"}) if epmeta else None
    thresh={}
    for th in THRESHOLDS:
        active=force_sum>th; segs=segments(active)
        thresh[str(int(th))]={"active_fraction":float(active.mean()),
          "active_frames":int(active.sum()),"segment_count":len(segs),
          "longest_segment_frames":int(max(map(len,segs),default=0))}
    return {
      "status":"ok","frames":int(T),"pads":len(pads),"taxels":int(X.shape[1]),
      "finite_fraction":float(finite.mean()),
      "force_sum_min_N":float(force_sum.min()),"force_sum_median_N":float(np.median(force_sum)),
      "force_sum_p95_N":float(np.quantile(force_sum,.95)),"force_sum_max_N":float(force_sum.max()),
      "timestamp":tstats,
      "metadata_glove_rate_hz":float(meta_rate) if meta_rate is not None else None,
      "effective_rate_abs_error_hz":abs(tstats["effective_rate_hz"]-float(meta_rate)) if meta_rate is not None and tstats.get("effective_rate_hz") else None,
      "metadata_contact_fraction":float(meta_contact) if meta_contact is not None else None,
      "contact_fraction_3N":thresh["3"]["active_fraction"],
      "contact_fraction_abs_error":abs(thresh["3"]["active_fraction"]-float(meta_contact)) if meta_contact is not None else None,
      "threshold_sensitivity":thresh,
      "split_half_structure":split_half(X,force_sum>3.0)
    }

def choose_sample(meta,shards):
    df=meta[meta["shard"].isin(shards)].copy()
    if len(df)<N_SAMPLE: raise RuntimeError(f"Only {len(df)} candidate rows in selected shards")
    df["episode_id"]=df["episode_id"].astype(str)
    df["contact_avg"]=df[["tactile_contact_frac_left","tactile_contact_frac_right"]].mean(axis=1)
    scene_counts=df.groupby("scene").size().sort_values()
    # Give every represented scene one episode, then allocate remaining slots proportional to residual scene volume.
    quotas={s:1 for s in scene_counts.index}
    rem=N_SAMPLE-len(quotas)
    weights=(scene_counts-1).clip(lower=0)
    raw=rem*weights/weights.sum()
    floors=np.floor(raw).astype(int)
    for s in floors.index: quotas[s]+=int(floors[s])
    left=N_SAMPLE-sum(quotas.values())
    remainders=(raw-floors).sort_values(ascending=False)
    for s in remainders.index[:left]: quotas[s]+=1
    rng=np.random.default_rng(SEED)
    selected=[]; actor_counts=defaultdict(int); task_counts=defaultdict(int); date_counts=defaultdict(int)
    for scene,quota in quotas.items():
        pool=df[df.scene==scene].copy()
        pool["contact_bin"]=pd.qcut(pool["contact_avg"].rank(method="first"),3,labels=["low","mid","high"])
        bin_q=[quota//3]*3
        for i in range(quota%3): bin_q[i]+=1
        for b,q in zip(["low","mid","high"],bin_q):
            bp=pool[pool.contact_bin==b].copy()
            for _ in range(q):
                if bp.empty:
                    continue
                scores=[]
                for idx,row in bp.iterrows():
                    score=(2.0/(1+actor_counts[row.actor_id])+
                           1.5/(1+task_counts[row.task_name])+
                           .25/(1+date_counts[row.collection_date])+rng.random()*.05)
                    scores.append((score,idx))
                idx=max(scores)[1]; row=bp.loc[idx]
                selected.append(row)
                actor_counts[row.actor_id]+=1; task_counts[row.task_name]+=1; date_counts[row.collection_date]+=1
                bp=bp.drop(idx)
        # Fill any scene quota not met because a contact bin had too few candidates.
        chosen_scene=sum(1 for r in selected if r["scene"]==scene)
        remaining=quota-chosen_scene
        if remaining:
            used={r["episode_id"] for r in selected}
            rest=pool[~pool.episode_id.isin(used)]
            for _,row in rest.sample(n=remaining,random_state=SEED).iterrows():
                selected.append(row)
    out=pd.DataFrame(selected)
    if len(out)!=N_SAMPLE or out.episode_id.nunique()!=N_SAMPLE:
        raise RuntimeError(f"Sampling invariant failed: {len(out)} rows, {out.episode_id.nunique()} unique")
    return out,quotas

def main():
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/"hf_cache"
    shards=[x.strip() for x in SHARD_CONFIG.read_text().splitlines() if x.strip()]
    meta_path=hf_hub_download(repo_id=REPO,filename="metadata/episodes.parquet",repo_type="dataset",token=token,cache_dir=str(cache))
    meta=pd.read_parquet(meta_path)
    shutil.rmtree(cache,ignore_errors=True)
    sample,quotas=choose_sample(meta,shards)
    sample.to_csv(OUT/"sample_selection.csv",index=False)
    by_shard=defaultdict(list)
    for row in sample.to_dict("records"): by_shard[row["shard"]].append(row)
    details={}; flat=[]; hashes={}
    for shard,rows in sorted(by_shard.items()):
        local=hf_hub_download(repo_id=REPO,filename=shard,repo_type="dataset",token=token,cache_dir=str(cache))
        hashes[shard]=sha256(local)
        with tarfile.open(local,"r:*") as tar:
            members={Path(m.name).name:m for m in tar if m.isfile()}
            for row in rows:
                eid=row["episode_id"]; hm=members.get(eid+".episode.h5"); jm=members.get(eid+".json")
                if hm is None:
                    details[eid]={"status":"h5_missing","scene":row["scene"],"shard":shard}
                    continue
                epmeta={}
                if jm is not None:
                    try: epmeta=json.loads(tar.extractfile(jm).read().decode("utf-8"))
                    except Exception as e: epmeta={"json_parse_error":str(e)}
                with h5py.File(io.BytesIO(tar.extractfile(hm).read()),"r") as h:
                    hands={}
                    for hand in ("left","right"):
                        p=f"obs/tactile/raw/{hand}"
                        hands[hand]=audit_hand(h[p],hand,epmeta) if p in h else {"status":"missing_hand"}
                    timeline=rate_stats(np.asarray(h["time/timestamp_ns"][()]),scale=1e-9) if "time/timestamp_ns" in h else {}
                details[eid]={"status":"ok","scene":row["scene"],"task_name":row["task_name"],
                    "actor_id":row["actor_id"],"collection_date":row["collection_date"],
                    "episode_json_present":jm is not None,"episode_timeline":timeline,"hands":hands}
                for hand,v in hands.items():
                    if v.get("status")!="ok":
                        flat.append({"episode_id":eid,"scene":row["scene"],"hand":hand,"status":v.get("status")}); continue
                    flat.append({
                      "episode_id":eid,"scene":row["scene"],"task_name":row["task_name"],
                      "actor_id":row["actor_id"],"hand":hand,"status":"ok",
                      "frames":v["frames"],"pads":v["pads"],"taxels":v["taxels"],
                      "finite_fraction":v["finite_fraction"],
                      "timestamp_rate_hz":v["timestamp"].get("median_rate_hz"),
                      "timestamp_effective_rate_hz":v["timestamp"].get("effective_rate_hz"),
                      "timestamp_dt_cv":v["timestamp"].get("dt_cv"),
                      "timestamp_monotonic":v["timestamp"].get("monotonic"),
                      "timestamp_max_gap_s":v["timestamp"].get("max_gap_s"),
                      "timestamp_gaps_over_0_25s":v["timestamp"].get("gaps_over_0_25s"),
                      "metadata_rate_hz":v["metadata_glove_rate_hz"],
                      "effective_rate_abs_error_hz":v["effective_rate_abs_error_hz"],
                      "metadata_contact_fraction":v["metadata_contact_fraction"],
                      "contact_fraction_3N":v["contact_fraction_3N"],
                      "contact_fraction_abs_error":v["contact_fraction_abs_error"],
                      "contact_fraction_1N":v["threshold_sensitivity"]["1"]["active_fraction"],
                      "contact_fraction_5N":v["threshold_sensitivity"]["5"]["active_fraction"],
                      "active_segments_3N":v["threshold_sensitivity"]["3"]["segment_count"],
                      "longest_active_segment_3N":v["threshold_sensitivity"]["3"]["longest_segment_frames"],
                      "split_half_edge_corr":v["split_half_structure"].get("split_half_edge_corr")
                    })
        shutil.rmtree(cache,ignore_errors=True)
    (OUT/"detail.json").write_text(json.dumps(details,indent=2,default=json_default),encoding="utf-8")
    pd.DataFrame(flat).to_csv(OUT/"hand_summary.csv",index=False)
    (OUT/"shard_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps({"selected":len(sample),"found":sum(v.get("status")=="ok" for v in details.values()),
      "missing":sum(v.get("status")!="ok" for v in details.values()),"shards":len(by_shard),
      "actors":int(sample.actor_id.nunique()),"scenes":int(sample.scene.nunique()),
      "dates":int(sample.collection_date.nunique()),"scene_quotas":quotas,"hand_records":len(flat),
      "output":str(OUT)},indent=2,default=json_default))

if __name__=="__main__": main()
