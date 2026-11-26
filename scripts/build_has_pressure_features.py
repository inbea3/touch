#!/usr/bin/env python3
"""Build episode-level pressure-only features and future targets for HAS.

Reads only obs_skin_force_left, timestamp, episode_index, classification_label.
Raw frame sequences are processed online and are never persisted.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET = "TUM-ICS/Hide-and-Seek"
SOURCE_SPLITS = ("train", "validation", "test")
N_CHANNELS = 44
EARLY_FRAMES = 5
MIN_FRAMES = 10
THRESHOLD = 0.7


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",default=None)
    p.add_argument("--split-manifest",default="results/has_object_grouped_split/episode_split_manifest.csv")
    p.add_argument("--object-manifest",default="results/has_object_grouped_split/object_split_manifest.csv")
    p.add_argument("--out",default="results/has_feature_dataset")
    a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    resolved_revision=a.revision or HfApi().dataset_info(DATASET).sha
    log_path=out/"run.log"
    def log(msg):
        line=f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line,flush=True)
        with log_path.open("a",encoding="utf-8") as f: f.write(line+"\n")

    # Load frozen metadata manifests; verify object-disjointness before reading sensor data.
    split_rows={}
    with open(a.split_manifest,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k=(r["official_split"],int(r["episode_index"]))
            if k in split_rows: raise RuntimeError(f"Duplicate episode key in split manifest: {k}")
            split_rows[k]=r
    object_rows=list(csv.DictReader(open(a.object_manifest,newline="",encoding="utf-8")))
    obj_sets={s:set() for s in ("train","validation","test")}
    for r in object_rows: obj_sets[r["grouped_split"]].add(r["object_group"])
    if any(obj_sets[x]&obj_sets[y] for x,y in (("train","validation"),("train","test"),("validation","test"))):
        raise RuntimeError("Object leakage detected in frozen object manifest")
    if {s:len(obj_sets[s]) for s in obj_sets}!={"train":21,"validation":6,"test":6}:
        raise RuntimeError(f"Unexpected object counts: { {s:len(v) for s,v in obj_sets.items()} }")
    if len(split_rows)!=4192: raise RuntimeError(f"Expected 4192 object-task episodes, got {len(split_rows)}")

    mapping=json.loads(Path("data/has_label_mapping.json").read_text(encoding="utf-8"))
    id_to_label={int(v):k for k,v in mapping["classification_labels"].items()}
    feature_rows=[]; future_mean_rows=[]; future_std_rows=[]; corr_rows=[]; edge_rows=[]
    index_rows=[]; diag_acc=defaultdict(lambda:{"frames":0,"zero":np.zeros(N_CHANNELS,dtype=np.int64),
        "sum":np.zeros(N_CHANNELS),"sum2":np.zeros(N_CHANNELS),"n":0,"episodes":0,
        "positive_edges":0,"total_edges":0,"constant_channels":0})
    seen=set(); rejected=[]; active_key=None; state=None; total_frames=0

    def init_state(key, split_record):
        return {"key":key,"meta":split_record,"n":0,"first_x":[],"first_t":[],
            "last_t":None,"future_n":0,"future_sum":np.zeros(N_CHANNELS),
            "future_sum2":np.zeros(N_CHANNELS),"future_cross":np.zeros((N_CHANNELS,N_CHANNELS)),
            "future_zero":np.zeros(N_CHANNELS,dtype=np.int64)}

    def finish(st):
        key=st["key"]; n=st["n"]; rec=st["meta"]
        if n<MIN_FRAMES:
            rejected.append({"official_split":key[0],"episode_index":key[1],"reason":f"too_few_frames:{n}"})
            return
        x=np.asarray(st["first_x"],dtype=np.float64); t=np.asarray(st["first_t"],dtype=np.float64)
        # Actual-time least-squares slope, in force units per second.
        tc=t-t.mean(); denom=float(np.dot(tc,tc))
        slope=((tc[:,None]*(x-x.mean(axis=0))).sum(axis=0)/denom) if denom>0 else np.zeros(N_CHANNELS)
        feat=np.concatenate([x.mean(axis=0),x.std(axis=0,ddof=0),slope,x[-1]],axis=0)
        nf=st["future_n"]
        if nf<5: rejected.append({"official_split":key[0],"episode_index":key[1],"reason":f"too_few_future_frames:{nf}"}); return
        mu=st["future_sum"]/nf
        var=np.maximum(st["future_sum2"]/nf-mu*mu,0.0)
        sd=np.sqrt(var)
        cov=st["future_cross"]/nf-np.outer(mu,mu)
        d=np.sqrt(np.maximum(np.diag(cov),0.0))
        denom_corr=np.outer(d,d)
        corr=np.divide(cov,denom_corr,out=np.zeros_like(cov),where=denom_corr>1e-15)
        corr=np.clip(corr,-1.0,1.0)
        corr=(corr+corr.T)/2
        np.fill_diagonal(corr,1.0)
        iu=np.triu_indices(N_CHANNELS,k=1)
        cvec=corr[iu].astype(np.float32)
        edges=(cvec>=THRESHOLD).astype(np.uint8)
        split=rec["grouped_split"]; obj=rec["object_group"]
        label=id_to_label.get(int(rec["label_id"]),rec["label_name"])
        early_duration=float(t[-1]-t[0]) if len(t)>1 else 0.0
        future_duration=float(st["last_t"]-st["first_t"][-1]) if st["last_t"] is not None else 0.0
        row_idx=len(feature_rows)
        feature_rows.append(feat.astype(np.float32))
        future_mean_rows.append(mu.astype(np.float32)); future_std_rows.append(sd.astype(np.float32))
        corr_rows.append(cvec); edge_rows.append(edges)
        index_rows.append({"array_row":row_idx,"grouped_split":split,"object_group":obj,
            "official_split":key[0],"episode_index":key[1],"label_id":rec["label_id"],
            "label_name":label,"n_frames":n,"future_frames":nf,
            "early_duration_seconds":early_duration,"future_duration_seconds":future_duration,
            "future_constant_channels":int(np.sum(var<=1e-15)),
            "future_finite_targets":int(np.isfinite(mu).sum()+np.isfinite(sd).sum()+np.isfinite(cvec).sum()),
            "positive_edge_fraction":float(edges.mean())})
        da=diag_acc[(split)]
        da["frames"]+=nf; da["zero"]+=st["future_zero"]; da["sum"]+=st["future_sum"]
        da["sum2"]+=st["future_sum2"]; da["n"]+=nf; da["episodes"]+=1
        da["positive_edges"]+=int(edges.sum()); da["total_edges"]+=len(edges)
        da["constant_channels"]+=int(np.sum(var<=1e-15))

    (out/"config.json").write_text(json.dumps({"dataset":DATASET,"resolved_revision":resolved_revision,"split_seed":20261123,"early_frames":EARLY_FRAMES,"minimum_episode_frames":MIN_FRAMES,"feature_definition":["mean","population_std","timestamp_slope","last"],"future_targets":["mean","population_std","pearson_upper_triangle","pearson_ge_0.7"],"allowed_fields":["obs_skin_force_left","timestamp","episode_index","classification_label"]},indent=2),encoding="utf-8")
    log(f"START dataset={DATASET} revision={resolved_revision} episodes={len(split_rows)}")
    for source_split in SOURCE_SPLITS:
        kw={"split":source_split,"streaming":True}
        kw["revision"]=resolved_revision
        ds=load_dataset(DATASET,**kw)
        log(f"LOADED source_split={source_split}")
        active_key=None; state=None; closed=set()
        for row_i,row in enumerate(ds):
            key=(source_split,int(row["episode_index"]))
            if key not in split_rows: continue
            if key!=active_key:
                if state is not None:
                    finish(state); closed.add(active_key)
                if key in closed: raise RuntimeError(f"Episode rows are not contiguous in source stream: {key}")
                active_key=key; state=init_state(key,split_rows[key])
            x=np.asarray(row["obs_skin_force_left"],dtype=np.float64)
            ts=float(row["timestamp"])
            if x.shape!=(N_CHANNELS,) or not np.isfinite(x).all(): raise RuntimeError(f"Invalid force vector at {key}")
            if state["last_t"] is not None and ts<=state["last_t"]:
                raise RuntimeError(f"Non-increasing timestamp at {key}")
            state["n"]+=1; total_frames+=1; state["last_t"]=ts
            if state["n"]<=EARLY_FRAMES:
                state["first_x"].append(x.copy()); state["first_t"].append(ts)
            else:
                state["future_n"]+=1; state["future_sum"]+=x
                state["future_sum2"]+=x*x; state["future_cross"]+=np.outer(x,x)
                state["future_zero"]+=(x==0)
            if row_i and row_i%100000==0: log(f"PROGRESS source_split={source_split} rows={row_i} frames={total_frames}")
        if state is not None: finish(state)
        log(f"FINISHED source_split={source_split} frames_seen={total_frames} feature_episodes={len(feature_rows)}")

    # Ensure all manifest episodes were observed and no episode was silently lost.
    processed={(r["official_split"],int(r["episode_index"])) for r in index_rows}
    missing=set(split_rows)-processed
    if missing: raise RuntimeError(f"Missing manifest episodes: {len(missing)}; examples={list(missing)[:10]}")
    X=np.stack(feature_rows); Ymean=np.stack(future_mean_rows); Ystd=np.stack(future_std_rows)
    R=np.stack(corr_rows); E=np.stack(edge_rows)
    np.savez_compressed(out/"episode_features_targets.npz",X_early=X,Y_future_mean=Ymean,
        Y_future_std=Ystd,Y_future_corr_upper=R,Y_future_edge_07=E,
        corr_i=np.triu_indices(N_CHANNELS,k=1)[0],corr_j=np.triu_indices(N_CHANNELS,k=1)[1])
    with (out/"episode_index.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(index_rows[0])); w.writeheader(); w.writerows(index_rows)
    diag=[]
    for split,da in diag_acc.items():
        n=max(da["n"],1); mu=da["sum"]/n; var=np.maximum(da["sum2"]/n-mu*mu,0)
        for ch in range(N_CHANNELS):
            diag.append({"grouped_split":split,"channel":ch,"future_frames":da["n"],
                "zero_fraction":float(da["zero"][ch]/n),"mean":float(mu[ch]),
                "std":float(np.sqrt(var[ch])),"constant_episode_channel_count":da["constant_channels"],
                "episodes":da["episodes"],"positive_edge_fraction":da["positive_edges"]/max(da["total_edges"],1)})
    with (out/"channel_diagnostics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(diag[0])); w.writeheader(); w.writerows(diag)
    summary={"dataset":DATASET,"revision":resolved_revision,"revision_status":"pinned_resolved_at_run_start",
        "split_seed":20261123,"early_frames":EARLY_FRAMES,"min_episode_frames":MIN_FRAMES,
        "feature_shape":list(X.shape),"future_mean_shape":list(Ymean.shape),"future_std_shape":list(Ystd.shape),
        "future_corr_shape":list(R.shape),"edge_shape":list(E.shape),"source_frames_processed":total_frames,
        "episodes_by_split":{s:sum(r["grouped_split"]==s for r in index_rows) for s in ("train","validation","test")},
        "objects_by_split":{s:len(obj_sets[s]) for s in obj_sets},
        "rejected_episodes":rejected,"positive_edge_fraction_by_split":{s:float(np.mean([r["positive_edge_fraction"] for r in index_rows if r["grouped_split"]==s])) for s in ("train","validation","test")},
        "constant_future_channel_episode_pairs":{s:sum(r["future_constant_channels"] for r in index_rows if r["grouped_split"]==s) for s in ("train","validation","test")},
        "elapsed_seconds":time.time()-started,"interpretation":"Derived episode-level features and future-window targets only; no model training."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"summary.md").write_text("# HAS feature dataset summary\n\n"+json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n",encoding="utf-8")
    log(f"DONE episodes={len(index_rows)} frames={total_frames} elapsed={summary['elapsed_seconds']:.1f}s")
    # Finalize run.log before hashing all outputs.
    files=sorted(x for x in out.iterdir() if x.is_file() and x.name!="manifest.sha256")
    (out/"manifest.sha256").write_text("\n".join(f"{sha256(x)}  {x.name}" for x in files)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
