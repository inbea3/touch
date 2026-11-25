#!/usr/bin/env python3
"""Seven-episode TouchScale HDF5 tactile feasibility audit; no video extraction or model fitting."""
from __future__ import annotations
import hashlib, json, os, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from huggingface_hub import hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
SHARD = "data/train-00174-of-00430.tar"
SELECTED = {
 "390106b7-80c9-4c0d-8f4b-f1237fe67967": {"family":"single_object_manipulation","task":"Pull the wooden dowel out of the pre-drilled hole","rate":59.8,"left":0.6265,"right":0.0},
 "33a7eb90-76b5-4045-84e3-8828416ab293": {"family":"scene_setup","task":"Set up the test-tube-and-rod comparison display","rate":35.0,"left":0.0158,"right":0.2176},
 "30e2c2bd-47bc-4705-bb25-3da5d20a34d0": {"family":"component_assembly","task":"Unscrew the glass reagent-bottle cap and place it on the tray","rate":35.0,"left":0.3895,"right":0.7314},
 "403923fe-7a7f-42cb-9a76-c8061bf9b457": {"family":"material_deformation","task":"Wring out a wet cloth once with both hands","rate":92.3,"left":0.4771,"right":0.4164},
 "b9500cfb-a22f-4de3-b51c-4aa7290e9715": {"family":"inspection_measurement","task":"Use a tape measure to measure the short-side length of the board","rate":92.2,"left":0.3696,"right":0.7400},
 "0b512291-260c-41b2-bdd6-a1c023d33c3b": {"family":"surface_cleaning","task":"Brush a wide-mouth plastic test tube's inner wall with the soft wide-head tube brush","rate":35.0,"left":0.2989,"right":0.6295},
 "93a5dd96-6e5b-475a-83f1-78b4fd3cc136": {"family":"transfer_pouring","task":"Use tweezers to hold a test strip and dip it into the liquid (simulated pH-paper test)","rate":35.0,"left":0.5714,"right":0.0},
}

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def segments(mask):
    idx=np.flatnonzero(mask)
    if not len(idx): return []
    groups=np.split(idx,np.where(np.diff(idx)>1)[0]+1)
    return [{"start_frame":int(g[0]),"end_frame":int(g[-1]),"frames":int(len(g))} for g in groups]

def split_half(X,active):
    ix=np.flatnonzero(active)
    if len(ix)<40: return {"status":"insufficient_active_frames","active_frames":int(len(ix))}
    mid=len(ix)//2
    A=X[ix[:mid]].astype(np.float32); B=X[ix[mid:]].astype(np.float32)
    valid=(A.std(axis=0)>1e-8)&(B.std(axis=0)>1e-8)
    n=int(valid.sum())
    if n<10: return {"status":"too_few_variable_taxels","active_frames":int(len(ix)),"variable_taxels":n}
    A=A[:,valid]; B=B[:,valid]
    C1=np.corrcoef(A,rowvar=False); C2=np.corrcoef(B,rowvar=False)
    iu=np.triu_indices(n,k=1); x=C1[iu]; y=C2[iu]
    ok=np.isfinite(x)&np.isfinite(y)
    corr=float(np.corrcoef(x[ok],y[ok])[0,1]) if ok.sum()>=20 and np.std(x[ok])>0 and np.std(y[ok])>0 else None
    return {"status":"computed","active_frames":int(len(ix)),"variable_taxels_both_halves":n,
            "valid_edges":int(ok.sum()),"split_half_edge_correlation_pearson":corr,
            "note":"Exploratory within-episode reliability diagnostic, not cross-object evidence."}

def hand_h5(group, expected_contact, metadata_rate):
    pad_keys=sorted([k for k in group.keys() if k.startswith("pad_")],key=lambda x:int(x.split("_")[1]))
    arrays=[np.asarray(group[k][()]) for k in pad_keys]
    if not arrays: return {"status":"no_pad_arrays"}
    lengths=[a.shape[0] for a in arrays]
    if len(set(lengths))!=1: return {"status":"pad_time_length_mismatch","pad_shapes":{k:list(group[k].shape) for k in pad_keys}}
    T=lengths[0]
    X=np.concatenate([a.reshape(T,-1).astype(np.float32) for a in arrays],axis=1)
    finite=np.isfinite(X); X=np.where(finite,X,0.0)
    force=X.sum(axis=1); active=force>3.0
    ts=np.asarray(group["t_epoch"][()]).reshape(-1) if "t_epoch" in group else np.array([])
    dt=np.diff(ts.astype(np.float64)) if len(ts)==T else np.array([])
    rate=float(1/np.median(dt)) if len(dt) and np.median(dt)>0 else None
    contact=float(active.mean())
    return {
      "status":"ok","frames":int(T),"taxel_count":int(X.shape[1]),"pad_count":len(pad_keys),
      "pad_shapes":{k:list(group[k].shape) for k in pad_keys},
      "timestamps_length":int(len(ts)),"timestamps_monotonic":bool(np.all(dt>0)) if len(dt) else None,
      "estimated_rate_hz_from_h5_t_epoch":rate,"metadata_glove_rate_hz":float(metadata_rate),
      "rate_abs_difference_hz":float(abs(rate-metadata_rate)) if rate else None,
      "finite_value_fraction":float(finite.mean()),"force_min":float(X.min()),"force_max":float(X.max()),
      "active_fraction_sum_force_gt_3N":contact,"metadata_contact_fraction":float(expected_contact),
      "contact_fraction_abs_difference":float(abs(contact-expected_contact)),
      "active_segment_count":len(segments(active)),"active_segments":segments(active)[:30],
      "split_half_structure_stability":split_half(X,active)
    }

def h5_audit(blob, meta):
    detail={"hands":{}}
    with h5py.File(__import__("io").BytesIO(blob),"r") as f:
        for hand in ("left","right"):
            path=f"obs/tactile/raw/{hand}"
            detail["hands"][hand]=hand_h5(f[path],meta[hand],meta["rate"]) if path in f else {"status":"missing_hand_group"}
        if "task" in f:
            tg=f["task"]
            detail["task"]={}
            for key in ("name","task_id","success","seg_start","seg_end","steps"):
                if key in tg:
                    value=np.asarray(tg[key][()])
                    if value.size<=100:
                        detail["task"][key]=value.tolist()
                    else:
                        detail["task"][key]={"shape":list(value.shape),"dtype":str(value.dtype)}
        if "time/timestamp_ns" in f:
            ts=np.asarray(f["time/timestamp_ns"][()]).reshape(-1)
            dt=np.diff(ts.astype(np.float64))*1e-9
            detail["episode_timeline"]={"frames":int(len(ts)),"monotonic":bool(np.all(dt>0)),
                "median_rate_hz":float(1/np.median(dt)) if len(dt) and np.median(dt)>0 else None}
    return detail

def main():
    out=Path("artifacts/touchscale_signal_pilot"); out.mkdir(parents=True,exist_ok=True)
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    shard_path=hf_hub_download(repo_id=REPO,filename=SHARD,repo_type="dataset",token=token)
    wanted={eid+".episode.h5":eid for eid in SELECTED}
    found={}
    with tarfile.open(shard_path,"r:*") as tar:
        for member in tar:
            base=Path(member.name).name
            if base in wanted:
                f=tar.extractfile(member)
                if f: found[wanted[base]]=f.read()
            if len(found)==len(SELECTED): break
    details={}; rows=[]
    for eid,meta in SELECTED.items():
        if eid not in found:
            details[eid]={"task":meta["task"],"status":"episode_h5_not_found_in_selected_shard"}
            continue
        audited=h5_audit(found[eid],meta)
        details[eid]={"task":meta["task"],"family":meta["family"],"shard":SHARD,**audited}
        for hand,h in audited["hands"].items():
            rows.append({"episode_id":eid,"family":meta["family"],"task":meta["task"],"hand":hand,
                         "status":h.get("status"),"frames":h.get("frames"),"taxels":h.get("taxel_count"),
                         "pads":h.get("pad_count"),"h5_rate_hz":h.get("estimated_rate_hz_from_h5_t_epoch"),
                         "metadata_rate_hz":h.get("metadata_glove_rate_hz"),
                         "contact_fraction":h.get("active_fraction_sum_force_gt_3N"),
                         "metadata_contact_fraction":h.get("metadata_contact_fraction"),
                         "contact_abs_diff":h.get("contact_fraction_abs_difference"),
                         "active_segments":h.get("active_segment_count"),
                         "split_half_edge_corr":h.get("split_half_structure_stability",{}).get("split_half_edge_correlation_pearson")})
    (out/"signal_pilot_detail.json").write_text(json.dumps(details,indent=2,default=str))
    pd.DataFrame(rows).to_csv(out/"signal_pilot_summary.csv",index=False)
    (out/"source_shard.txt").write_text(f"{SHARD}\nsha256={file_sha256(shard_path)}\n")
    print(json.dumps({"episodes_found":len(found),"episodes_selected":len(SELECTED),
                      "output":str(out.resolve()),"summary_rows":len(rows)},indent=2))
if __name__=="__main__": main()
