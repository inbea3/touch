#!/usr/bin/env python3
"""Small TouchScale tactile-signal feasibility audit. One shard, seven selected episodes; no video extraction. Pilot only; no model fitting."""
from __future__ import annotations
import io, json, os, tarfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from huggingface_hub import hf_hub_download

REPO = "2077AIDataFoundation/TouchScale"
SHARD = "data/train-00174-of-00430.tar"
SELECTED = {
 "390106b7-80c9-4c0d-8f4b-f1237fe67967": {"family":"single_object_manipulation","task":"Pull the wooden dowel out of the pre-drilled hole"},
 "33a7eb90-76b5-4045-84e3-8828416ab293": {"family":"scene_setup","task":"Set up the test-tube-and-rod comparison display"},
 "30e2c2bd-47bc-4705-bb25-3da5d20a34d0": {"family":"component_assembly","task":"Unscrew the glass reagent-bottle cap and place it on the tray"},
 "403923fe-7a7f-42cb-9a76-c8061bf9b457": {"family":"material_deformation","task":"Wring out a wet cloth once with both hands"},
 "b9500cfb-a22f-4de3-b51c-4aa7290e9715": {"family":"inspection_measurement","task":"Use a tape measure to measure the short-side length of the board"},
 "0b512291-260c-41b2-bdd6-a1c023d33c3b": {"family":"surface_cleaning","task":"Brush a wide-mouth plastic test tube's inner wall with the soft wide-head tube brush"},
 "93a5dd96-6e5b-475a-83f1-78b4fd3cc136": {"family":"transfer_pouring","task":"Use tweezers to hold a test strip and dip it into the liquid (simulated pH-paper test)"},
}
SUFFIXES = (".left_hand_data.npz", ".right_hand_data.npz", ".episode.h5", ".json")

def contiguous_segments(mask):
    ix=np.flatnonzero(mask)
    if not len(ix): return []
    cuts=np.where(np.diff(ix)>1)[0]+1
    groups=np.split(ix,cuts)
    return [{"start":int(g[0]),"end":int(g[-1]),"frames":int(len(g))} for g in groups]

def corr_split_stability(X, active):
    idx=np.flatnonzero(active)
    if len(idx)<40: return {"status":"insufficient_active_frames","active_frames":int(len(idx))}
    # Split active samples chronologically; this is a diagnostic, not a final target estimator.
    half=len(idx)//2
    A=X[idx[:half]].astype(np.float32); B=X[idx[half:]].astype(np.float32)
    sd_a=A.std(axis=0); sd_b=B.std(axis=0)
    valid=(sd_a>1e-8)&(sd_b>1e-8)
    n=int(valid.sum())
    if n<10: return {"status":"too_few_variable_taxels","active_frames":int(len(idx)),"variable_taxels":n}
    A=A[:,valid]; B=B[:,valid]
    # Bound memory and compute time if all 880 taxels vary.
    C1=np.corrcoef(A,rowvar=False); C2=np.corrcoef(B,rowvar=False)
    iu=np.triu_indices(n,k=1)
    v1=C1[iu]; v2=C2[iu]
    good=np.isfinite(v1)&np.isfinite(v2)
    if good.sum()<20: return {"status":"too_few_valid_edges","active_frames":int(len(idx)),"variable_taxels":n}
    x=v1[good]; y=v2[good]
    corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
    return {"status":"computed","active_frames":int(len(idx)),"half_frames":[int(len(A)),int(len(B))],
            "variable_taxels_both_halves":n,"valid_edges":int(good.sum()),
            "split_half_edge_correlation_pearson":corr,
            "interpretation":"Within-episode split-half stability diagnostic only; not evidence of cross-object generalization."}

def audit_npz(blob, expected_contact):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        keys=list(z.files)
        ts_key=next((k for k in ("timestamps","t_epoch","time") if k in keys),None)
        ts=np.asarray(z[ts_key]).reshape(-1) if ts_key else None
        pad_keys=sorted([k for k in keys if k.startswith("tactile_")])
        arrays=[np.asarray(z[k]) for k in pad_keys]
        shapes={k:list(np.asarray(z[k]).shape) for k in pad_keys}
        if not arrays:
            return {"keys":keys,"error":"no tactile_* arrays found"}
        lengths=[a.shape[0] if a.ndim else 0 for a in arrays]
        if len(set(lengths))!=1:
            return {"keys":keys,"pad_shapes":shapes,"error":"pad time dimensions disagree"}
        T=lengths[0]
        X=np.concatenate([a.reshape(T,-1).astype(np.float32) for a in arrays],axis=1)
        finite=np.isfinite(X)
        X=np.where(finite,X,0.0)
        total_force=X.sum(axis=1)
        active=total_force>3.0
        dt=np.diff(ts.astype(np.float64)) if ts is not None and len(ts)==T else np.array([])
        sample_hz=float(1/np.median(dt)) if len(dt) and np.median(dt)>0 else None
        segs=contiguous_segments(active)
        split=corr_split_stability(X,active)
        return {
          "keys":keys,"timestamp_key":ts_key,"frames":int(T),"taxel_count":int(X.shape[1]),
          "pad_count":len(pad_keys),"pad_shapes":shapes,
          "timestamps_length":int(len(ts)) if ts is not None else None,
          "timestamps_monotonic":bool(np.all(dt>0)) if len(dt) else None,
          "estimated_sampling_hz":sample_hz,
          "finite_value_fraction":float(finite.mean()),
          "force_min":float(np.min(X)),"force_max":float(np.max(X)),
          "force_mean_per_taxel_sample":float(np.mean(X)),
          "active_fraction_force_sum_gt_3N":float(active.mean()),
          "metadata_contact_fraction":float(expected_contact) if expected_contact is not None else None,
          "contact_fraction_abs_difference":float(abs(active.mean()-expected_contact)) if expected_contact is not None else None,
          "active_segments":segs[:20],"active_segment_count":len(segs),
          "split_half_structure_stability":split
        }

def inspect_h5(blob):
    out={}
    with h5py.File(io.BytesIO(blob),"r") as f:
        out["root_attrs"]={k:str(v) for k,v in f.attrs.items()}
        paths=[]
        def visitor(name,obj):
            if isinstance(obj,h5py.Dataset):
                item={"shape":list(obj.shape),"dtype":str(obj.dtype)}
                if obj.size<=20:
                    try: item["values"]=np.asarray(obj[()]).tolist()
                    except Exception: pass
                paths.append({name:item})
        f.visititems(visitor)
        out["datasets"]=paths[:200]
        out["dataset_count"]=len(paths)
        for p in ("task","time","obs/tactile/raw"):
            out[f"has_{p.replace('/','_')}"]=p in f
    return out

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    out=Path("artifacts/touchscale_signal_pilot"); out.mkdir(parents=True,exist_ok=True)
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    shard_path=hf_hub_download(repo_id=REPO,filename=SHARD,repo_type="dataset",token=token)
    found={eid:{} for eid in SELECTED}
    wanted={eid+suffix:(eid,suffix) for eid in SELECTED for suffix in SUFFIXES}
    with tarfile.open(shard_path,"r:*") as tar:
        for member in tar:
            base=Path(member.name).name
            if base in wanted:
                eid,suffix=wanted[base]
                f=tar.extractfile(member)
                if f: found[eid][suffix]=f.read()
            if all(any(s in found[eid] for s in (".left_hand_data.npz",".right_hand_data.npz")) for eid in SELECTED):
                # Still continue to get H5 task segmentation and JSON metadata.
                if all(".episode.h5" in found[eid] for eid in SELECTED): break
    rows=[]; detailed={}
    for eid,meta in SELECTED.items():
        rec={"episode_id":eid,**meta,"shard":SHARD}
        for hand in ("left","right"):
            suffix=f".{hand}_hand_data.npz"
            if suffix not in found[eid]:
                rec[f"{hand}_error"]="selected hand NPZ missing from shard"; continue
            expected=None
            # metadata contact fractions are included in the embedded episode JSON where present;
            # these are not used if the JSON lacks them.
            try:
                m=json.loads(found[eid].get(".json",b"{}").decode("utf-8"))
                expected=m.get(f"tactile_contact_frac_{hand}")
            except Exception: m={}
            rec[hand]=audit_npz(found[eid][suffix],expected)
        if ".episode.h5" in found[eid]:
            rec["episode_h5"]=inspect_h5(found[eid][".episode.h5"])
        else:
            rec["episode_h5"]={"error":"episode.h5 not found"}
        detailed[eid]=rec
        for hand in ("left","right"):
            h=rec.get(hand,{})
            rows.append({"episode_id":eid,"family":meta["family"],"task":meta["task"],"hand":hand,
                         "frames":h.get("frames"),"taxels":h.get("taxel_count"),"pads":h.get("pad_count"),
                         "sample_hz":h.get("estimated_sampling_hz"),"active_fraction":h.get("active_fraction_force_sum_gt_3N"),
                         "active_segments":h.get("active_segment_count"),
                         "split_half_edge_corr":h.get("split_half_structure_stability",{}).get("split_half_edge_correlation_pearson"),
                         "signal_error":h.get("error")})
    (out/"signal_pilot_detail.json").write_text(json.dumps(detailed,indent=2,default=str))
    pd.DataFrame(rows).to_csv(out/"signal_pilot_summary.csv",index=False)
    (out/"source_shard.txt").write_text(f"{SHARD}\nsha256={hashlib.sha256(Path(shard_path).read_bytes()).hexdigest()}\n")
    print(json.dumps({"episodes":len(detailed),"hands":len(rows),"output":str(out.resolve()),
                      "found_files":{k:list(v.keys()) for k,v in found.items()}},indent=2))
if __name__=="__main__": main()
