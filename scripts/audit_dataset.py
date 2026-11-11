#!/usr/bin/env python3
"""Inventory the official STAG classification MAT metadata without training."""
from __future__ import annotations
import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

def load_mat(path: Path):
    try:
        from scipy.io import loadmat
        data = loadmat(path, simplify_cells=True)
        return {k:v for k,v in data.items() if not k.startswith("__")}, "scipy"
    except (NotImplementedError, ValueError):
        import h5py
        data = {}
        with h5py.File(path, "r") as f:
            for key, value in f.items():
                data[key] = value[()] if isinstance(value, h5py.Dataset) else {"hdf5_group": True}
        return data, "h5py"

def strings(value):
    if value is None: return []
    if isinstance(value, np.ndarray):
        out=[]
        for x in value.reshape(-1): out.extend(strings(x))
        return out
    if isinstance(value, bytes): return [value.decode("utf-8", errors="replace")]
    if isinstance(value, str): return [value]
    if isinstance(value, np.generic): return strings(value.item())
    return []

def ints(value):
    try: return np.asarray(value).reshape(-1).astype(np.int64)
    except (TypeError, ValueError): return np.array([], dtype=np.int64)

def jsonable(value):
    if isinstance(value, np.ndarray):
        if value.size <= 30: return value.tolist()
        return {"shape":list(value.shape),"dtype":str(value.dtype)}
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace")
    if isinstance(value, (str,int,float,bool)) or value is None: return value
    return str(value)

def extract_zip(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    root=target.resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            dest=(target/item.filename).resolve()
            if dest != root and root not in dest.parents:
                raise ValueError(f"Unsafe ZIP path: {item.filename}")
        z.extractall(target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir",type=Path)
    src.add_argument("--archive",type=Path)
    p.add_argument("--extract-to",type=Path,default=Path("data/interim"))
    p.add_argument("--output-dir",type=Path,default=Path("results/data_audit"))
    a=p.parse_args()
    if a.archive:
        if not a.archive.is_file(): p.error(f"Archive not found: {a.archive}")
        extract_zip(a.archive,a.extract_to)
        root=a.extract_to
    else:
        root=a.data_dir
        if not root.is_dir(): p.error(f"Directory not found: {root}")
    candidates=sorted(root.rglob("metadata.mat"))
    if not candidates: raise FileNotFoundError(f"metadata.mat not found under {root}")
    if len(candidates)>1:
        print("Multiple metadata.mat files found; auditing first:", candidates[0])
    mat_path=candidates[0]
    variables,reader=load_mat(mat_path)
    objects=strings(variables.get("objects"))
    object_ids=ints(variables.get("objectId"))
    recording_names=strings(variables.get("recordings"))
    recording_ids=ints(variables.get("recordingId"))
    pressure=variables.get("pressure")
    valid=np.asarray(variables.get("hasValidLabel",[])).reshape(-1)
    balanced=np.asarray(variables.get("isBalanced",[])).reshape(-1)
    grasp=np.asarray(variables.get("isGrasp",[])).reshape(-1)
    transition=np.asarray(variables.get("isTransition",[])).reshape(-1)
    split_ids=ints(variables.get("splitId"))
    sensor_count=int(np.count_nonzero(np.nanstd(np.asarray(pressure,dtype=np.float32),axis=0)>0)) if pressure is not None else None
    inventory={}
    for key,value in variables.items():
        if isinstance(value,np.ndarray):
            inventory[key]={"shape":list(value.shape),"dtype":str(value.dtype)}
        else:
            inventory[key]={"type":type(value).__name__,"preview":jsonable(value)}
    report={
      "dataset":"MIT STAG / HumanGrasp classification pressure-only",
      "metadata_file":str(mat_path),
      "mat_reader":reader,
      "variables":inventory,
      "counts":{
        "object_names":len(objects),
        "object_id_values":int(object_ids.size),
        "unique_object_ids":int(np.unique(object_ids).size) if object_ids.size else None,
        "recording_names":len(recording_names),
        "recording_id_values":int(recording_ids.size),
        "unique_recording_ids":int(np.unique(recording_ids).size) if recording_ids.size else None,
        "valid_label_values":int(valid.size),
        "valid_label_true":int(np.count_nonzero(valid)) if valid.size else None,
        "balanced_true":int(np.count_nonzero(balanced)) if balanced.size else None,
        "isGrasp_true":int(np.count_nonzero(grasp)) if grasp.size else None,
        "isTransition_true":int(np.count_nonzero(transition)) if transition.size else None,
        "unique_split_ids":np.unique(split_ids).tolist() if split_ids.size else [],
        "physical_sensor_pixels_by_nonconstant_mask":sensor_count,
        "pressure_min":int(np.min(pressure)) if pressure is not None else None,
        "pressure_max":int(np.max(pressure)) if pressure is not None else None,
        "pressure_shape":list(np.asarray(pressure).shape) if pressure is not None else None,
      },
      "object_names":objects,
      "notes":[
        "Counts are computed from this local file, not copied from the project webpage.",
        "Official documentation describes 32x32 pressure maps with 548 physical sensor pixels.",
        "Actual field name in the uploaded metadata is isGrasp (not the common typo isGrap).",
        "Frames, overlapping windows and sensor pairs are not independent object-level samples.",
        "This inventory does not infer sensor coordinates or construct prediction labels."
      ]
    }
    a.output_dir.mkdir(parents=True,exist_ok=True)
    out=a.output_dir/"data_audit.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if object_ids.size and objects:
        counts=Counter(int(x) for x in object_ids)
        with (a.output_dir/"object_frame_counts.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f); w.writerow(["object_id_zero_based","object_name","frame_count"])
            for oid,count in sorted(counts.items()):
                name=objects[oid] if 0<=oid<len(objects) else f"unknown_{oid}"
                w.writerow([oid,name,count])
    print("Metadata:",mat_path)
    print("MAT reader:",reader)
    print("Objects listed:",len(objects))
    print("Unique object IDs:",int(np.unique(object_ids).size) if object_ids.size else "unavailable")
    print("Recording names:",len(recording_names))
    print("Pressure shape:",report["counts"]["pressure_shape"])
    print("Audit report:",out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
