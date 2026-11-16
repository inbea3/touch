#!/usr/bin/env python3
"""Audit a candidate future co-activation label using STAG metadata.mat."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from touch_research.coactivation.labels import pairwise_pearson

def vec(mat,key,dtype):
    if key not in mat: raise KeyError(f"metadata.mat 缺少字段: {key}")
    return np.asarray(mat[key]).reshape(-1).astype(dtype)

def summary(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return {"n":0}
    return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "sd":float(a.std(ddof=1)) if len(a)>1 else 0.0,
            "min":float(a.min()),"max":float(a.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,default=Path("data/interim/metadata.mat"))
    p.add_argument("--output-dir",type=Path,default=Path("results/coactivation_audit"))
    p.add_argument("--k-early",type=int,default=5)
    p.add_argument("--thresholds",type=float,nargs="+",default=[0.5,0.7,0.9])
    a=p.parse_args()
    if not a.metadata.is_file(): p.error(f"找不到 {a.metadata}")
    mat=loadmat(a.metadata,simplify_cells=True)
    pressure=np.asarray(mat["pressure"]); n=pressure.shape[0]
    rec=vec(mat,"recordingId",np.int64); obj=vec(mat,"objectId",np.int64)
    ts=vec(mat,"ts",np.float64); valid=vec(mat,"hasValidLabel",bool)
    split=vec(mat,"splitId",np.int64)
    if any(v.size!=n for v in (rec,obj,ts,valid,split)):
        raise ValueError("压力矩阵与元数据长度不一致")
    names=[str(x) for x in np.asarray(mat["objects"],dtype=object).reshape(-1)]
    sensor_mask=np.nanstd(pressure.astype(np.float32),axis=0)>0
    if int(sensor_mask.sum())!=548:
        raise ValueError(f"有效传感器像素应为 548，实际 {int(sensor_mask.sum())}")
    pvec=pressure.reshape(n,-1)[:,sensor_mask.reshape(-1)].astype(np.float64)
    tri=np.triu_indices(548,k=1); n_edges=len(tri[0]); rows=[]
    for rid in np.unique(rec):
        ids=np.flatnonzero(rec==rid)
        classes=np.unique(obj[ids]); splits=np.unique(split[ids])
        if len(classes)!=1 or len(splits)!=1:
            raise ValueError(f"Recording {rid} 内 objectId 或 splitId 不唯一")
        cls=int(classes[0])
        if cls==0: continue
        valid_local=np.flatnonzero(valid[ids])
        if len(valid_local)<=a.k_early+3:
            raise ValueError(f"Recording {rid} 有效帧不足")
        early_ids=ids[valid_local[:a.k_early]]
        future_ids=ids[valid_local[a.k_early:]]
        xf=pvec[future_ids]; corr=pairwise_pearson(xf); edge=corr[tri]
        finite=np.isfinite(edge)
        row={"recording_id":int(rid),"object_id":cls,"object_name":names[cls],
             "official_split_id":int(splits[0]),"n_valid_contact_frames":int(len(valid_local)),
             "n_early_frames":int(len(early_ids)),"n_future_frames":int(len(future_ids)),
             "early_input_span_seconds":float(ts[early_ids[-1]]-ts[early_ids[0]]),
             "early_frames_consecutive":bool(np.all(np.diff(early_ids)==1)),
             "n_candidate_edges":int(n_edges),"finite_edge_fraction":float(finite.mean())}
        for tau in a.thresholds:
            pos=finite&(edge>=tau); neg=finite&(edge<=-tau); key=f"{tau:g}"
            row[f"positive_edges_tau_{key}"]=int(pos.sum())
            row[f"positive_edge_fraction_tau_{key}"]=float(pos.sum()/finite.sum())
            row[f"negative_edges_tau_{key}"]=int(neg.sum())
        mid=len(xf)//2; c1=pairwise_pearson(xf[:mid]); c2=pairwise_pearson(xf[mid:])
        v1=c1[tri]; v2=c2[tri]; both=np.isfinite(v1)&np.isfinite(v2)
        row["split_half_edge_corr_spearman"]=float(spearmanr(v1[both],v2[both]).statistic)
        for tau in a.thresholds:
            b1=v1[both]>=tau; b2=v2[both]>=tau; union=int(np.sum(b1|b2))
            row[f"split_half_positive_edge_jaccard_tau_{tau:g}"]=float(np.sum(b1&b2)/union) if union else 1.0
        rows.append(row)
    transition=vec(mat,"isTransition",bool) if "isTransition" in mat else np.zeros(n,bool)
    grasp_key="isGrasp" if "isGrasp" in mat else ("isGrap" if "isGrap" in mat else None)
    grasp=vec(mat,grasp_key,bool) if grasp_key else np.zeros(n,bool)
    event_recs=set(np.unique(rec[grasp]).tolist())
    rec_class={int(r):int(obj[np.flatnonzero(rec==r)[0]]) for r in np.unique(rec)}
    no_marker=[{"recording_id":r,"object_name":names[c]} for r,c in rec_class.items()
               if c!=0 and r not in event_recs]
    marker={"grasp_field_used":grasp_key,"n_transition_markers":int(transition.sum()),
      "n_grasp_markers":int(grasp.sum()),"n_grasp_also_transition":int(np.sum(grasp&transition)),
      "n_recordings_with_grasp_marker":len(event_recs),"object_recordings_without_grasp_marker":no_marker,
      "note":"官方 HTML 拼写为 isGrap，本地 MAT 实际字段为 isGrasp；事件标记只作审查，不参与本版标签。"}
    cols=[k for k in rows[0] if k.startswith(("positive_edge_fraction_","negative_edges_",
          "split_half_edge_corr","split_half_positive_edge_jaccard"))]
    object_ids=sorted(set(r["object_id"] for r in rows))
    result={"summary":{"n_recordings":len(rows),"n_objects":len(object_ids),
      "recordings_per_object_min":min(sum(x["object_id"]==c for x in rows) for c in object_ids),
      "recordings_per_object_max":max(sum(x["object_id"]==c for x in rows) for c in object_ids),
      "n_sensors":548,"n_undirected_candidate_edges":int(n_edges),"n_early_frames":a.k_early,
      "n_future_frames":summary([r["n_future_frames"] for r in rows]),
      "early_input_span_seconds":summary([r["early_input_span_seconds"] for r in rows]),
      "n_recordings_with_consecutive_early_frames":sum(r["early_frames_consecutive"] for r in rows),
      "metrics_by_recording":{c:summary([r[c] for r in rows]) for c in cols},
      "marker_audit":marker,
      "protocol":{"input":"first K valid object-held frames (hasValidLabel=True)",
       "target":"all subsequent valid object-held frames in the same recording",
       "primary_target":"continuous Pearson correlation for each unordered sensor pair",
       "binary_secondary":"positive edge iff Pearson r >= tau; thresholds are sensitivity settings",
       "exclude":"empty_hand; do not use isBalanced subsampling",
       "split":"object-level grouping is required for unseen-object generalization; official splitId is recording-level",
       "dependence":"recording is the graph observation; object is the outer generalization/task unit; edges within a graph are not independent"}},
      "recordings":rows}
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    (out/"coactivation_label_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"coactivation_recording_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    s=result["summary"]
    print(f"recordings={len(rows)}, objects={s['n_objects']}, sensors=548, edges={n_edges}")
    print("future frames:",s["n_future_frames"])
    for tau in a.thresholds:
        print(f"tau={tau:g}: positive edge fraction",s["metrics_by_recording"][f"positive_edge_fraction_tau_{tau:g}"])
        print(f"tau={tau:g}: split-half Jaccard",s["metrics_by_recording"][f"split_half_positive_edge_jaccard_tau_{tau:g}"])
    print("split-half edge Spearman",s["metrics_by_recording"]["split_half_edge_corr_spearman"])
    print("marker audit",marker)
    print(f"saved {out/'coactivation_label_audit.json'} and {out/'coactivation_recording_metrics.csv'}")

if __name__=="__main__": main()
