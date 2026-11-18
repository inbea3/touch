#!/usr/bin/env python3
"""Build object-disjoint STAG splits and exhaustive 1/2-shot Support/Query episodes."""
from __future__ import annotations
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

DEFAULT_SEED=20261118
SPLIT_COUNTS={"meta_train":18,"meta_validation":4,"meta_test":4}

def arr1(x,name):
    a=np.asarray(x).reshape(-1)
    if not len(a): raise ValueError(f"{name} is empty")
    return a

def make_object_split(object_ids,seed=DEFAULT_SEED):
    ids=sorted(int(x) for x in object_ids)
    if len(ids)!=26 or len(set(ids))!=26 or 0 in ids:
        raise ValueError("Expected exactly 26 unique non-empty object IDs (excluding ID 0)")
    perm=np.random.default_rng(seed).permutation(np.asarray(ids,dtype=int)).tolist()
    out={"meta_train":sorted(map(int,perm[:18])),
         "meta_validation":sorted(map(int,perm[18:22])),
         "meta_test":sorted(map(int,perm[22:]))}
    return out,list(map(int,perm))

def extract_recording_manifest(m,k_early=5):
    names=arr1(m["recordings"],"recordings").astype(str)
    objects=arr1(m["objects"],"objects").astype(str)
    rid=arr1(m["recordingId"],"recordingId").astype(int)
    oid=arr1(m["objectId"],"objectId").astype(int)
    frame=arr1(m["frame"],"frame").astype(int)
    ts=arr1(m["ts"],"ts").astype(float)
    valid=arr1(m["hasValidLabel"],"hasValidLabel").astype(bool)
    official=arr1(m["splitId"],"splitId").astype(int)
    n=len(rid)
    if any(len(x)!=n for x in (oid,frame,ts,valid,official)):
        raise ValueError("Frame-level metadata arrays have inconsistent lengths")
    rows=[]
    for r in sorted(np.unique(rid).tolist()):
        ix=np.flatnonzero(rid==r)
        oids=np.unique(oid[ix])
        if len(oids)!=1: raise ValueError(f"Recording {r} maps to multiple objects")
        o=int(oids[0])
        if o==0: continue
        if not (0<=r<len(names) and 0<=o<len(objects)): raise ValueError("Invalid recording/object ID")
        ix=ix[np.argsort(ts[ix],kind="stable")]
        if np.any(np.diff(ts[ix])<0): raise ValueError(f"Timestamp order error in recording {r}")
        vpos=np.flatnonzero(valid[ix])
        if len(vpos)<=k_early: raise ValueError(f"Recording {r} lacks future valid frames for K={k_early}")
        early=vpos[:k_early]
        future=vpos[vpos>early[-1]]
        ef=frame[ix[early]].astype(int).tolist()
        et=ts[ix[early]].astype(float).tolist()
        ff=frame[ix[future]].astype(int)
        rows.append({
          "recording_id":int(r),"recording_name":str(names[r]),"object_id":o,"object_name":str(objects[o]),
          "n_frames":int(len(ix)),"n_valid_frames":int(len(vpos)),"k_early":int(k_early),
          "input_frame_ids":";".join(map(str,ef)),
          "input_timestamps_sec":";".join(f"{x:.9f}" for x in et),
          "input_span_sec":float(et[-1]-et[0]),
          "input_is_contiguous":bool(np.all(np.diff(ef)==1)),
          "boundary_frame_id":int(ef[-1]),"boundary_timestamp_sec":float(et[-1]),
          "n_future_valid_frames":int(len(future)),
          "target_first_frame_id":int(ff[0]),"target_last_frame_id":int(ff[-1]),
          "official_split_ids_diagnostic_only":";".join(map(str,sorted(map(int,np.unique(official[ix])))))
        })
    return rows

def enumerate_episodes(recordings,object_split,split):
    byobj={}
    for r in recordings:
        if int(r["object_id"]) in object_split[split]:
            byobj.setdefault(int(r["object_id"]),[]).append(r)
    out=[]
    for oid in sorted(object_split[split]):
        rr=sorted(byobj.get(oid,[]),key=lambda x:int(x["recording_id"]))
        if len(rr)!=3: raise ValueError(f"Object {oid} expected 3 recordings, found {len(rr)}")
        for shots in (1,2):
            for support in itertools.combinations(rr,shots):
                sids=sorted(int(x["recording_id"]) for x in support)
                query=[x for x in rr if int(x["recording_id"]) not in sids]
                if not query: raise ValueError("Episode has empty Query")
                out.append({
                  "episode_id":f"{split}_obj{oid:02d}_{shots}shot_support{'-'.join(map(str,sids))}",
                  "split":split,"object_id":oid,"object_name":rr[0]["object_name"],
                  "support_shots":shots,"support_recording_ids":";".join(map(str,sids)),
                  "query_recording_ids":";".join(str(int(x["recording_id"])) for x in query),
                  "support_recording_names":";".join(x["recording_name"] for x in sorted(support,key=lambda x:int(x["recording_id"]))),
                  "query_recording_names":";".join(x["recording_name"] for x in query),
                  "n_support_recordings":len(support),"n_query_recordings":len(query),
                  "support_future_valid_frames":";".join(str(x["n_future_valid_frames"]) for x in sorted(support,key=lambda x:int(x["recording_id"]))),
                  "query_future_valid_frames":";".join(str(x["n_future_valid_frames"]) for x in query)
                })
    return out

def validate_manifests(recordings,object_split,episodes):
    flat=[int(x) for group in object_split.values() for x in group]
    if len(flat)!=26 or len(set(flat))!=26: raise ValueError("Object sets overlap or omit objects")
    byobj={}
    for r in recordings: byobj.setdefault(int(r["object_id"]),[]).append(r)
    if len(recordings)!=78 or any(len(byobj.get(x,[]))!=3 for x in flat):
        raise ValueError("Expected 78 recordings and exactly 3 per object")
    owner={int(r["recording_id"]):int(r["object_id"]) for r in recordings}
    recsplit={}
    for split,oids in object_split.items():
        for oid in oids:
            for r in byobj[int(oid)]:
                rid=int(r["recording_id"])
                if rid in recsplit: raise ValueError(f"Recording {rid} assigned more than once")
                recsplit[rid]=split
    if len(recsplit)!=78: raise ValueError("Recording split is incomplete")
    seen=set()
    for e in episodes:
        if e["episode_id"] in seen: raise ValueError("Duplicate episode ID")
        seen.add(e["episode_id"])
        s=set(map(int,e["support_recording_ids"].split(";")))
        q=set(map(int,e["query_recording_ids"].split(";")))
        if not s or not q or s&q: raise ValueError(f"Support/Query overlap in {e['episode_id']}")
        if len(s)!=int(e["support_shots"]): raise ValueError("Support count mismatch")
        if any(owner.get(r)!=int(e["object_id"]) for r in s|q): raise ValueError("Episode crosses object")
        if any(recsplit.get(r)!=e["split"] for r in s|q): raise ValueError("Episode crosses outer split")
    return {"object_partition_disjoint":True,"recording_partition_complete":True,
      "same_object_recordings_stay_in_one_split":True,"support_query_recording_disjoint":True,
      "episode_recordings_match_episode_object":True,"episode_recordings_match_outer_split":True,
      "n_recordings":len(recordings),"n_episodes":len(episodes)}

def write_csv(path,rows):
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build(metadata,output_dir,seed=DEFAULT_SEED,k_early=5):
    m=loadmat(metadata,simplify_cells=True)
    recs=extract_recording_manifest(m,k_early)
    split,perm=make_object_split(sorted({r["object_id"] for r in recs}),seed)
    episodes=[e for s in SPLIT_COUNTS for e in enumerate_episodes(recs,split,s)]
    checks=validate_manifests(recs,split,episodes)
    lookup={oid:s for s,ids in split.items() for oid in ids}
    for r in recs: r["outer_split"]=lookup[r["object_id"]]
    object_names=arr1(m["objects"],"objects").astype(str)
    split_objects={s:[{"object_id":int(oid),"object_name":str(object_names[oid]),
      "recording_ids":sorted(int(r["recording_id"]) for r in recs if r["object_id"]==oid)} for oid in ids]
      for s,ids in split.items()}
    split_summary={}
    for s,ids in split.items():
        means=[float(np.mean([r["n_future_valid_frames"] for r in recs if r["object_id"]==oid])) for oid in ids]
        split_summary[s]={"n_objects":len(ids),"n_recordings":sum(r["outer_split"]==s for r in recs),
          "mean_future_valid_frames_per_object":float(np.mean(means)),
          "min_object_mean_future_valid_frames":float(np.min(means)),
          "max_object_mean_future_valid_frames":float(np.max(means))}
    output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(output_dir/"recording_manifest.csv",recs)
    write_csv(output_dir/"episode_manifest.csv",episodes)
    manifest={"schema_version":"1.0","dataset":"STAG/HumanGrasp pressure-only classification archive",
      "scope":"No synchronized video, image annotation, or multimodal data","seed":int(seed),
      "object_permutation_before_sorting":perm,"split_counts":SPLIT_COUNTS,"k_early":int(k_early),
      "split_strategy":"deterministic random permutation of sorted non-empty object IDs; 18/4/4",
      "object_splits":split_objects,
      "episode_design":{"task_unit":"object","sample_unit":"recording","support_shots":[1,2],
        "query_rule":"all remaining recordings of the same object",
        "enumeration":"all combinations; 3 one-shot plus 3 two-shot episodes per object",
        "train_use":"sample from meta-train episode pool; independent task count remains 18 objects",
        "validation_use":"adapt on Support labels; Query only for model selection",
        "test_use":"adapt on Support labels after all choices are frozen; Query only for final evaluation",
        "warning":"Support/Query permutations reuse recordings and are correlated; aggregate metrics by object"},
      "checks":checks}
    (output_dir/"object_split_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    audit={"n_objects":26,"n_recordings":78,"n_empty_hand_recordings_excluded":4,
      "n_episodes_total":len(episodes),
      "episodes_by_split":{s:sum(e["split"]==s for e in episodes) for s in SPLIT_COUNTS},
      "episodes_by_split_and_shots":{s:{str(k):sum(e["split"]==s and e["support_shots"]==k for e in episodes) for k in (1,2)} for s in SPLIT_COUNTS},
      "split_summary":split_summary,
      "recording_level":{"future_valid_frames_mean":float(np.mean([r["n_future_valid_frames"] for r in recs])),
        "future_valid_frames_median":float(np.median([r["n_future_valid_frames"] for r in recs])),
        "future_valid_frames_min":int(min(r["n_future_valid_frames"] for r in recs)),
        "future_valid_frames_max":int(max(r["n_future_valid_frames"] for r in recs)),
        "early_input_contiguous_recordings":sum(r["input_is_contiguous"] for r in recs),
        "early_input_noncontiguous_recordings":sum(not r["input_is_contiguous"] for r in recs),
        "early_input_span_median_sec":float(np.median([r["input_span_sec"] for r in recs])),
        "early_input_span_max_sec":float(max(r["input_span_sec"] for r in recs))},
      "leakage_checks":checks,
      "limitations":["Only four held-out meta-test objects in this single split; object-level uncertainty is high.",
        "Six Support/Query permutations per object are correlated, not six independent tasks.",
        "Official Recording-level split is diagnostic only and does not define the object split.",
        "The earliest five valid frames are not always consecutive; input span sensitivity remains necessary.",
        "No raw pressure arrays or edge-level labels are exported."]}
    (output_dir/"split_episode_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=DEFAULT_SEED);p.add_argument("--k-early",type=int,default=5)
    a=p.parse_args();build(a.metadata,a.output_dir,a.seed,a.k_early)
if __name__=="__main__": main()
