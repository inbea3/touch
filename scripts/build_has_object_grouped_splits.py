#!/usr/bin/env python3
"""Create deterministic object-disjoint HAS splits from approved metadata only."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, sys, platform
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

DATASET="TUM-ICS/Hide-and-Seek"
SPLITS=("train","validation","test")
ALLOWED=("episode_index","classification_label")

def base_object(label_name: str) -> str:
    if label_name == "negative":
        return "negative"
    # Mapping labels are <physical-object>_<weight>; split on final underscore.
    return label_name.rsplit("_",1)[0]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",default=None)
    p.add_argument("--seed",type=int,default=20261123)
    p.add_argument("--out",default="results/has_object_grouped_split")
    a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    mapping=json.loads(Path("data/has_label_mapping.json").read_text())
    id_to_name={int(v):k for k,v in mapping["classification_labels"].items()}
    episodes={}
    for split in SPLITS:
        kw={"split":split,"streaming":True}
        if a.revision: kw["revision"]=a.revision
        ds=load_dataset(DATASET,**kw)
        for row in ds:
            # Strict metadata projection: no force vector or other modality is read.
            eid=int(row["episode_index"]); lab=int(row["classification_label"])
            key=(split,eid)
            if key not in episodes:
                name=id_to_name.get(lab,f"UNKNOWN_{lab}")
                episodes[key]={"official_split":split,"episode_index":eid,"label_id":lab,
                               "label_name":name,"object_group":base_object(name)}
            elif episodes[key]["label_id"]!=lab:
                raise RuntimeError(f"Episode label changed within {key}")
    by_obj=defaultdict(list)
    for rec in episodes.values(): by_obj[rec["object_group"]].append(rec)
    objects=sorted(by_obj)
    if "negative" in by_obj:
        raise RuntimeError("Negative/no-object episodes require a separately defined policy; do not silently group them with physical objects.")
    rng=random.Random(a.seed)
    # Exact object counts: 21 train / 6 validation / 6 test for 33 groups.
    # Greedy assignment balances episode counts while preserving the fixed object counts.
    shuffled=objects[:]; rng.shuffle(shuffled)
    targets={"train":21,"validation":6,"test":6}
    target_eps={k:len(episodes)*v/len(objects) for k,v in targets.items()}
    counts={k:0 for k in targets}; assigned={}
    # Largest episode groups first; deterministic seeded tie-breaking.
    tie={o:rng.random() for o in objects}
    order=sorted(shuffled,key=lambda o:(-len(by_obj[o]),tie[o]))
    for obj in order:
        available=[s for s in SPLITS if counts[s]<targets[s]]
        # Minimize normalized episode-count deviation after assigning this group.
        def cost(s):
            trial={k:counts[k]+(len(by_obj[obj]) if k==s else 0) for k in SPLITS}
            return sum(((trial[k]-target_eps[k])/max(target_eps[k],1))**2 for k in SPLITS)
        chosen=min(available,key=lambda s:(cost(s),SPLITS.index(s)))
        assigned[obj]=chosen; counts[chosen]+=len(by_obj[obj])
    if {s:sum(assigned[o]==s for o in objects) for s in SPLITS}!=targets:
        raise RuntimeError("Object count allocation invariant failed")
    rows=[]
    for obj in objects:
        for rec in by_obj[obj]:
            rows.append({**rec,"grouped_split":assigned[obj]})
    rows.sort(key=lambda r:(SPLITS.index(r["grouped_split"]),r["object_group"],r["official_split"],r["episode_index"]))
    with (out/"episode_split_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    objrows=[{"object_group":o,"grouped_split":assigned[o],"n_episodes":len(by_obj[o]),
              "n_label_variants":len({r["label_name"] for r in by_obj[o]})} for o in objects]
    with (out/"object_split_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(objrows[0])); w.writeheader(); w.writerows(objrows)
    summary={"dataset":DATASET,"revision":a.revision,"seed":a.seed,"allowed_fields":list(ALLOWED),
      "n_objects":len(objects),"n_episodes":len(episodes),
      "object_counts":{s:sum(assigned[o]==s for o in objects) for s in SPLITS},
      "episode_counts":{s:sum(assigned[o]==s for o in objects for _ in by_obj[o]) for s in SPLITS},
      "episode_weighted_counts":{s:sum(len(by_obj[o]) for o in objects if assigned[o]==s) for s in SPLITS},
      "objects_by_split":{s: [o for o in objects if assigned[o]==s] for s in SPLITS},
      "official_split_crossing_retained_as_metadata":True,
      "grouping_rule":"base physical object category; weight suffix variants grouped together",
      "interpretation":"Object-disjoint exploratory split. With only 33 object categories, estimates will have wide uncertainty; split seed must be locked before modelling."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    def sha(p):
        h=hashlib.sha256(p.read_bytes()).hexdigest(); return f"{h}  {p.name}"
    files=sorted(x for x in out.iterdir() if x.is_file() and x.name!="manifest.sha256")
    (out/"manifest.sha256").write_text("\n".join(sha(x) for x in files)+"\n")
    (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
