#!/usr/bin/env python3
"""Stage 24A: audit HAS trajectory-level Support/Query independence and provenance."""
import argparse,csv,hashlib,json,platform,sys,time,zipfile
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def ids(s):
 return [x for x in str(s).split(";") if x]
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--window-data",default="results/has_window_dataset")
 ap.add_argument("--reptile-zip",default="results/has_reptile_stage23.zip")
 ap.add_argument("--out",default="results/has_stage24_independence_audit")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(msg):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {msg}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 # Audit canonical episode/trajectory identifiers from the dataset's own episode index.
 rows=list(csv.DictReader(open(Path(a.window_data)/"window_5_index.csv",encoding="utf-8")))
 if len(rows)!=4192:raise RuntimeError(f"Expected 4192 episodes, got {len(rows)}")
 key_to_obj={};key_to_split={}
 for r in rows:
  key=f'{r["official_split"]}:{r["episode_index"]}'
  if key in key_to_obj:raise RuntimeError(f"Duplicate episode trajectory key: {key}")
  key_to_obj[key]=r["object_group"];key_to_split[key]=r["grouped_split"]
 split_objects={s:set(r["object_group"] for r in rows if r["grouped_split"]==s) for s in ("train","validation","test")}
 overlaps={f"{x}-{y}":sorted(split_objects[x]&split_objects[y]) for x,y in (("train","validation"),("train","test"),("validation","test"))}
 if any(overlaps.values()):raise RuntimeError(f"Object leakage: {overlaps}")
 # Reptile artifact is a zip; no extraction of weights is required for ID audit.
 z=zipfile.ZipFile(a.reptile_zip)
 with z.open("episode_metrics.csv") as f:metrics=list(csv.DictReader(line.decode() for line in f))
 if not metrics:raise RuntimeError("No episode metrics")
 errors=[];support_rows=[];support_cache={}
 for r in metrics:
  sp=r["split"];ob=r["object"];shots=int(r["shots"]);seed=int(r["seed"]);rep=int(r["repeat"])
  sup=ids(r["support_ids"]);qry=ids(r["query_ids"])
  label=f'{sp}|{ob}|seed={seed}|repeat={rep}|shots={shots}|model={r["model"]}'
  if len(sup)!=shots:errors.append(label+":support_count")
  if len(sup)!=len(set(sup)):errors.append(label+":duplicate_support")
  if len(qry)!=len(set(qry)):errors.append(label+":duplicate_query")
  if set(sup)&set(qry):errors.append(label+":support_query_overlap")
  for k in sup+qry:
   if k not in key_to_obj:errors.append(label+":unknown_episode:"+k);continue
   if key_to_obj[k]!=ob:errors.append(label+":wrong_object:"+k)
   if key_to_split[k]!=sp:errors.append(label+":wrong_split:"+k)
  cache=(sp,ob,seed,rep,shots)
  val=(tuple(sup),tuple(qry))
  if cache in support_cache and support_cache[cache]!=val:errors.append(label+":model_support_query_mismatch")
  support_cache[cache]=val
  support_rows.append({"split":sp,"object":ob,"seed":seed,"repeat":rep,"shots":shots,"model":r["model"],
   "n_support":len(sup),"n_query":len(qry),"support_query_overlap":len(set(sup)&set(qry))})
 if errors:raise RuntimeError("Support/query audit failed: "+repr(errors[:20]))
 # Verify every trajectory key occurs in exactly one object and one split, and all models use matching IDs.
 summary={"dataset_episode_identifier":"(official_split, episode_index), documented by upstream dataset as episode/trajectory identifier",
  "episodes":len(rows),"unique_trajectory_keys":len(key_to_obj),
  "objects_by_split":{s:len(v) for s,v in split_objects.items()},"object_overlap":overlaps,
  "reptile_metric_rows":len(metrics),"support_query_configurations":len(support_cache),
  "all_support_counts_match_shots":True,"all_support_query_sets_disjoint":True,
  "all_episode_ids_map_to_same_object_and_split":True,"all_models_use_identical_support_query_ids":True,
  "recording_session_id":"not exposed in the published frame-level dataset schema; no recording-session independence claim is made",
  "limitations":["Episode/trajectory-level separation is verified; higher-level recording-session grouping cannot be verified from published columns.",
   "The six Validation and six Test object groups remain small; prior Test inspection means Test results remain exploratory.",
   "This audit checks sample identity and split isolation, not temporal autocorrelation between distinct trajectories."]}
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 with (out/"trajectory_index_audit.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["trajectory_key","object_group","grouped_split"]);w.writeheader()
  for k in sorted(key_to_obj):w.writerow({"trajectory_key":k,"object_group":key_to_obj[k],"grouped_split":key_to_split[k]})
 with (out/"support_query_audit.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(support_rows[0]));w.writeheader();w.writerows(support_rows)
 (out/"config.json").write_text(json.dumps({"window_data":"Stage22A run 45193420854","reptile_run":"Stage23 run 45201237382","key":"official_split:episode_index","checks":["unique trajectory key","object split disjointness","support count","support/query disjointness","object/split consistency","model support/query identity"]},indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\n")
 log(f"PASS trajectories={len(key_to_obj)} metrics_rows={len(metrics)}")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
