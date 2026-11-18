import sys,unittest
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_repeated_object_cv import make_folds,all_object_manifest,aggregate,boot,prepare_input_windows

class RepeatedObjectCVTests(unittest.TestCase):
 def test_reproducible_balanced_folds(self):
  a=make_folds(range(1,27),5,5,19);b=make_folds(range(1,27),5,5,19)
  self.assertEqual(a,b)
  for r in a:
   self.assertEqual(sorted(len(f["test_object_ids"]) for f in r["folds"]),[5,5,5,5,6])
 def test_each_object_held_out_once_per_repeat(self):
  reps=make_folds(range(1,27),5,3,23)
  for r in reps:
   held=[x for f in r["folds"] for x in f["test_object_ids"]]
   self.assertEqual(len(held),26);self.assertEqual(set(held),set(range(1,27)))
   for f in r["folds"]: self.assertFalse(set(f["train_object_ids"])&set(f["test_object_ids"]))
 def test_all_object_manifest(self):
  groups={"meta_train":[],"meta_validation":[],"meta_test":[]}
  for oid in range(1,27):
   s="meta_train" if oid<=18 else ("meta_validation" if oid<=22 else "meta_test")
   groups[s].append({"object_id":oid,"object_name":str(oid),"recording_ids":[3*oid-2,3*oid-1,3*oid]})
  m,ids=all_object_manifest({"object_splits":groups})
  self.assertEqual(ids,list(range(1,27)));self.assertEqual(len(m["object_splits"]["all_objects"]),26)
 def test_aggregate_repeated_object_values(self):
  rows=[{"object_id":1,"object_name":"a","shots":1,"model":"m","metric_group":"regression","metric":"mae","value":.2},
        {"object_id":1,"object_name":"a","shots":1,"model":"m","metric_group":"regression","metric":"mae","value":.4}]
  a=aggregate(rows);self.assertEqual(len(a),1);self.assertAlmostEqual(a[0]["mean_across_repeats"],.3)

 def test_consecutive_input_window_selection(self):
  n=78; rec=np.repeat(np.arange(1,n+1),12); obj=np.repeat(np.repeat(np.arange(1,27),3),12)
  frame=np.tile(np.arange(1,13),n); ts=np.tile(np.arange(12,dtype=float)*.136,n)
  valid=np.tile(np.asarray([1,1,0,1,1,1,1,1,1,1,1,1],dtype=bool),n)
  mat={"recordingId":rec,"objectId":obj,"frame":frame,"ts":ts,"hasValidLabel":valid,
       "pressure":np.zeros((n*12,32,32),dtype=np.float32)}
  groups={"all_objects":[{"object_id":i,"object_name":str(i),"recording_ids":[3*i-2,3*i-1,3*i]} for i in range(1,27)]}
  changed,rows,audit=prepare_input_windows(mat,{"object_splits":groups},"earliest_consecutive_valid",5)
  self.assertEqual(rows[0]["input_frame_ids"],"4;5;6;7;8")
  self.assertEqual(audit["n_contiguous_windows"],78)
  self.assertEqual(audit["n_shifted_windows"],78)
  self.assertFalse(changed["hasValidLabel"][0])
  self.assertTrue(np.all(changed["hasValidLabel"][3:8]))

 def test_bootstrap_ci_contains_mean(self):
  x=boot([1,2,3,4,5],7,1000);self.assertEqual(x["n_objects"],5)
  self.assertLessEqual(x["ci95_low"],x["mean"]);self.assertGreaterEqual(x["ci95_high"],x["mean"])
if __name__=="__main__":unittest.main()
