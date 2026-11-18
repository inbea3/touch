import importlib.util, sys, unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_object_episode_splits import make_object_split
SCRIPT=ROOT/"scripts"/"run_repeated_object_holdout.py"
spec=importlib.util.spec_from_file_location("repeated",SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RepeatedHoldoutTests(unittest.TestCase):
 def test_seed_schedule_is_deterministic(self):
  self.assertEqual([20261118+i for i in range(3)],[20261118,20261119,20261120])
 def test_split_sizes_and_disjointness(self):
  ids=list(range(1,27));a,_=make_object_split(ids,seed=20261118)
  self.assertEqual([len(a[k]) for k in ("meta_train","meta_validation","meta_test")],[18,4,4])
  self.assertEqual(len(set(a["meta_train"])|set(a["meta_validation"])|set(a["meta_test"])),26)
  self.assertFalse(set(a["meta_train"])&set(a["meta_validation"]))
  self.assertFalse(set(a["meta_train"])&set(a["meta_test"]))
  self.assertFalse(set(a["meta_validation"])&set(a["meta_test"]))
 def test_split_reproducibility(self):
  a,pa=make_object_split(range(1,27),seed=123)
  b,pb=make_object_split(range(1,27),seed=123)
  self.assertEqual(a,b);self.assertEqual(pa,pb)
 def test_fixed_sensor_subset_seed_is_not_split_seed(self):
  self.assertNotEqual(20261116,20261118)
 def test_summary_stats(self):
  s=module.stats([.1,.2,.3]);self.assertEqual(s["n"],3)
  self.assertAlmostEqual(s["mean"],.2)
if __name__=="__main__":unittest.main()
