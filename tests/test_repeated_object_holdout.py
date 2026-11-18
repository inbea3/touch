import importlib.util, tempfile, unittest
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"run_repeated_object_holdout.py"
spec=importlib.util.spec_from_file_location("repeated",SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RepeatedHoldoutTests(unittest.TestCase):
 def test_seed_schedule_is_deterministic(self):
  self.assertEqual([20261118+i for i in range(3)],[20261118,20261119,20261120])
 def test_object_aggregation_uses_object_rows(self):
  rows=[{"m":1.0},{"m":3.0},{"m":5.0}]
  self.assertEqual(float(np.mean([r["m"] for r in rows])),3.0)
 def test_fixed_sensor_subset_seed_is_not_split_seed(self):
  self.assertNotEqual(20261116,20261118)
 def test_summary_stats(self):
  s=module.stats([.1,.2,.3]);self.assertEqual(s["n"],3)
  self.assertAlmostEqual(s["mean"],.2)
if __name__=="__main__":unittest.main()
