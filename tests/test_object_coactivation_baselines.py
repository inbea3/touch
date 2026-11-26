import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_object_coactivation_baselines import corr,stats,adapt_logistic,episodes
class BaselineTests(unittest.TestCase):
 def test_corr_symmetry_and_diagonal(self):
  c=corr(np.random.default_rng(7).normal(size=(30,6)))
  self.assertTrue(np.allclose(c,c.T,equal_nan=True));self.assertTrue(np.isnan(np.diag(c)).all())
 def test_constant_sensor_pair_nan(self):
  c=corr(np.column_stack([np.arange(8),np.ones(8),np.arange(8)**2]))
  self.assertTrue(np.isnan(c[0,1]));self.assertTrue(np.isfinite(c[0,2]))
 def test_stats_ignore_nan(self):
  s=stats([1,np.nan,3]);self.assertEqual(s["n"],2);self.assertEqual(s["mean"],2)
 def test_episode_disjointness(self):
  d={r:{"object_id":1 if r<3 else 2,"object_name":"x","split":"meta_test"} for r in range(6)}
  es=episodes(d,"meta_test");self.assertEqual(len(es),12)
  for _,_,s,q in es:self.assertTrue(set(s).isdisjoint(q));self.assertEqual(len(s)+len(q),3)
 def test_logistic_adaptation_finite(self):
  rng=np.random.default_rng(9);X=rng.normal(size=(300,4));y=(X[:,0]>.2).astype(int)
  model=LogisticRegression(class_weight="balanced").fit(X,y);w,b=adapt_logistic(model,X[:80],y[:80])
  self.assertTrue(np.isfinite(w).all());self.assertTrue(np.isfinite(b))
if __name__=="__main__":unittest.main()
