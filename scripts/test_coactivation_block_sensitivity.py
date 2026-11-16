import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_coactivation_block_sensitivity import corr_pearson,block_indices,compare,lag_autocorrelation

class BlockSensitivityTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123);a=rng.normal(size=250)
        self.x=np.column_stack([a,a+.1*rng.normal(size=250),rng.normal(size=250)])
    def test_block_indices_length_and_bounds(self):
        ix=block_indices(103,20,np.random.default_rng(8))
        self.assertEqual(len(ix),103);self.assertTrue(np.all((ix>=0)&(ix<103)))
    def test_pearson_matrix_is_symmetric(self):
        c=corr_pearson(self.x);self.assertTrue(np.allclose(c,c.T,equal_nan=True))
    def test_identical_matrices_have_perfect_agreement(self):
        c=corr_pearson(self.x);mad,rho,jac=compare(c,c)
        self.assertAlmostEqual(mad,0.,places=7);self.assertAlmostEqual(rho,1.);self.assertAlmostEqual(jac,1.)
    def test_fixed_seed_is_reproducible(self):
        a=block_indices(100,10,np.random.default_rng(2026));b=block_indices(100,10,np.random.default_rng(2026))
        self.assertTrue(np.array_equal(a,b))
    def test_autocorrelation_reports_requested_lags(self):
        rows=lag_autocorrelation(self.x,lags=(1,5,10))
        self.assertEqual([r["lag_frames"] for r in rows],[1,5,10])
        self.assertTrue(all(-1<=r["mean_signed_acf"]<=1 for r in rows))

if __name__=="__main__":unittest.main()
