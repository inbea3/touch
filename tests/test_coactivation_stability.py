import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_coactivation_stability import corr_pearson,corr_spearman,corr_ledoit_wolf,block_indices,compare

class CoactivationStabilityTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42);a=rng.normal(size=160)
        self.x=np.column_stack([a,a+.15*rng.normal(size=160),np.sin(a)+.1*rng.normal(size=160),rng.normal(size=160)])

    def test_pearson_symmetric_and_bounded(self):
        c=corr_pearson(self.x)
        self.assertTrue(np.allclose(c,c.T,equal_nan=True))
        v=c[np.isfinite(c)]
        self.assertTrue(np.all((v>=-1)&(v<=1)))

    def test_spearman_monotonic_relation(self):
        self.assertGreater(corr_spearman(self.x)[0,1],.9)

    def test_ledoit_wolf_valid_correlation(self):
        c,s=corr_ledoit_wolf(self.x)
        self.assertTrue(0<=s<=1)
        v=c[np.isfinite(c)]
        self.assertTrue(np.all((v>=-1)&(v<=1)))

    def test_block_indices_keep_length_and_bounds(self):
        ix=block_indices(101,10,np.random.default_rng(7))
        self.assertEqual(len(ix),101)
        self.assertTrue(np.all((ix>=0)&(ix<101)))

    def test_identical_matrices_have_perfect_agreement(self):
        c=corr_pearson(self.x);m=compare(c,c)
        self.assertAlmostEqual(m["edge_spearman"],1.)
        self.assertAlmostEqual(m["jaccard"]["0.7"],1.)

if __name__=="__main__":unittest.main()
