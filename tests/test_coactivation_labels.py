import unittest
import numpy as np
from touch_research.coactivation.labels import (
    pairwise_pearson, threshold_positive_coactivation, upper_triangle_values
)

class CoactivationLabelTests(unittest.TestCase):
    def test_pairwise_pearson_is_symmetric(self):
        x=np.array([[1,2,3],[2,4,2],[3,6,1],[4,8,0]],dtype=float)
        c=pairwise_pearson(x)
        self.assertTrue(np.allclose(c,c.T,equal_nan=True))
        self.assertAlmostEqual(c[0,1],1.0,places=7)
        self.assertAlmostEqual(c[0,2],-1.0,places=7)
        self.assertTrue(np.isnan(np.diag(c)).all())

    def test_constant_sensor_is_undefined(self):
        x=np.array([[1,5],[2,5],[3,5],[4,5]],dtype=float)
        c=pairwise_pearson(x)
        self.assertTrue(np.isnan(c[0,1]))
        self.assertFalse(threshold_positive_coactivation(c,0.7)[0,1])

    def test_positive_threshold_excludes_negative_edges(self):
        x=np.array([[1,4],[2,3],[3,2],[4,1]],dtype=float)
        c=pairwise_pearson(x)
        y=threshold_positive_coactivation(c,0.7)
        self.assertFalse(y[0,1])
        self.assertFalse(y[1,0])

    def test_upper_triangle_contains_each_pair_once(self):
        m=np.arange(9).reshape(3,3)
        self.assertEqual(upper_triangle_values(m).tolist(),[1,2,5])

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            pairwise_pearson(np.ones((2,3)))
        with self.assertRaises(ValueError):
            threshold_positive_coactivation(np.eye(2),1.0)

if __name__=="__main__":
    unittest.main()
