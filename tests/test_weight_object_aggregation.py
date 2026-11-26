import itertools
import numpy as np

from scripts.analyze_weight_object_aggregation import fisher, metrics


def test_equal_grasp_fisher_mean_matches_scalar_fisher_average():
    a = np.array([[np.nan, 0.2], [0.2, np.nan]])
    b = np.array([[np.nan, 0.4], [0.4, np.nan]])
    got = fisher([a, b])
    expected = np.tanh((np.arctanh(0.2) + np.arctanh(0.4)) / 2)
    assert np.isclose(got[0, 1], expected)
    assert np.isnan(got[0, 0])


def test_fisher_median_is_robust_to_extreme_grasp():
    mats = []
    for value in (0.1, 0.2, 0.3, 0.4, 0.99):
        mats.append(np.array([[np.nan, value], [value, np.nan]]))
    got = fisher(mats, median=True)
    assert np.isclose(got[0, 1], 0.3)


def test_weighted_fisher_uses_future_frame_weights():
    a = np.array([[np.nan, 0.1], [0.1, np.nan]])
    b = np.array([[np.nan, 0.5], [0.5, np.nan]])
    got = fisher([a, b], weights=[9, 3])
    expected = np.tanh((9*np.arctanh(0.1)+3*np.arctanh(0.5))/12)
    assert np.isclose(got[0, 1], expected)


def test_ten_unique_three_vs_three_partitions():
    splits = [a for a in itertools.combinations(range(6), 3) if 0 in a]
    assert len(splits) == 10
    for a in splits:
        b = set(range(6)) - set(a)
        assert len(a) == len(b) == 3
        assert set(a).isdisjoint(b)


def test_identical_matrices_have_perfect_edge_agreement():
    a = np.array([[np.nan, 0.8, 0.1], [0.8, np.nan, 0.2], [0.1, 0.2, np.nan]])
    result = metrics(a, a.copy())
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["mean_abs_diff"], 0.0)
    assert np.isclose(result["jaccard_r_ge_0.7"], 1.0)
