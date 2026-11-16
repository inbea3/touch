"""Statistical co-activation targets for tactile sensor time series."""
from __future__ import annotations
import numpy as np

def pairwise_pearson(x: np.ndarray) -> np.ndarray:
    """Return an undirected sensor correlation matrix; undefined pairs are NaN."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x 必须是二维数组 (n_observations, n_sensors)")
    if x.shape[0] < 3:
        raise ValueError("至少需要 3 个观测才能估计 Pearson 相关")
    if x.shape[1] < 2:
        raise ValueError("至少需要 2 个传感器")
    x = x - np.mean(x, axis=0, keepdims=True)
    ss = np.sum(x * x, axis=0)
    denom = np.sqrt(np.outer(ss, ss))
    numer = x.T @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numer / denom
    corr = np.where(np.isfinite(corr), np.clip(corr, -1.0, 1.0), np.nan)
    np.fill_diagonal(corr, np.nan)
    return corr

def threshold_positive_coactivation(corr: np.ndarray, tau: float = 0.7) -> np.ndarray:
    """Threshold positive Pearson co-variation; diagonal and undefined pairs are False."""
    if not -1.0 < tau < 1.0:
        raise ValueError("tau 必须位于 (-1, 1)")
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr 必须是方阵")
    return np.isfinite(c) & (c >= tau) & (~np.eye(c.shape[0], dtype=bool))

def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Return each undirected pair exactly once, excluding the diagonal."""
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix 必须是方阵")
    return a[np.triu_indices(a.shape[0], k=1)]
