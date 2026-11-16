"""STAG hand-pose / object-pressure decomposition.

The public STAG metadata stores the paper's detected event anchors:
`isTransition` marks the pre-contact anchor and `transitionDir` gives the
forward offset to the associated post-contact local maximum. `isGrasp`
(the metadata spelling; the HTML README calls it `isGrap`) marks reliable
events. We use those published preprocessing outputs rather than pretending
that the underspecified event detector can be reconstructed exactly from the
paper text alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class GraspEvent:
    recording_id: int
    event_index: int
    transition_index: int
    peak_index: int
    offset: int
    reliable: bool


@dataclass(frozen=True)
class DecomposedEvent:
    event: GraspEvent
    window_start: int
    window_end: int
    baseline: np.ndarray
    object_pressure: np.ndarray
    window_rule: str


def active_sensor_mask(pressure: np.ndarray, expected_sensors: int = 548) -> np.ndarray:
    """Find physical sensor cells; inactive cells are constant in the MAT file."""
    p = np.asarray(pressure)
    if p.ndim != 3 or p.shape[1:] != (32, 32):
        raise ValueError("pressure 必须具有 (n_frames, 32, 32) 形状")
    mask = np.nanstd(p.astype(np.float32), axis=0) > 0
    if int(mask.sum()) != expected_sensors:
        raise ValueError(f"预期 {expected_sensors} 个有效传感器，实际 {int(mask.sum())}")
    return mask


def normalize_pressure(pressure: np.ndarray, sensor_mask: np.ndarray,
                       low: float = 500.0, high: float = 650.0) -> np.ndarray:
    """Apply the paper's 500--650 to 0--1 affine normalization and clipping."""
    p = np.asarray(pressure)
    mask = np.asarray(sensor_mask, dtype=bool)
    if p.ndim != 3 or mask.shape != p.shape[1:]:
        raise ValueError("pressure 与 sensor_mask 形状不匹配")
    if not high > low:
        raise ValueError("high 必须大于 low")
    x = (p[:, mask].astype(np.float32) - low) / (high - low)
    return np.clip(x, 0.0, 1.0)


def smoothed_mean_pressure(normalized_pressure: np.ndarray, sigma_frames: float = 3.0) -> np.ndarray:
    """Mean over physical sensors, then symmetric Gaussian smoothing."""
    x = np.asarray(normalized_pressure, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("normalized_pressure 必须是至少两帧的二维矩阵")
    if sigma_frames <= 0:
        raise ValueError("sigma_frames 必须大于 0")
    mean = x.mean(axis=1, dtype=np.float64)
    # `reflect` supplies a symmetric boundary extension for the symmetric kernel.
    return gaussian_filter1d(mean, sigma=sigma_frames, mode="reflect").astype(np.float64)


def official_grasp_events(recording_id: int, is_transition: np.ndarray,
                          transition_dir: np.ndarray, is_grasp: np.ndarray) -> list[GraspEvent]:
    """Decode the event anchors already stored in STAG metadata.mat."""
    tr = np.asarray(is_transition, dtype=bool).reshape(-1)
    dr = np.asarray(transition_dir, dtype=np.int64).reshape(-1)
    gr = np.asarray(is_grasp, dtype=bool).reshape(-1)
    if not (tr.size == dr.size == gr.size):
        raise ValueError("isTransition、transitionDir、isGrasp 长度必须一致")
    events: list[GraspEvent] = []
    for j, t in enumerate(np.flatnonzero(tr)):
        offset = int(dr[t])
        if offset <= 0:
            continue
        events.append(GraspEvent(
            recording_id=int(recording_id), event_index=int(j),
            transition_index=int(t), peak_index=int(t + offset),
            offset=offset, reliable=bool(gr[t])))
    return events


def _event_relative_trace(smoothed_mean: np.ndarray, transition: int, peak: int) -> np.ndarray:
    baseline = float(smoothed_mean[transition])
    peak_value = float(smoothed_mean[peak])
    delta = peak_value - baseline
    if not np.isfinite(delta) or delta <= 0:
        return np.full(smoothed_mean.shape, np.nan, dtype=np.float64)
    return (smoothed_mean - baseline) / delta


def event_window(smoothed_mean: np.ndarray, transition: int, peak: int,
                 radius: int = 80, threshold: float = 0.5,
                 normalization: str = "recording") -> tuple[int, int] | None:
    """Find the contiguous valid window around the peak, bounded by radius.

    normalization='recording' (default) uses the full-recording min--max
    scale. This is the primary paper-aligned interpretation because the
    preceding event criteria are explicitly defined against recording dynamic
    range. 'event' uses the transition-to-peak rise as a sensitivity analysis.
    The paper does not state the 0.5 threshold denominator unambiguously.
    """
    y = np.asarray(smoothed_mean, dtype=np.float64).reshape(-1)
    if not (0 <= transition < y.size and 0 <= peak < y.size):
        raise IndexError("transition/peak 超出 recording 范围")
    if peak <= transition:
        return None
    if normalization == "event":
        z = _event_relative_trace(y, transition, peak)
    elif normalization == "recording":
        span = float(np.nanmax(y) - np.nanmin(y))
        z = (y - np.nanmin(y)) / span if span > 0 else np.full_like(y, np.nan)
    else:
        raise ValueError("normalization 只能是 'event' 或 'recording'")
    if not np.isfinite(z[peak]) or z[peak] < threshold:
        return None
    lo_limit = max(0, peak - int(radius))
    hi_limit = min(y.size - 1, peak + int(radius))
    lo = peak
    hi = peak
    while lo > lo_limit and np.isfinite(z[lo - 1]) and z[lo - 1] >= threshold:
        lo -= 1
    while hi < hi_limit and np.isfinite(z[hi + 1]) and z[hi + 1] >= threshold:
        hi += 1
    return int(lo), int(hi)


def decompose_event(normalized_pressure: np.ndarray, smoothed_mean: np.ndarray,
                    event: GraspEvent, radius: int = 80, threshold: float = 0.5,
                    normalization: str = "recording") -> DecomposedEvent | None:
    """Subtract the pre-contact hand-pose frame from the object-contact window."""
    x = np.asarray(normalized_pressure, dtype=np.float32)
    y = np.asarray(smoothed_mean, dtype=np.float64)
    t, q = event.transition_index, event.peak_index
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("pressure 与 mean signal 的帧数不一致")
    if not (0 <= t < x.shape[0] and 0 <= q < x.shape[0]):
        raise IndexError("事件锚点超出 recording 范围")
    if not event.reliable:
        return None
    window = event_window(y, t, q, radius=radius, threshold=threshold,
                          normalization=normalization)
    if window is None:
        return None
    lo, hi = window
    baseline = x[t].astype(np.float32, copy=True)
    obj = x[lo:hi + 1] - baseline[None, :]
    return DecomposedEvent(event, lo, hi, baseline, obj.astype(np.float32), normalization)


def decompose_recording(normalized_pressure: np.ndarray, smoothed_mean: np.ndarray,
                        events: Iterable[GraspEvent], radius: int = 80,
                        threshold: float = 0.5,
                        normalization: str = "recording") -> list[DecomposedEvent]:
    """Decompose all reliable events in one recording."""
    out = []
    for event in events:
        item = decompose_event(normalized_pressure, smoothed_mean, event,
                               radius=radius, threshold=threshold,
                               normalization=normalization)
        if item is not None:
            out.append(item)
    return out
