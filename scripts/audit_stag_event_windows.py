#!/usr/bin/env python3
"""Audit STAG event windows under recording-wide and event-relative 0.5 rules.

The recording-wide min--max rule is the primary paper-aligned interpretation.
The event-relative rule is retained as a sensitivity analysis because the
paper does not define the denominator for the 0.5 threshold explicitly.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from touch_research.decomposition.stag import (
    active_sensor_mask, normalize_pressure, official_grasp_events,
    smoothed_mean_pressure, event_window,
)


def vec(mat, key, dtype):
    return np.asarray(mat[key]).reshape(-1).astype(dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, default=Path("data/interim/metadata.mat"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/stag_decomposition"))
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--radius", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    m = loadmat(args.metadata, simplify_cells=True)
    pressure = np.asarray(m["pressure"])
    recording_id = vec(m, "recordingId", np.int64)
    object_id = vec(m, "objectId", np.int64)
    transition = vec(m, "isTransition", bool)
    grasp_key = "isGrasp" if "isGrasp" in m else "isGrap"
    grasp = vec(m, grasp_key, bool)
    transition_dir = vec(m, "transitionDir", np.int64)
    has_valid = vec(m, "hasValidLabel", bool)
    objects = [str(x) for x in np.asarray(m["objects"], dtype=object).reshape(-1)]
    recordings = [str(x) for x in np.asarray(m["recordings"], dtype=object).reshape(-1)]
    mask = active_sensor_mask(pressure)

    event_rows, no_marker_rows = [], []
    for rid in np.unique(recording_id):
        ids = np.flatnonzero(recording_id == rid)
        cls = int(object_id[ids[0]])
        if cls == 0:
            continue
        x = normalize_pressure(pressure[ids], mask)
        mean = smoothed_mean_pressure(x, sigma_frames=args.sigma)
        events = official_grasp_events(int(rid), transition[ids],
                                       transition_dir[ids], grasp[ids])
        reliable = [e for e in events if e.reliable]
        if not events:
            no_marker_rows.append({
                "recording_id": int(rid), "recording": recordings[int(rid)],
                "object_id": cls, "object": objects[cls],
                "n_frames": int(len(ids)),
                "n_hasValidLabel": int(has_valid[ids].sum()),
                "n_isTransition": int(transition[ids].sum()),
                "n_isGrasp": int(grasp[ids].sum()),
            })
        span = float(np.nanmax(mean) - np.nanmin(mean))
        z_global = ((mean - np.nanmin(mean)) / span if span > 0
                    else np.full_like(mean, np.nan))
        for e in reliable:
            t, q = e.transition_index, e.peak_index
            delta = float(mean[q] - mean[t])
            z_event_q = 1.0 if delta > 0 else np.nan
            w_global = event_window(mean, t, q, args.radius, args.threshold, "recording")
            w_event = event_window(mean, t, q, args.radius, args.threshold, "event")
            event_row = {
                "recording_id": int(rid), "recording": recordings[int(rid)],
                "object_id": cls, "object": objects[cls],
                "transition_index": t, "peak_anchor_index": q,
                "transition_dir": int(e.offset),
                "recording_normalized_peak": float(z_global[q]),
                "baseline_to_peak_rise": delta,
                "peak_anchor_hasValidLabel": bool(has_valid[ids[q]]),
                "recording_window_start": w_global[0] if w_global else None,
                "recording_window_end": w_global[1] if w_global else None,
                "recording_window_length": (w_global[1]-w_global[0]+1) if w_global else 0,
                "event_window_start": w_event[0] if w_event else None,
                "event_window_end": w_event[1] if w_event else None,
                "event_window_length": (w_event[1]-w_event[0]+1) if w_event else 0,
                "event_peak_normalized": z_event_q,
            }
            event_rows.append(event_row)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if event_rows:
        with (out / "event_window_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(event_rows[0]))
            w.writeheader(); w.writerows(event_rows)
    with (out / "no_marker_recordings_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["recording_id","recording","object_id","object","n_frames",
                  "n_hasValidLabel","n_isTransition","n_isGrasp"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(no_marker_rows)

    global_ok = sum(r["recording_window_length"] > 0 for r in event_rows)
    event_ok = sum(r["event_window_length"] > 0 for r in event_rows)
    rejected = [r for r in event_rows if r["recording_window_length"] == 0]
    print({
        "reliable_events": len(event_rows),
        "recording_relative_windows": global_ok,
        "event_relative_windows": event_ok,
        "recording_relative_rejected_peak_hasValidLabel": sum(r["peak_anchor_hasValidLabel"] for r in rejected),
        "recordings_without_official_markers": no_marker_rows,
    })


if __name__ == "__main__":
    main()
