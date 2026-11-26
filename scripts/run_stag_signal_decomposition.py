#!/usr/bin/env python3
"""Run STAG's documented hand-pose/object-pressure decomposition.

The official MAT event markers are used as event anchors. The paper describes
its detector, but does not publish enough implementation detail to reproduce
its exact peak/minimum selection; the dataset therefore provides the event
anchors (`isTransition`, `transitionDir`, and `isGrasp`/`isGrap`).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from touch_research.coactivation.labels import pairwise_pearson, upper_triangle_values
from touch_research.decomposition.stag import (
    active_sensor_mask, decompose_recording, normalize_pressure,
    official_grasp_events, smoothed_mean_pressure,
)


def vec(mat, key, dtype):
    if key not in mat:
        raise KeyError(f"metadata.mat 缺少字段: {key}")
    return np.asarray(mat[key]).reshape(-1).astype(dtype)


def safe_summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()),
            "median": float(np.median(a)), "sd": float(a.std(ddof=1)) if len(a)>1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def corr_metrics(corr_a, corr_b, thresholds):
    if corr_a is None or corr_b is None:
        return {"edge_spearman": None, "mean_abs_corr_difference": None,
                "finite_common_edges": 0,
                "positive_edge_fraction_a": {f"{t:g}": None for t in thresholds},
                "positive_edge_fraction_b": {f"{t:g}": None for t in thresholds},
                "positive_edge_jaccard": {f"{t:g}": None for t in thresholds}}
    a = upper_triangle_values(corr_a)
    b = upper_triangle_values(corr_b)
    common = np.isfinite(a) & np.isfinite(b)
    common_idx = np.flatnonzero(common)
    if common_idx.size >= 3:
        rng = np.random.default_rng(20261116)
        sample_idx = np.sort(rng.choice(common_idx, size=20000, replace=False)) if common_idx.size > 20000 else common_idx
        rho = float(spearmanr(a[sample_idx], b[sample_idx]).statistic)
        mad = float(np.mean(np.abs(a[common] - b[common])))
    else:
        rho = mad = None
    result = {"edge_spearman": rho, "mean_abs_corr_difference": mad,
              "finite_common_edges": int(common.sum()),
              "positive_edge_fraction_a": {}, "positive_edge_fraction_b": {},
              "positive_edge_jaccard": {}}
    for tau in thresholds:
        key = f"{tau:g}"
        va, vb = np.isfinite(a), np.isfinite(b)
        pa, pb = va & (a >= tau), vb & (b >= tau)
        result["positive_edge_fraction_a"][key] = float(pa.sum()/va.sum()) if va.any() else None
        result["positive_edge_fraction_b"][key] = float(pb.sum()/vb.sum()) if vb.any() else None
        both = va & vb
        aa, bb = both & (a >= tau), both & (b >= tau)
        union = int(np.sum(aa | bb))
        result["positive_edge_jaccard"][key] = float(np.sum(aa & bb)/union) if union else 1.0
    return result


def density_metrics(corr, thresholds):
    if corr is None:
        return {f"{t:g}": None for t in thresholds}
    a = upper_triangle_values(corr)
    finite = np.isfinite(a)
    return {f"{t:g}": float(np.sum(finite & (a >= t))/finite.sum()) if finite.any() else None
            for t in thresholds}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, default=Path("data/interim/metadata.mat"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/stag_decomposition"))
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--radius", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    args = ap.parse_args()
    if not args.metadata.is_file():
        ap.error(f"找不到 metadata 文件: {args.metadata}")

    mat = loadmat(args.metadata, simplify_cells=True)
    pressure = np.asarray(mat["pressure"])
    n = pressure.shape[0]
    rec = vec(mat, "recordingId", np.int64)
    obj = vec(mat, "objectId", np.int64)
    valid = vec(mat, "hasValidLabel", bool)
    transition = vec(mat, "isTransition", bool)
    grasp_key = "isGrasp" if "isGrasp" in mat else ("isGrap" if "isGrap" in mat else None)
    if grasp_key is None:
        raise KeyError("metadata.mat 缺少 isGrasp/isGrap 事件标记")
    grasp = vec(mat, grasp_key, bool)
    transition_dir = vec(mat, "transitionDir", np.int64)
    split = vec(mat, "splitId", np.int64)
    names = [str(x) for x in np.asarray(mat["objects"], dtype=object).reshape(-1)]
    if any(x.size != n for x in (rec, obj, valid, transition, grasp, transition_dir, split)):
        raise ValueError("pressure 与 metadata 字段长度不一致")

    mask = active_sensor_mask(pressure)
    recording_rows, event_rows = [], []
    comparison_names = ["raw_valid_vs_event_raw", "event_raw_vs_decomposed",
                        "raw_valid_vs_decomposed", "handpose_vs_event_raw",
                        "handpose_vs_decomposed"]
    per_metric = {f"{name}_{mode}": [] for name in comparison_names
                  for mode in ("event", "recording")}
    per_density = {"raw_valid": [], "event_raw_event": [], "decomposed_event": [],
                   "event_raw_recording": [], "decomposed_recording": [], "handpose": []}
    totals = {"official_transitions": 0, "reliable_events": 0,
              "event_windows_event_norm": 0, "event_windows_recording_norm": 0,
              "expanded_object_frames_event_norm": 0,
              "expanded_object_frames_recording_norm": 0,
              "unique_object_frames_event_norm": 0,
              "unique_object_frames_recording_norm": 0,
              "handpose_baseline_frames": 0}

    for rid in np.unique(rec):
        ids = np.flatnonzero(rec == rid)
        cls_values = np.unique(obj[ids]); split_values = np.unique(split[ids])
        if len(cls_values) != 1 or len(split_values) != 1:
            raise ValueError(f"Recording {rid} 内 objectId/splitId 不唯一")
        cls = int(cls_values[0])
        if cls == 0:
            continue
        x = normalize_pressure(pressure[ids], mask)
        sm = smoothed_mean_pressure(x, sigma_frames=args.sigma)
        events = official_grasp_events(int(rid), transition[ids], transition_dir[ids], grasp[ids])
        reliable = [e for e in events if e.reliable]
        raw_x = x[valid[ids]]
        raw_corr = pairwise_pearson(raw_x) if raw_x.shape[0] >= 3 else None
        hand = np.stack([x[e.transition_index] for e in reliable]) if reliable else np.empty((0, x.shape[1]))
        hand_corr = pairwise_pearson(hand) if hand.shape[0] >= 3 else None
        per_density["raw_valid"].append(density_metrics(raw_corr, args.taus))
        per_density["handpose"].append(density_metrics(hand_corr, args.taus))
        row = {"recording_id": int(rid), "object_id": cls, "object_name": names[cls],
               "official_split_id": int(split_values[0]), "n_recording_frames": int(len(ids)),
               "n_hasValidLabel_frames": int(valid[ids].sum()),
               "n_official_transition_anchors": len(events), "n_reliable_isGrasp_events": len(reliable)}
        for mode in ("event", "recording"):
            dec = decompose_recording(x, sm, events, radius=args.radius,
                                      threshold=args.threshold, normalization=mode)
            obj_x = np.concatenate([d.object_pressure for d in dec], axis=0) if dec else np.empty((0, x.shape[1]))
            raw_window_x = np.concatenate([x[d.window_start:d.window_end+1] for d in dec], axis=0) if dec else np.empty((0, x.shape[1]))
            unique_frames = sorted({j for d in dec for j in range(d.window_start, d.window_end+1)})
            obj_corr = pairwise_pearson(obj_x) if len(obj_x) >= 3 else None
            window_raw_corr = pairwise_pearson(raw_window_x) if len(raw_window_x) >= 3 else None
            row[f"n_decomposed_events_{mode}"] = len(dec)
            row[f"n_decomposed_frames_expanded_{mode}"] = len(obj_x)
            row[f"n_decomposed_frames_unique_{mode}"] = len(unique_frames)
            row[f"n_decomposed_frames_overlap_hasValidLabel_{mode}"] = int(sum(valid[ids][j] for j in unique_frames))
            row[f"decomposed_corr_defined_{mode}"] = obj_corr is not None
            row[f"event_window_raw_corr_defined_{mode}"] = window_raw_corr is not None
            row[f"positive_edge_fraction_event_window_raw_{mode}"] = density_metrics(window_raw_corr, args.taus)
            row[f"positive_edge_fraction_decomposed_{mode}"] = density_metrics(obj_corr, args.taus)
            comparisons = {
                "raw_valid_vs_event_raw": corr_metrics(raw_corr, window_raw_corr, args.taus),
                "event_raw_vs_decomposed": corr_metrics(window_raw_corr, obj_corr, args.taus),
                "raw_valid_vs_decomposed": corr_metrics(raw_corr, obj_corr, args.taus),
            }
            if hand_corr is not None:
                comparisons["handpose_vs_event_raw"] = corr_metrics(hand_corr, window_raw_corr, args.taus)
                comparisons["handpose_vs_decomposed"] = corr_metrics(hand_corr, obj_corr, args.taus)
            for comp_name, metrics in comparisons.items():
                for metric_name in ("edge_spearman", "mean_abs_corr_difference", "finite_common_edges"):
                    row[f"{comp_name}_{mode}_{metric_name}"] = metrics[metric_name]
                for tau in args.taus:
                    key = f"{tau:g}"
                    row[f"{comp_name}_{mode}_jaccard_tau_{key}"] = metrics["positive_edge_jaccard"][key]
                if metrics["edge_spearman"] is not None:
                    per_metric[f"{comp_name}_{mode}"].append(metrics)
            per_density[f"event_raw_{mode}"].append(density_metrics(window_raw_corr, args.taus))
            per_density[f"decomposed_{mode}"].append(density_metrics(obj_corr, args.taus))
            for d in dec:
                event_rows.append({"recording_id": int(rid), "object_id": cls, "object_name": names[cls],
                    "event_index": d.event.event_index, "transition_index": d.event.transition_index,
                    "peak_index": d.event.peak_index, "transition_to_peak_frames": d.event.offset,
                    "window_start": d.window_start, "window_end": d.window_end,
                    "window_length": int(d.window_end-d.window_start+1), "normalization": mode,
                    "hasValidLabel_fraction_in_window": float(valid[ids][d.window_start:d.window_end+1].mean()),
                    "mean_absolute_object_pressure_change": float(np.mean(np.abs(d.object_pressure)))})
            totals[f"event_windows_{mode}_norm"] += len(dec)
            totals[f"expanded_object_frames_{mode}_norm"] += len(obj_x)
            totals[f"unique_object_frames_{mode}_norm"] += len(unique_frames)
        totals["official_transitions"] += len(events)
        totals["reliable_events"] += len(reliable)
        totals["handpose_baseline_frames"] += len(reliable)
        recording_rows.append(row)

    aggregates = {"matrix_comparisons": {}, "positive_edge_density": {}}
    for comp_name, values in per_metric.items():
        aggregates["matrix_comparisons"][comp_name] = {
            "edge_spearman": safe_summary([m["edge_spearman"] for m in values]),
            "mean_abs_corr_difference": safe_summary([m["mean_abs_corr_difference"] for m in values]),
            "n_recordings_with_comparable_matrices": len(values)}
        for tau in args.taus:
            key = f"{tau:g}"
            aggregates["matrix_comparisons"][comp_name][f"positive_edge_jaccard_tau_{key}"] = safe_summary(
                [m["positive_edge_jaccard"][key] for m in values])
    for matrix_name, values in per_density.items():
        aggregates["positive_edge_density"][matrix_name] = {}
        for tau in args.taus:
            key = f"{tau:g}"
            aggregates["positive_edge_density"][matrix_name][key] = safe_summary(
                [v[key] for v in values if v[key] is not None])

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "stag_decomposition_recording_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(recording_rows[0])); w.writeheader(); w.writerows(recording_rows)
    with (out / "stag_decomposition_event_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(event_rows[0])); w.writeheader(); w.writerows(event_rows)
    report = {
        "protocol": {
            "pressure_normalization": "clip((raw - 500) / 150, 0, 1) on 548 active sensor cells",
            "mean_signal": "per-frame mean over 548 active sensors",
            "smoothing": f"symmetric Gaussian sigma={args.sigma} frames (reflect boundary)",
            "event_anchors": "official metadata: isTransition at pre-contact frame; transitionDir is forward offset; isGrasp/isGrap selects reliable event anchors",
            "object_signal": "normalized pressure frame minus normalized pre-contact baseline frame",
            "window_radius": args.radius, "window_threshold": args.threshold,
            "window_normalization": {"event": "event-relative rise (signal - transition value) / (peak value - transition value); primary operational interpretation",
                                     "recording": "full-recording min-max normalization; sensitivity analysis because the paper does not specify the denominator for 0.5 threshold"},
            "comparisons": "all hasValidLabel raw frames vs raw event-window frames; raw event-window frames vs baseline-subtracted object-pressure frames; all hasValidLabel raw frames vs baseline-subtracted frames; event-baseline hand-pose samples vs raw/decomposed event-window matrices",
            "caveat": "The paper's event-detector implementation is not released. Official event markers are used to reproduce its event anchors; the 0.5 frame-validity normalization is under-specified and both interpretations are reported.",
            "spearman_edge_sampling": "deterministic sample of up to 20,000 common sensor pairs per Recording; mean absolute differences and edge densities/Jaccard use all finite pairs"},
        "counts": {"n_objects": len(set(r["object_id"] for r in recording_rows)),
                   "n_recordings": len(recording_rows), "n_sensors": int(mask.sum()), **totals},
        "recording_level_metrics": aggregates,
        "window_length_event_normalization": safe_summary([r["window_length"] for r in event_rows if r["normalization"] == "event"]),
        "window_length_recording_normalization": safe_summary([r["window_length"] for r in event_rows if r["normalization"] == "recording"]),
        "recordings": recording_rows}
    (out / "stag_decomposition_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "window_length_event": report["window_length_event_normalization"],
                      "window_length_recording": report["window_length_recording_normalization"],
                      "metrics": aggregates}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
