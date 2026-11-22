#!/usr/bin/env python3
"""Audit STAG Weight pressure-only grasp variance and covariance-estimator sensitivity.

No model is trained. Statistical/inferential unit: physical object.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, logging, platform, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.covariance import LedoitWolf, OAS

ESTIMATORS = ("pearson", "ledoit_wolf", "oas")
WINDOWS = ("future_after_5", "full_valid_sensitivity")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def correlation_and_covariance(x: np.ndarray, estimator: str):
    if estimator == "pearson":
        cov = np.cov(x, rowvar=False)
        shrinkage = None
    else:
        model = (LedoitWolf() if estimator == "ledoit_wolf" else OAS()).fit(x)
        cov = model.covariance_
        shrinkage = float(model.shrinkage_)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, np.nan)
    return corr, shrinkage, cov


def condition_positive_spectrum(cov: np.ndarray) -> float:
    # For sample covariance p>n, zero eigenvalues are expected. Report the
    # condition ratio over positive eigenvalues above 1e-12, not full rank.
    eig = np.linalg.eigvalsh((cov + cov.T) * 0.5)
    positive = eig[eig > 1e-12]
    return float(positive.max() / positive.min()) if len(positive) else float("inf")


def edge_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    tri = np.triu_indices(a.shape[0], 1)
    x, y = a[tri], b[tri]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        rho = pr = mad = float("nan")
    else:
        rho = float(spearmanr(x, y).statistic)
        pr = float(pearsonr(x, y).statistic)
        mad = float(np.mean(np.abs(x - y)))
    out = {"spearman": rho, "pearson": pr, "mean_abs_diff": mad, "n_edges": int(len(x))}
    for threshold in (0.5, 0.7, 0.9):
        left, right = x >= threshold, y >= threshold
        union = int(np.sum(left | right))
        out[f"jaccard_r_ge_{threshold:.1f}"] = float(np.sum(left & right) / union) if union else float("nan")
        out[f"union_count_r_ge_{threshold:.1f}"] = union
    return out


def bootstrap_mean(values: np.ndarray, reps: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(reps)])
    return {"n_objects": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, required=True, help="Path to extracted metadata.mat")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261122)
    ap.add_argument("--sensor-seed", type=int, default=20261116)
    ap.add_argument("--nodes", type=int, default=96)
    ap.add_argument("--archive-sha256", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Validate schema and print inventory only")
    args = ap.parse_args()
    if args.bootstrap_reps < 100:
        ap.error("--bootstrap-reps must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output_dir / "run.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("weight_audit")
    log.info("START metadata=%s", args.metadata)
    raw = loadmat(args.metadata, simplify_cells=True)
    required = ("pressure", "recordingId", "objectId", "ts", "graspIndex", "graspValid", "objects")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing metadata fields: {missing}")
    pressure = np.asarray(raw["pressure"])
    recording_id = np.asarray(raw["recordingId"]).ravel().astype(int)
    object_id = np.asarray(raw["objectId"]).ravel().astype(int)
    timestamp = np.asarray(raw["ts"]).ravel().astype(float)
    grasp_index = np.asarray(raw["graspIndex"]).ravel().astype(int)
    grasp_valid = np.asarray(raw["graspValid"]).ravel().astype(bool)
    object_names = np.asarray(raw["objects"]).ravel()
    if pressure.shape != (len(recording_id), 32, 32) or pressure.dtype != np.int16:
        raise ValueError(f"Unexpected pressure array {pressure.shape}, {pressure.dtype}")
    if not np.isfinite(timestamp).all():
        raise ValueError("Non-finite timestamps")
    mask = np.nanstd(pressure.astype(np.float32), axis=0) > 0
    active = np.flatnonzero(mask.ravel())
    if len(active) != 548:
        raise ValueError(f"Expected 548 active cells, found {len(active)}")
    selected = np.sort(np.random.default_rng(args.sensor_seed).choice(len(active), args.nodes, replace=False))
    flat = active[selected]
    sensor_doc = {"seed": args.sensor_seed, "active_sensor_count": len(active),
                  "selected_sensor_count": args.nodes, "selected_flat_indices": flat.tolist(),
                  "selected_xy": [[int(i % 32), int(i // 32)] for i in flat],
                  "sha256": hashlib.sha256(np.asarray(flat, dtype=np.int64).tobytes()).hexdigest()}
    (args.output_dir / "sensor_selection.json").write_text(json.dumps(sensor_doc, indent=2), encoding="utf-8")
    inventory = {"metadata_sha256": sha256_file(args.metadata), "pressure_shape": list(pressure.shape),
                 "pressure_dtype": str(pressure.dtype), "pressure_min": int(pressure.min()),
                 "pressure_max": int(pressure.max()), "nan_count": int(np.isnan(pressure).sum()),
                 "inf_count": int(np.isinf(pressure).sum()), "recordings": int(len(np.unique(recording_id))),
                 "physical_objects": int(len(set(object_id) - {0})), "valid_frames": int(grasp_valid.sum()),
                 "active_sensor_count": int(len(active)), "archive_sha256": args.archive_sha256}
    (args.output_dir / "input_manifest.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(inventory, indent=2))
        log.info("DRY RUN COMPLETE")
        return
    if len(set(object_id) - {0}) != 26:
        raise ValueError("Expected 26 physical objects")
    groups, variance_rows = {}, []
    for oid in sorted(set(object_id) - {0}):
        name = str(object_names[oid])
        for gi in range(6):
            ix = np.flatnonzero((object_id == oid) & (grasp_index == gi) & grasp_valid)
            ix = ix[np.argsort(timestamp[ix], kind="stable")]
            if len(ix) < 6:
                raise ValueError(f"Too few valid frames: object={name}, grasp={gi}, n={len(ix)}")
            raw_x = pressure[ix].reshape(len(ix), -1)[:, flat].astype(np.float64)
            x = np.clip((raw_x - 500) / 150, 0, 1)
            future = x[5:]
            var = np.var(future, axis=0, ddof=1) if len(future) > 1 else np.zeros(args.nodes)
            raw_var = np.var(raw_x[5:], axis=0, ddof=1) if len(future) > 1 else np.zeros(args.nodes)
            participation = (float(var.sum() ** 2 / np.sum(var ** 2)) if np.sum(var ** 2) > 0 else float("nan"))
            row = {"object_id": int(oid), "object_name": name, "grasp_index": gi,
                   "valid_frames": int(len(ix)), "future_frames": int(len(future)),
                   "raw_mean": float(raw_x.mean()), "raw_sd": float(raw_x.std(ddof=1)),
                   "transformed_mean": float(x.mean()), "transformed_sd": float(x.std(ddof=1)),
                   "future_mean": float(future.mean()), "future_sd": float(future.std(ddof=1)),
                   "future_sensor_nonzero_variance_fraction": float(np.mean(var > 0)),
                   "future_sensor_variance_gt_1e-6_fraction": float(np.mean(var > 1e-6)),
                   "future_effective_sensor_count": participation,
                   "future_raw_sensor_nonzero_variance_fraction": float(np.mean(raw_var > 0))}
            for threshold in (.01, .02, .05):
                row[f"future_contact_fraction_gt_{threshold:.2f}"] = float(np.mean(future > threshold))
            variance_rows.append(row)
            groups[(oid, gi)] = {"x": x, "future": future, "timestamp": timestamp[ix]}
    variance_df = pd.DataFrame(variance_rows)
    variance_df.to_csv(args.output_dir / "variance_by_grasp.csv", index=False)
    coverage_rows = []
    for oid, group in variance_df.groupby("object_id"):
        row = {"object_id": int(oid), "object_name": group.object_name.iloc[0],
               "n_grasps": len(group), "valid_frames_total": int(group.valid_frames.sum()),
               "future_frames_total": int(group.future_frames.sum()),
               "valid_frames_median": float(group.valid_frames.median()),
               "future_frames_median": float(group.future_frames.median())}
        for col in ("future_sensor_nonzero_variance_fraction", "future_sensor_variance_gt_1e-6_fraction",
                    "future_effective_sensor_count", "future_contact_fraction_gt_0.01",
                    "future_contact_fraction_gt_0.02", "future_contact_fraction_gt_0.05"):
            row[col + "_median"] = float(group[col].median())
            row[col + "_min"] = float(group[col].min())
            row[col + "_max"] = float(group[col].max())
        coverage_rows.append(row)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "coverage_by_object.csv", index=False)
    log.info("PART A COMPLETE: %d grasp intervals", len(variance_df))

    partition_rows = []
    for oid in sorted(set(object_id) - {0}):
        name = str(object_names[oid])
        g = {j: groups[(oid, j)] for j in range(6)}
        splits = [(A, tuple(j for j in range(6) if j not in A))
                  for A in itertools.combinations(range(6), 3) if 0 in A]
        if len(splits) != 10:
            raise RuntimeError("Expected ten unique partitions per object")
        for split_idx, (A, B) in enumerate(splits):
            for window in WINDOWS:
                XA = np.concatenate([g[j]["future" if window == "future_after_5" else "x"] for j in A])
                XB = np.concatenate([g[j]["future" if window == "future_after_5" else "x"] for j in B])
                for estimator in ESTIMATORS:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        CA, shrinkA, covA = correlation_and_covariance(XA, estimator)
                        CB, shrinkB, covB = correlation_and_covariance(XB, estimator)
                    metrics = edge_metrics(CA, CB)
                    partition_rows.append({"object_id": int(oid), "object_name": name,
                        "split_index": split_idx, "group_A": ";".join(map(str, A)),
                        "group_B": ";".join(map(str, B)), "window_protocol": window,
                        "estimator": estimator, "frames_A": len(XA), "frames_B": len(XB),
                        "shrinkage_A": shrinkA, "shrinkage_B": shrinkB,
                        "condition_number_A": condition_positive_spectrum(covA),
                        "condition_number_B": condition_positive_spectrum(covB),
                        "warnings": "; ".join(str(w.message) for w in caught), **metrics})
        log.info("PART B object complete: %s", name)
    part_df = pd.DataFrame(partition_rows)
    part_df.to_csv(args.output_dir / "partition_metrics.csv", index=False)
    object_rows = []
    measure_cols = ("spearman", "pearson", "mean_abs_diff", "jaccard_r_ge_0.5",
                    "jaccard_r_ge_0.7", "jaccard_r_ge_0.9", "condition_number_A",
                    "condition_number_B", "shrinkage_A", "shrinkage_B")
    for (oid, name, window, estimator), group in part_df.groupby(
            ["object_id", "object_name", "window_protocol", "estimator"]):
        row = {"object_id": int(oid), "object_name": name, "window_protocol": window,
               "estimator": estimator, "n_partitions": len(group)}
        for col in measure_cols:
            vals = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[col + "_mean"] = float(vals.mean()) if len(vals) else None
            row[col + "_median"] = float(vals.median()) if len(vals) else None
        object_rows.append(row)
    object_df = pd.DataFrame(object_rows)
    object_df.to_csv(args.output_dir / "object_summary.csv", index=False)
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = {}
    for (window, estimator), group in object_df.groupby(["window_protocol", "estimator"]):
        bootstrap.setdefault(window, {})[estimator] = {}
        for metric in ("spearman_mean", "pearson_mean", "mean_abs_diff_mean",
                       "jaccard_r_ge_0.5_mean", "jaccard_r_ge_0.7_mean", "jaccard_r_ge_0.9_mean"):
            values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            result = bootstrap_mean(values, args.bootstrap_reps, rng)
            if result is not None:
                bootstrap[window][estimator][metric] = result
    (args.output_dir / "bootstrap_summary.json").write_text(json.dumps(
        {"bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": args.bootstrap_seed,
         "unit": "physical object; n=26", "summaries": bootstrap}, indent=2), encoding="utf-8")
    diag_rows = []
    for (window, estimator), group in part_df.groupby(["window_protocol", "estimator"]):
        row = {"window_protocol": window, "estimator": estimator, "n_partition_pairs": len(group),
               "warning_rows": int(group.warnings.fillna("").astype(str).str.len().gt(0).sum()),
               "mean_finite_edges": float(pd.to_numeric(group.n_edges).mean())}
        for col in ("shrinkage_A", "shrinkage_B", "condition_number_A", "condition_number_B"):
            values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                row[col + "_median"] = float(values.median())
                row[col + "_p95"] = float(values.quantile(.95))
        diag_rows.append(row)
    pd.DataFrame(diag_rows).to_csv(args.output_dir / "estimator_diagnostics.csv", index=False)
    # Object-level variance/coverage bootstrap (separate RNG stream).
    variance_summary = {}
    variance_rng = np.random.default_rng(args.bootstrap_seed)
    for metric in ("valid_frames", "future_frames", "future_sensor_nonzero_variance_fraction",
                   "future_sensor_variance_gt_1e-6_fraction", "future_effective_sensor_count",
                   "future_contact_fraction_gt_0.01", "future_contact_fraction_gt_0.02",
                   "future_contact_fraction_gt_0.05"):
        values = variance_df.groupby("object_id")[metric].median().to_numpy(dtype=float)
        boot = np.asarray([variance_rng.choice(values, size=len(values), replace=True).mean()
                           for _ in range(args.bootstrap_reps)])
        variance_summary[metric] = {
            "n_objects": int(len(values)), "mean_of_object_medians": float(values.mean()),
            "median_of_object_medians": float(np.median(values)),
            "object_median_iqr": [float(np.quantile(values, .25)), float(np.quantile(values, .75))],
            "object_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        }
    (args.output_dir / "variance_bootstrap_summary.json").write_text(
        json.dumps(variance_summary, indent=2), encoding="utf-8")

    # Paired object-level estimator differences versus sample Pearson.
    paired = {}
    primary = object_df[object_df.window_protocol == "future_after_5"]
    base = primary[primary.estimator == "pearson"].set_index("object_id")
    for estimator in ("ledoit_wolf", "oas"):
        current = primary[primary.estimator == estimator].set_index("object_id")
        paired[estimator] = {}
        for metric in ("spearman_mean", "mean_abs_diff_mean",
                       "jaccard_r_ge_0.5_mean", "jaccard_r_ge_0.7_mean"):
            diff = (current[metric] - base[metric]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            boot = np.asarray([rng.choice(diff, size=len(diff), replace=True).mean()
                               for _ in range(args.bootstrap_reps)])
            paired[estimator][metric] = {
                "n_objects": int(len(diff)), "mean_difference": float(diff.mean()),
                "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
            }
    (args.output_dir / "paired_estimator_differences.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8")
    config = {"metadata": str(args.metadata), "metadata_sha256": sha256_file(args.metadata),
              "sensor_seed": args.sensor_seed, "nodes": args.nodes, "pressure_transform": "clip((raw-500)/150,0,1)",
              "grasp_indices": list(range(6)), "early_input_valid_frames": 5,
              "window_protocols": list(WINDOWS), "estimators": list(ESTIMATORS),
              "partitions_per_object": 10, "bootstrap_reps": args.bootstrap_reps,
              "bootstrap_seed": args.bootstrap_seed, "model_training": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
           "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
           "note": "No model training; training trace not applicable."}
    (args.output_dir / "environment.txt").write_text(json.dumps(env, indent=2), encoding="utf-8")
    log.info("COMPLETE: %d partition rows; bootstrap reps=%d", len(part_df), args.bootstrap_reps)
    print(f"Completed Weight audit: {len(part_df)} partition rows; outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
