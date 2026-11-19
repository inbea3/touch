#!/usr/bin/env python3
"""Reproduce the G0 STAG pressure-only sensor spatial topology audit."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SEED = 20261116

def graph_stats(xy, k):
    d = cdist(xy.astype(float), xy.astype(float))
    np.fill_diagonal(d, np.inf)
    kth = np.partition(d, k-1, axis=1)[:, k-1]
    rows, cols = np.where(d <= kth[:, None] + 1e-9)  # include ties
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xy), len(xy)))
    A = ((A + A.T) > 0).astype(np.int8)
    A.setdiag(0); A.eliminate_zeros()
    nc, labels = connected_components(A, directed=False, return_labels=True)
    degree = np.asarray(A.sum(axis=1)).ravel()
    r, c = A.nonzero()
    lengths = np.sqrt(((xy[r] - xy[c]) ** 2).sum(axis=1))
    lengths = lengths[r < c]
    return {"k": int(k), "n_nodes": int(len(xy)), "n_undirected_edges": int(A.nnz // 2),
            "n_components": int(nc), "largest_component_nodes": int(np.bincount(labels).max()),
            "isolated_nodes": int((degree == 0).sum()),
            "degree_min": int(degree.min()), "degree_median": float(np.median(degree)),
            "degree_mean": float(degree.mean()), "degree_max": int(degree.max()),
            "edge_distance_px": {"min": float(lengths.min()), "p25": float(np.quantile(lengths,.25)),
                "median": float(np.median(lengths)), "p75": float(np.quantile(lengths,.75)),
                "p95": float(np.quantile(lengths,.95)), "max": float(lengths.max())},
            "edge_weight_sigma_px": float(np.median(lengths)), "adjacency": A}

def fps_sample(xy, n=96, seed=SEED):
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(xy)))]
    nearest = cdist(xy, xy[chosen]).ravel()
    while len(chosen) < n:
        nxt = int(np.argmax(nearest)); chosen.append(nxt)
        nearest = np.minimum(nearest, cdist(xy, xy[[nxt]]).ravel())
    return np.array(sorted(chosen), dtype=int)

def coverage(full, subset):
    nearest = cdist(full.astype(float), subset.astype(float)).min(axis=1)
    out = {"n_selected": int(len(subset)), "nearest_full_to_selected_px": {
        "mean": float(nearest.mean()), "median": float(np.median(nearest)),
        "p90": float(np.quantile(nearest,.9)), "p95": float(np.quantile(nearest,.95)),
        "max": float(nearest.max())}}
    for bins in (4, 8):
        cells = np.minimum(subset // (32 // bins), bins-1)
        counts = np.bincount(cells[:,0] * bins + cells[:,1], minlength=bins*bins)
        out[f"occupancy_{bins}x{bins}"] = {"occupied_cells": int((counts>0).sum()),
            "empty_cells": int((counts==0).sum()), "min_count": int(counts.min()),
            "max_count": int(counts.max()), "cv": float(counts.std()/counts.mean())}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    mat = loadmat(a.metadata, simplify_cells=True)
    pressure = np.asarray(mat["pressure"])
    mask = np.nanstd(pressure.astype(np.float32), axis=0) > 0
    if int(mask.sum()) != 548:
        raise RuntimeError(f"Expected 548 active cells, got {int(mask.sum())}")
    coords = np.argwhere(mask)  # grid row y, column x
    random_idx = np.sort(np.random.default_rng(SEED).choice(len(coords),96,replace=False))
    random96 = coords[random_idx]
    fps96 = coords[fps_sample(coords)]
    groups = {"full_548": coords, "legacy_random_96": random96, "spatially_balanced_96_fps": fps96}
    result = {"dataset": {"pressure_shape": list(pressure.shape), "active_sensor_count": int(mask.sum()),
              "grid_shape": [32,32], "active_coordinate_order": "row,column (y,x)",
              "legacy_random_seed": SEED}, "sampling": {}, "graphs": {}}
    result["sampling"]["legacy_random_96"] = coverage(coords, random96)
    result["sampling"]["spatially_balanced_96_fps"] = coverage(coords, fps96)
    for name, xy in groups.items():
        result["graphs"][name] = {}
        flat = xy[:,0]*32 + xy[:,1]
        np.savetxt(a.output_dir/f"nodes_{name}.csv",
            np.column_stack([np.arange(len(xy)),flat,xy]), fmt="%d", delimiter=",",
            header="node_index,flat_sensor_index,grid_row_y,grid_col_x", comments="")
        for k in (4,6,8):
            g = graph_stats(xy,k); A = g.pop("adjacency")
            result["graphs"][name][str(k)] = g
            r,c = A.nonzero(); keep = r < c
            dist = np.sqrt(((xy[r[keep]]-xy[c[keep]])**2).sum(axis=1))
            sigma = float(np.median(dist)); weight = np.exp(-(dist**2)/(2*sigma**2))
            np.savetxt(a.output_dir/f"edges_{name}_knn{k}.csv",
                np.column_stack([r[keep],c[keep],dist,weight]),
                fmt=["%d","%d","%.8f","%.8f"], delimiter=",",
                header="source,target,distance_px,gaussian_weight", comments="")
    rows = []
    for seed in range(1000,1100):
        idx = np.sort(np.random.default_rng(seed).choice(len(coords),96,replace=False))
        xy = coords[idx]; cov = coverage(coords,xy); g = graph_stats(xy,6)
        rows.append({"seed":seed,"coverage_mean_px":cov["nearest_full_to_selected_px"]["mean"],
            "coverage_p95_px":cov["nearest_full_to_selected_px"]["p95"],
            "coverage_max_px":cov["nearest_full_to_selected_px"]["max"],
            "occupied_8x8_cells":cov["occupancy_8x8"]["occupied_cells"],
            "k6_edge_median_px":g["edge_distance_px"]["median"],
            "k6_edge_p95_px":g["edge_distance_px"]["p95"],"k6_components":g["n_components"]})
    result["random_96_seed_sensitivity"] = {}
    for key in rows[0]:
        if key == "seed": continue
        vals = np.array([x[key] for x in rows],float)
        result["random_96_seed_sensitivity"][key] = {"median":float(np.median(vals)),
            "p05":float(np.quantile(vals,.05)),"p95":float(np.quantile(vals,.95))}
    with (a.output_dir/"random_96_seed_sensitivity.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result["interpretation_notes"] = [
        "Spatial edges are coordinate-derived priors, not co-activation labels.",
        "k-NN includes all ties at the kth-neighbor radius and uses union symmetrization.",
        "The farthest-point 96-node subset is a geometry-only sensitivity condition.",
        "No future pressure values or co-activation targets were used to construct topology."]
    (a.output_dir/"g0_spatial_topology_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig, axes = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    axes[0].scatter(coords[:,1],coords[:,0],s=11); axes[0].set_title("All active sensors (548)")
    axes[1].scatter(coords[:,1],coords[:,0],s=7,alpha=.18); axes[1].scatter(random96[:,1],random96[:,0],s=22)
    axes[1].set_title("Legacy random subset (96)")
    axes[2].scatter(coords[:,1],coords[:,0],s=7,alpha=.18); axes[2].scatter(fps96[:,1],fps96[:,0],s=22)
    axes[2].set_title("Farthest-point subset (96)")
    for ax in axes:
        ax.set_xlim(-1,32); ax.set_ylim(32,-1); ax.set_aspect("equal")
        ax.set_xticks(range(0,32,4)); ax.set_yticks(range(0,32,4)); ax.grid(alpha=.2)
        ax.set_xlabel("Grid column (x)"); ax.set_ylabel("Grid row (y)")
    fig.suptitle("STAG pressure-only sensor spatial coverage audit")
    fig.savefig(a.output_dir/"g0_sensor_spatial_coverage.png",dpi=180)
    sha=hashlib.sha256(a.metadata.read_bytes()).hexdigest()
    (a.output_dir/"input_sha256.txt").write_text(sha+"  metadata.mat\n")
    print(json.dumps(result,indent=2))
if __name__ == "__main__": main()
