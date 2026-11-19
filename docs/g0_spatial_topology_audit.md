# G0: STAG Pressure-only Spatial Topology Audit

**Status:** completed; topology-only audit, no predictive model trained  
**Dataset:** classification_lite.zip / metadata.mat  
**Legacy 96-node sampling seed:** 20261116

## 1. Scope and leakage boundary

This audit checks whether the 548 active pressure-sensor locations support a spatial graph and whether the existing random 96-sensor subset preserves local geometry. It uses only static 32 x 32 sensor coordinates inferred from nonzero temporal variation in the pressure matrix. It does not use hasValidLabel, future pressure targets, correlation labels, object identities, or train/validation/test labels to construct edges.

The active mask follows the established project definition, nanstd(pressure, axis=0) > 0, yielding 548 cells from a pressure array of shape (135187, 32, 32). Coordinates are grid row/column indices, not calibrated physical distances in millimeters. The mask is estimated from the complete pressure archive; although it is unsupervised and label-free, a later strict inductive pipeline should freeze this hardware mask as dataset metadata or document the global preprocessing explicitly.

## 2. Graph construction

- Euclidean distance on the 32 x 32 grid.
- k-nearest-neighbor candidates for k = 4, 6, 8.
- Include all nodes tied at the kth-neighbor distance, avoiding arbitrary row-major tie-breaking on the regular grid.
- Symmetrize by the union rule: retain an undirected edge if either endpoint selects the other.
- Export Gaussian distance weights exp(-d^2 / (2 sigma^2)), with sigma equal to the median undirected edge length for that graph.

These edges are spatial priors only. They are not co-activation labels and do not constrain the model's output pairs.

## 3. Connectivity and locality

| Node set | k | Undirected edges | Components | Degree min / median / max | Edge distance median / p95 / max (grid px) |
|---|---:|---:|---:|---:|---:|
| All 548 | 4 | 1,395 | 1 | 4 / 5 / 8 | 1.000 / 1.414 / 2.000 |
| All 548 | 6 | 2,226 | 1 | 6 / 8 / 11 | 1.414 / 2.000 / 2.236 |
| All 548 | 8 | 2,329 | 1 | 8 / 8 / 14 | 1.414 / 2.000 / 2.828 |
| Legacy random 96 | 4 | 243 | 1 | 4 / 5 / 8 | 2.236 / 5.000 / 7.810 |
| Legacy random 96 | 6 | 373 | 1 | 6 / 7.5 / 12 | 3.162 / 5.631 / 8.062 |
| Legacy random 96 | 8 | 486 | 1 | 8 / 10 / 15 | 3.606 / 7.053 / 9.055 |
| Spatially balanced FPS 96 | 4 | 237 | 1 | 4 / 5 / 7 | 3.000 / 4.472 / 7.071 |
| Spatially balanced FPS 96 | 6 | 341 | 1 | 6 / 7 / 11 | 3.162 / 5.385 / 8.544 |
| Spatially balanced FPS 96 | 8 | 459 | 1 | 8 / 9 / 13 | 4.000 / 6.708 / 11.402 |

All tested graphs are connected and have no isolated nodes. The full 548-node graph is substantially more local. At k=6, its median edge length is 1.414 grid pixels and p95 is 2.000; the legacy 96-node graph has median 3.162 and p95 5.631. Thus, reducing to 96 randomly selected nodes preserves connectivity but changes the scale of neighborhood aggregation.

## 4. Does the existing random 96-node subset cover the array?

| 96-node subset | Mean nearest distance from each of 548 sensors | p95 | Maximum | Occupied 8 x 8 cells |
|---|---:|---:|---:|---:|
| Legacy random (seed 20261116) | 1.222 | 2.828 | 4.123 | 40 / 64 |
| Geometry-only farthest-point sample | 0.962 | 2.000 | 2.000 | 46 / 64 |

A 100-seed sensitivity audit of random 96-node samples found:

- Median mean nearest-sensor coverage distance: 1.199 px (5th-95th percentile: 1.140-1.278).
- Median maximum coverage distance: 4.000 px (5th-95th percentile: 3.162-5.407).
- Median number of occupied 8 x 8 cells: 40 (5th-95th percentile: 37-43).
- The k=6 graph remained connected in all 100 random samples.
- Median k=6 graph edge distance: 3.162 px; median p95 edge distance: 5.831 px.

The farthest-point subset improves geometric coverage, but it does not restore the very local neighborhood scale available when all 548 sensors are retained. It is a useful sensitivity subset, not a substitute for the full sensor graph.

## 5. Decision for the next stage

1. **Primary GNN topology candidate:** build the spatial graph on all 548 active sensors. The full graph is connected and retains local neighborhoods at approximately 1-2 grid pixels.
2. **Controlled 96-node comparison:** retain the existing random 96-sensor subset for a strict same-node comparison against the current Edge-MLP baseline, while describing it as a coarser spatial graph.
3. **Sampling sensitivity:** include the geometry-only farthest-point 96-node subset as a separate sensitivity condition; do not select it using prediction labels.
4. **Candidate k:** carry k=4, 6, 8 forward to G1. k=6 is a reasonable central candidate, not a selected winner. The choice must be evaluated on validation objects only.
5. **Output space:** future relation prediction must still cover all unordered pairs among selected nodes, n(n-1)/2, not only spatial graph edges. Spatial adjacency is an encoder prior, not the target edge set.

For a fair architecture claim, the next comparison should keep the same node set, input features, target labels, object-level folds, and training budget across Node-MLP and Spatial-GNN. The full-548 setting should be reported as a separate scale/coverage condition and should have a matching non-graph baseline if computationally feasible.

## 6. Reproduction

Run scripts/audit_sensor_spatial_graph.py with the metadata.mat path and an output directory. The script writes the JSON summary, 100-seed random-subset sensitivity CSV, sensor coverage plot, node-index mappings, and weighted edge lists for 548-node, legacy random-96, and geometry-balanced FPS-96 graphs at k=4/6/8. Raw pressure arrays are not written to the result directory.
