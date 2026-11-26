# G2-R3 Reconstruction Model Specification — 2026-11-24

Status: architecture frozen for implementation; smoke test passed at module level; full end-to-end script smoke run pending.
Experiment ID: G2-R3-RECONSTRUCTION-20261124

## Dataset and target
- Input archive: classification_lite.zip; SHA-256 ce0f27ae87e8ce2ec405f756dfe278b0b8011ef12e9de93e468b7ef31ad76c58.
- metadata.mat SHA-256: 19062ed3415df6e21ae055c5149315a5350652a27d59ab55fed74e395f2f1378.
- 78 recordings, 26 non-empty objects, 548 active pressure cells.
- Each recording: earliest five hasValidLabel frames as input; all later hasValidLabel frames form target period.
- Pressure transform: clip((p-500)/150,0,1).
- Target: Pearson correlation for all valid unordered sensor pairs in the future period; 149,878 possible pairs per recording. Constant target channels produce invalid pairs and are excluded consistently for all conditions.

## Node features
Ten dimensions per sensor:
1. early-window mean;
2. early-window standard deviation;
3. early-window minimum;
4. early-window maximum;
5. early-window range;
6. timestamp-based early-window linear slope;
7. last early pressure;
8. first early pressure;
9. normalized sensor row coordinate;
10. normalized sensor column coordinate.

Only the first eight pressure-derived dimensions are standardized, with mean/SD fitted on outer-training objects only. Coordinates are fixed in [0,1].

## Five conditions
1. Node-MLP: node encoder only; no message passing.
2. Spatial-GNN k6: two residual normalized-adjacency message-passing layers.
3–5. Rewired-GNN 1–3: identical architecture using the three degree-preserving, connected rewired graphs.

Node encoder: Linear(10,64) -> ReLU -> Linear(64,64) -> ReLU.
Each graph layer: ReLU(Linear(A_norm @ H) + H), with two layers and shared node feature/decoder definitions.
Undirected pair decoder: concatenate [h_i+h_j, abs(h_i-h_j), h_i*h_j, 5-D pair geometry], then Linear(197,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,1) -> Tanh.
The pair representation is invariant to swapping i and j. Pair geometry contains normalized coordinate sums, absolute row/column offsets and normalized Euclidean distance.

## Graph construction
- Active coordinates: numpy.argwhere(mask), row-major.
- Spatial graph: six nearest neighbors per node, Euclidean grid distance, union symmetrization, no self loops.
- Corrected graph: 548 nodes, 1,966 undirected edges, degree range 6–10, median 7, one connected component.
- Edge-list hash (sorted undirected pairs, little-endian int32): df15364f3efbec0fd22ff192768bd19c2c5dee8597c2a2150ba19ed51709229c.
- Rewiring: NetworkX 3.6.1 connected_double_edge_swap, requested 10×E swaps, seeds 2026112201/02/03. Actual successful swaps: 19226/19215/19186. Each graph has 1,966 edges, preserves degree sequence and is connected.
- Rewired edge hashes (little-endian int32 sorted pairs):
  - 2026112201: d26d64dcae978a47e374b2a538c73708ca7d91ab219436675d17863ed92a45b1
  - 2026112202: e46d9b65f8ce52c989e422a1881aabd8b2e5554959e8738b6c111e462a5f179d
  - 2026112203: 882a8953759549a99b451895a548a4a203244e1171a448e8b481571eafca4fb1

## Training
- AdamW, learning rate 5e-4, weight decay 1e-3.
- Smooth L1 loss.
- At each epoch sample up to 8,192 valid sensor pairs per training recording, without replacement.
- Max 100 epochs, early stopping patience 15, min_delta 1e-5, based on validation-object-macro MAE.
- Evaluation processes all valid pairs in chunks of 32,768.
- Outer split: five object-grouped folds for each of three split seeds 20261119/20/21.
- Internal validation: random 20% of outer-training objects, seeded deterministically from split seed and fold. Remaining objects train.
- Initialization seeds 20261123/24/25.
- Five conditions × 5 folds × 3 split seeds × 3 initialization seeds = 225 fits.
- Primary metric: object-macro MAE. Secondary: object-macro RMSE and Spearman. Recording metrics are averaged within object; objects are the inferential unit.

## Deviations and limitations
- Historical G2-R3 exact source and edge list are unavailable; this is a new, explicit reconstruction, not a bitwise reproduction.
- The previous preliminary graph count of 1,925 edges was incorrect; direct recomputation gives 1,966. Use the corrected count and hashes above.
- The pairwise target is a future within-recording correlation matrix; it is not causal ground truth or a physical contact graph.
- No validation/test object is used for scaler fitting. Test is evaluated only after each fit's checkpoint is selected using internal validation.
- Keep per-fit configs, training_trace.csv, checkpoint, per-recording/per-object metrics, environment, run.log and SHA-256 manifest. No raw pressure array is uploaded.
