# G2-R3 Reconstruction Deviations and Graph Freeze

Date: 2026-11-24
Experiment ID: G2-R3-RECONSTRUCTION-20261124
Status: graph topology frozen; model architecture and training pipeline still pending.

## Reconstructed spatial graph
- Coordinates: 548 active cells in row-major (row, column) order, from `active_sensor_mask` in `src/touch_research/decomposition/stag.py`.
- Distance: Euclidean grid distance.
- Directed neighbor selection: six nearest other active sensors per node.
- Undirected conversion: union symmetrization, no self-loops.
- Graph: 548 nodes, 1,966 undirected edges, degree range 6–10, median 7, connected.
- Edge-list SHA-256: df15364f3efbec0fd22ff192768bd19c2c5dee8597c2a2150ba19ed51709229c.

## Rewiring algorithm
- Library: NetworkX 3.6.1.
- Function: `networkx.connected_double_edge_swap`.
- Swap budget: 10 × original edge count = 19,250 successful swaps requested.
- Seeds: 2026112201, 2026112202, 2026112203.
- Acceptance: preserve node degree sequence, preserve connectedness, preserve node count and edge count; no self-loops or duplicate edges.

## Frozen randomized graph checks
| Seed | Successful swaps | Connected | Degree sequence preserved | Edge-list SHA-256 |
|---:|---:|---|---|---|
| 2026112201 | 19226 | yes | yes | e661ce2b5d6f9b87ccd8f6c4abb5cad4172e4fd0a5e22aa377d80735c0f3d70e |
| 2026112202 | 19215 | yes | yes | 21b51d9a8282eda0f3f36147a169c0648ff8d4456d17f933ec0061859c8807fc |
| 2026112203 | 19186 | yes | yes | 079c4ae6f052f3ac2bcd1512271060aa796add0abdcf02462b8d33dae30f35a6 |

## Explicit deviation
The historical G2-R3 graph edge list and rewiring implementation were not recoverable. The above is a newly constructed and hashed graph family, not a bitwise reproduction of the historical experiment. The k=6 label denotes six outgoing nearest neighbors before union symmetrization; final undirected degrees vary from 6 to 11.

## Still to freeze before training
- Node encoder and pair decoder architecture.
- Message-passing layer and normalization.
- Pair feature construction, invariant to swapping sensor indices.
- Pair sampling and invalid-target handling.
- Train-only normalization and internal validation selection.
- Deterministic training configuration and checkpoint selection.

Do not launch the 225-fit grid until these are implemented, unit-tested and documented.
## Recheck correction
Direct execution with the uploaded archive and NetworkX 3.6.1 corrected the preliminary edge count and rewiring counts above. The correct union-symmetrized spatial graph has 1,966 edges. `connected_double_edge_swap` in NetworkX 3.6.1 does not accept `max_tries`; the implementation was corrected to use the supported API. The three randomized graphs preserve degree sequence and connectedness. Their listed swap counts are the actual return values from the recheck.
