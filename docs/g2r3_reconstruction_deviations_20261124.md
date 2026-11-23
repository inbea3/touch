# G2-R3 Reconstruction Deviations and Graph Freeze

Date: 2026-11-24
Experiment ID: G2-R3-RECONSTRUCTION-20261124
Status: graph topology frozen; model architecture and training pipeline still pending.

## Reconstructed spatial graph
- Coordinates: 548 active cells in row-major (row, column) order, from `active_sensor_mask` in `src/touch_research/decomposition/stag.py`.
- Distance: Euclidean grid distance.
- Directed neighbor selection: six nearest other active sensors per node.
- Undirected conversion: union symmetrization, no self-loops.
- Graph: 548 nodes, 1,925 undirected edges, degree range 6–11, median 7, connected.
- Edge-list SHA-256: ee7fe5fafbed3882d375868df7c3a431218088b56eec661c142fc7e191e022ad.

## Rewiring algorithm
- Library: NetworkX 3.6.1.
- Function: `networkx.connected_double_edge_swap`.
- Swap budget: 10 × original edge count = 19,250 successful swaps requested.
- Seeds: 2026112201, 2026112202, 2026112203.
- Acceptance: preserve node degree sequence, preserve connectedness, preserve node count and edge count; no self-loops or duplicate edges.

## Frozen randomized graph checks
| Seed | Successful swaps | Connected | Degree sequence preserved | Edge-list SHA-256 |
|---:|---:|---|---|---|
| 2026112201 | 18804 | yes | yes | b39d6db79565c515700a0b9b974b9ba179022d9e1a1cdef9d6fa68096a2c7085 |
| 2026112202 | 18824 | yes | yes | 1ce0f101278302aaf0e15e7c41a0d276b5811775c37103ad0df79085f1bf99fd |
| 2026112203 | 18811 | yes | yes | 129097b6c68053da101d7f29fee86bbe8dd9a253b0da64fe286a9bc332702cc4 |

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