# HAS Stage 26 — object-centered residual structure audit

- Workflow run: 45217962230
- URL: https://github.com/inbea3/touch/actions/runs/45217962230
- Artifact: has-stage26-object-centered-audit-45217962230
- Artifact ID: 14839721387
- Artifact SHA-256 (GitHub): 8049d8e08b232f50dd2a3e85d1ea0a4f5a87fcb9d861241f89623eea8f98ebb7
- Workflow: success; all steps succeeded
- Input: Stage 22A run 45193420854; 5-frame 168-D input, 42-D target; 4,192 episodes
- Ridge alpha fixed at 1000 from Stage 24B
- Output manifest: all entries verified independently.

## Validation: equal-object channel summaries and within-object correlations

The six Validation objects are equally weighted after computing per-object metrics and per-object-centered correlation matrices.

| Residual source | Mean absolute off-diagonal correlation | Median absolute correlation | Maximum absolute correlation |
|---|---:|---:|---:|
| Persistence | 0.0595 | 0.0434 | 0.6724 |
| Ridge residual | 0.0790 | 0.0599 | 0.5104 |

The largest Persistence residual pair is source channels 23 and 31 (r=0.6724). The largest Ridge residual pair is source channels 11 and 12 (r=0.5104). These estimates are now within-object correlations, then Fisher-z aggregated with equal object weights; they are not confounded by constant between-object residual offsets in the same way as Stage 25.

### Validation LOOO recurrence

For Persistence, source channels 11/12 recur in the top-10 pair list in all 6 leave-one-object-out runs; channels 6/8 recur in 5/6 runs. For Ridge, pairs 2/11, 6/8, 11/12 and 27/29 recur in 6/6 runs. This indicates some correlation rankings are not driven by a single Validation object. It does not establish physical connectivity or causal interaction.

## Validation object-level performance heterogeneity

| Object | Persistence MAE | Ridge MAE |
|---|---:|---:|
| cubemarsbox | 0.0259 | 0.0267 |
| foam | 0.0128 | 0.0143 |
| maxwell | 0.0152 | 0.0171 |
| papercup | 0.0169 | 0.0190 |
| ricecake | 0.0104 | 0.0111 |
| sugar | 0.0107 | 0.0121 |

Persistence has lower object-level MAE on all six Validation objects. Ridge has lower object-level RMSE on cubemarsbox, foam and ricecake, but higher on maxwell, papercup and sugar. The residual structure therefore does not translate into MAE gains.

## Test: exploratory only

Test within-object correlation summaries:
- Persistence: mean absolute off-diagonal r=0.0639; max=0.6438.
- Ridge: mean absolute off-diagonal r=0.0795; max=0.5359.
- LOOO recurrent Persistence pairs include 6/8 and 6/11 (6/6), and several pairs involving channel 16 (5/6).
- LOOO recurrent Ridge pairs include 6/8, 6/11 and 33/38 (6/6).

The fixed Test objects were inspected in earlier development; these results are descriptive only and do not drive decisions.

## Interpretation and decision

1. Stage 26 confirms non-zero within-object cross-channel residual association, with some pair rankings recurring under Validation LOOO.
2. Persistence and Ridge have different residual correlation structures; Ridge does not reduce overall residual dependence (mean absolute correlation rises from 0.0595 to 0.0790 on Validation).
3. Stable correlation pairs are hypothesis-generating only. They are not a physical sensor graph. The current evidence does not justify claiming a GNN advantage.
4. Persistence remains better on object-level Validation MAE for all six objects; no learned correction has demonstrated an MAE improvement.
5. Next step should be channel provenance/physical-layout recovery: identify the sensor-to-source-channel mapping from dataset documentation or raw acquisition metadata. If no independently supported geometry exists, test a graph only as a carefully labeled exploratory correlation-graph ablation, not as a physically grounded model. Any graph must be built using Train objects only and evaluated with the same object-disjoint split.

## Provenance
Artifact includes object-macro channel metrics, per-object per-channel metrics, object-level metrics, four within-object Fisher-z correlation matrices, LOOO stability summaries, config, environment, run.log and SHA-256 manifest. Prior artifacts were not overwritten.
