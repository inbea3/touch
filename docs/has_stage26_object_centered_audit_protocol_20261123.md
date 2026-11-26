# HAS Stage 26 — object-centered residual structure audit

## Motivation
Stage 25 reported episode-pooled channel errors and correlations across all objects. This may confound within-object channel dependence with between-object offsets and unequal episode counts. Stage 26 corrects the aggregation and correlation estimand.

## Frozen inputs
- Stage 22A matched-window artifact run 45193420854.
- Five-frame input, 168 features, 42 future-mean targets.
- Fixed 21/6/6 object-disjoint split.
- Persistence is last observed pressure vector X[:,126:168].
- Ridge residual model uses alpha=1000 fixed from Stage 24B Validation selection; fit on Train only.
- Original channel mapping: retained channel indices map to source indices 0..43 excluding 18 and 25.

## Analyses
1. Per-object, per-channel bias, MAE, RMSE and residual SD for Persistence and Ridge.
2. Object-macro per-channel metrics: compute metrics within each object, then average objects equally.
3. Within-object residual correlation: center each object's residual matrix by that object's channel-wise residual mean, calculate 42x42 correlation matrix per object, then Fisher-z average matrices with equal object weights.
4. Leave-one-object-out (LOOO) correlation stability on Validation: recompute equal-object Fisher-z aggregate after omitting each of six objects; report recurring top-10 channel pairs.
5. Test is reported as exploratory only.

## Interpretation constraints
- LOOO frequency is descriptive stability analysis, not inferential confirmation.
- Correlation is error covariance, not physical sensor adjacency.
- No GNN may be justified without independently sourced sensor geometry or a defensible physical adjacency.
- Six Validation objects limit precision; report per-object heterogeneity.

## Provenance
Save config, per-object channel metrics, object-macro channel metrics, object-macro metrics, within-object correlation matrices, LOOO pair stability, summary, environment, run.log and SHA-256 manifest. No raw pressure sequences are saved; previous artifacts remain unchanged.
