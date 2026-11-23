# HAS Stage 20b — baseline refinement protocol

Purpose: improve model comparison after Stage 20 MLP failed to beat simple reference predictors.

Frozen input: Stage 19 artifact run 45115559336, dataset revision 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf, object split seed 20261123. Test objects remain can, muesli, softlego, spam, toiletroll, tumbox. Do not change split or inspect Test for model selection.

## Experiments

1. Future mean residual Ridge: predict future mean minus last observed value. Ridge alpha grid {0.01, 0.1, 1, 10, 100, 1000}; select alpha by validation standardized MSE.
2. Future mean residual MLP: same residual target, 168 -> 256 -> 128 -> 42; dropout 0.15; AdamW lr 0.0005, weight decay 0.001; three seeds; early stopping on validation standardized MSE. Reconstruct future mean by adding the last observed input.
3. Future correlation Ridge: 861 upper-triangle targets; same alpha grid, validation-only selection.
4. Correlation template shrinkage: predict lambda times the train-mean correlation vector, lambda grid {0, 0.1, 0.25, 0.5, 0.75, 1}; choose lambda on Validation only.

All scalers and regression estimators fit on Train only. The locked 42-channel primary policy excludes channels 18 and 25. No all-44 tuning in this refinement run.

## Metrics

Report MAE, RMSE and per-episode Spearman; report six test-object metrics and object-macro MAE/RMSE. Compare directly with Stage 20 persistence and zero-correlation references. All seeds are reported. Test is used once after validation selection.

## Provenance

Save fitted Ridge estimators, residual-MLP checkpoints, traces, predictions, config, metrics, environment, run.log and SHA-256 manifest. Preserve prior Stage 20 artifacts. This is still an exploratory benchmark over six held-out object categories; no confirmatory generalization claim.
