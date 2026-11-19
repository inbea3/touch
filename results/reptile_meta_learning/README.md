# Stage 9 Reptile implementation status

This directory currently stores the proposed pilot configuration and data provenance checks only. **No Reptile training run has been executed and no performance result is available.**

- `pilot_config.json`: planned smoke/pilot settings; not an executed run record.
- `pilot_provenance.json`: archive and extracted MAT-file integrity hashes verified locally; it explicitly records that model training and tests have not been run.

Earlier unverified pilot summary and object-metric files were removed. Do not cite or use the previously circulated MAE values.

Before a valid run can be recorded:
1. Obtain a local checkout of `integration/old-research-assets`.
2. Install and record dependencies.
3. Run syntax checks and all 11 unit tests.
4. Run a real-data smoke test against the supplied pressure-only `metadata.mat`.
5. Audit the matched MLP/Reptile initialization, data exposure and gradient/compute budgets.
6. Only then run the multi-seed Meta-validation experiment and save actual logs, checkpoints, task samples, episode/object metrics, scaler and hashes.

The fixed 18/4/4 object split has been used in prior stages. Any future Meta-validation/Meta-test result on it remains exploratory, not an untouched confirmatory evaluation.
