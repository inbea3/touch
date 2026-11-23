# G2-R3 recovery audit — 2026-11-24

## Result: NOT RECOVERABLE from the currently inspected repository state

This audit was performed before launching or rerunning any G2 job, to avoid overwriting or duplicating an allegedly existing 225-run experiment.

### Checks performed
1. Queried GitHub Actions workflow runs for `inbea3/touch` across five pages (up to 500 recent run records returned by the endpoint). No workflow run name or workflow path matched G2, graph, 548-node, or spatial-graph experiments.
2. Enumerated the current `integration/old-research-assets` branch root, `.github/workflows`, `scripts`, `docs`, and `results` directory listings.
3. The current branch contains the pressure-only baseline workflow and STAG thesis documents, but the inspected workflow/script/docs/results listings do not expose the G2-R2/R3 scripts, workflow, report, fold manifests, progress checkpoint, model checkpoints, or artifacts described in the Stage A blueprint.
4. The previous Stage A blueprint records a historical statement that G2-R3 was started on 2026-11-22 and that a stale snapshot showed 8/225 runs complete. That statement is not independently verifiable from the currently accessible Actions run history and repository files inspected in this audit.

### Interpretation
- This is an evidence-recovery failure, not evidence that the 225 runs failed or never existed.
- Do not claim G2-R3 completed, failed, or is still running.
- Do not launch a replacement 225-run experiment until the original workflow URL/run ID, branch/commit, artifact name, or local archive is identified.
- Do not cite G2-R2/R3 metrics as verified primary thesis results until their underlying run artifacts and per-object outputs are recovered.
- The historical blueprint's G2-R2/R3 section must be treated as an unverified research-log entry pending source recovery.

### Required recovery key
At least one of:
1. Exact GitHub Actions run URL or numeric run ID for G2-R2 and G2-R3.
2. The branch/repository where the G2 scripts and workflow were executed.
3. Artifact names or downloadable ZIP files.
4. A local archive containing config, logs, checkpoints, fold/object metrics and SHA-256 manifest.

### Next safe action
Continue with STAG E01 input-frame continuity/time-span sensitivity analysis only after confirming its script, input data source and split manifests. Keep G2-R3 blocked pending recovery. No previous artifacts were modified and no new training was launched.

## Provenance
Repository inspected: `inbea3/touch`
Branch inspected: `integration/old-research-assets`
Audit date: 2026-11-24
