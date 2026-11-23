# G2-R3 Execution Readiness Audit — 2026-11-24

## Current verified state
- Uploaded pressure-only archive was parsed locally: 78 recordings, 26 non-empty objects, 548 active sensors.
- End-to-end one-epoch smoke harness completed for Node-MLP and Spatial-GNN on one object-grouped split.
- Three degree-preserving connected rewired graphs were generated and hashed.
- All five model conditions passed synthetic forward/backward and pair-swap invariance checks.
- Smoke archive contains checkpoints, epoch traces, recording/object metrics, run logs and SHA-256 manifest.

## Important limitation
The local smoke harness used a separately executed one-epoch harness; it did not invoke the committed scripts/run_g2r3_reconstruction.py --smoke-only command. Therefore the committed CLI path remains to be validated with the raw metadata file in the same runtime.

## Code audit changes
- Added deterministic per-fit directory names.
- Completed fits are skipped on rerun using run_status.json.
- Interrupted partial fits are preserved in an _attemptN directory; a fresh attempt is started rather than overwriting prior traces.
- Environment metadata is written before the per-fit SHA-256 manifest.
- Top-level output directory can be reused, allowing completed runs to be resumed.
- CI workflow now triggers on relevant code changes and supports manual dispatch.
- Removed pip cache configuration because the repository has no requirements.txt or pyproject.toml.

## CI status
The latest G2-R3 CI run is being checked separately. Do not launch the 225-fit experiment until script compilation and all module tests pass, and the committed CLI smoke path has been validated.

## Execution gate
Next: inspect latest CI result; repair any failures; then run the committed CLI smoke command against the uploaded metadata. The 225-fit grid remains NOT STARTED.
