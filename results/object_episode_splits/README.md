# Object-level split and Episode manifests

Generated from the pressure-only metadata using seed 20261118 and K=5.

- 26 objects / 78 recordings; 4 empty-hand recordings excluded.
- Meta-train: 18 objects / 54 recordings / 108 Episodes.
- Meta-validation: 4 objects / 12 recordings / 24 Episodes.
- Meta-test: 4 objects / 12 recordings / 24 Episodes.
- 1-shot and 2-shot settings are exhaustively enumerated.
- All object/recording/Support-Query leakage checks passed.

The CSV manifests are regenerated locally by scripts/build_object_episode_splits.py; raw pressure arrays are not committed. See docs/object_episode_protocol.md for the full protocol and limitations.
