# HAS object-disjoint split construction report

- Workflow run: 45112395632
- Run URL: https://github.com/inbea3/touch/actions/runs/45112395632
- Artifact: has-object-grouped-split-45112395632
- Artifact ID: 14790716045
- Artifact SHA-256: 4d3153afbffc330e57b47ab25deb6fc213d8876d51abf1f05222ddc22a177a73
- Workflow dispatch commit: 11e7370ec5b122f60ef8478ee9ddec0f46ad61dd
- Split builder revision: 34c63a4bcd7b01e8d4bb4e4ec54720a6cf9a7180 (verified from successful artifact output/config and corrected run logic)
- Dataset: TUM-ICS/Hide-and-Seek
- Seed: 20261123
- Hugging Face revision: not pinned; current default revision at execution time
- Python: 3.11.18
- Status: successful

## Results

- 33/33 object categories assigned exactly once.
- Object split counts: train 21, validation 6, test 6.
- Object-disjoint invariant: PASS; no object category appears in more than one split.
- Object-conditioned episodes: 4,192.
- Excluded no-object/negative episodes: 122.
- Total episodes accounted for: 4,314.
- Episode counts: train 2,452; validation 880; test 860.
- All weight variants of each base object category were grouped together.
- The original official split is retained as metadata only; an episode may be assigned to a new grouped split irrespective of its original split.

## Exact object allocation

Train (21): bag, bluecube, bowl, chip, cleanser, cube, measuringcup, mustard, noodle, pitcher, ppbox, pringles, redcup, tape, teabox, toast, tomatosoup, udon, weinglass, wood, youcook.

Validation (6): cubemarsbox, foam, maxwell, papercup, ricecake, sugar.

Test (6): can, muesli, softlego, spam, toiletroll, tumbox.

## Interpretation and limitations

This is a valid object-disjoint exploratory split, not a multi-seed robustness result. Only 6 object categories are held out for final test, so uncertainty across object identities may be substantial. The 122 negative episodes were excluded from this object-conditioned task and must be handled separately if a future presence/absence or open-set task is defined.

The dataset revision was not pinned. For strict long-term reproducibility, pin a Hugging Face dataset commit SHA in the next data-processing run and retain the manifests. Do not change this locked split after inspecting model test performance.

The split builder used only episode_index and classification_label metadata; it did not read or save raw force vectors.

## Artifact contents

config.json; environment.txt; episode_split_manifest.csv; excluded_negative_episodes.csv; manifest.sha256; object_split_manifest.csv; run.log; summary.json.

Artifact internal SHA-256 manifest was present and verified structurally. Artifact retention: 90 days.
