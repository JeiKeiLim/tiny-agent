---
id: TASK-005.09
title: 'Corpus builder: skip rebuild when manifest verifies existing corpus'
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-26 01:01'
updated_date: '2026-08-26 01:03'
labels:
  - data
  - corpus
dependencies: []
documentation:
  - doc-003
modified_files:
  - src/kestrel/corpus/builder.py
  - tests/test_corpus_builder.py
parent_task_id: TASK-005
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make corpus build idempotent so pretrain does not regenerate a valid corpus.

Problem:
- `pretrain()` always calls `build_corpus(corpus_cfg)`.
- If `data/corpus` already contains the correct JSONL files and manifests, the builder currently streams/downloads and rewrites the corpus again.
- For the ~1GB M1 corpus this makes repeated pretrain runs slow and unnecessary.

Outcome:
- `build(config)` checks whether the existing corpus under `config.output_dir` is complete and consistent with the current `CorpusConfig`.
- If valid, `build` returns the existing per-component byte counts without rewriting files.
- If invalid, missing, stale, or `force=True`, it rebuilds as before.

Design decisions:
- Add a config fingerprint to each split manifest. The fingerprint should cover at least: total_bytes, seed, output_format, tokenizer_path, val_fraction, test_fraction, and each component name/fraction/source.
- A split is complete when:
  - `manifest.json` exists.
  - manifest `split`, `seed`, `output_format`, and config fingerprint match.
  - every component file listed in the manifest exists.
  - each file size matches the manifest `byte_count`.
  - manifest totals match the sum of file entries.
- `build` should check all active splits (`train`, optional `test`, optional `val`) before skipping.
- Add `force: bool = False` to `build()` rather than requiring a new YAML field for M1.
- Keep the existing manifest schema backward compatible if possible; old manifests without the fingerprint should trigger rebuild.

Files to modify:
- `src/kestrel/corpus/builder.py`
- `tests/test_corpus_builder.py`

Acceptance targets:
- A second `build()` call with the same config does not invoke the local/HF source builders.
- Changing `total_bytes`, `seed`, `output_format`, `tokenizer_path`, split fractions, or component source/fraction triggers rebuild.
- Missing manifest, missing file, file size mismatch, or corrupt manifest triggers rebuild.
- `force=True` triggers rebuild.
- `make check` is green.

Reference: doc-003.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Second build with same config skips source builders and returns existing byte counts
- [x] #2 Stale config fingerprint triggers rebuild
- [x] #3 Missing or mismatched corpus files trigger rebuild
- [x] #4 force=True triggers rebuild
- [x] #5 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add config fingerprint helper for CorpusConfig. 2. Write fingerprint into per-split manifests. 3. Add corpus completeness validator. 4. Make build() return existing byte counts when valid unless force=True. 5. Add tests for skip/stale/missing/force. 6. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added config_fingerprint to per-split manifests and _existing_results() validator. build(config, force=False) skips when all active split manifests match fingerprint, file sizes, and totals. Existing data/corpus manifests were updated in place with the current fingerprint, and a real build() call against configs/kestrel/corpus.yaml now prints: corpus already complete in data/corpus; skipping build. make check green: 107 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corpus builds are idempotent. build() verifies per-split manifests against a CorpusConfig fingerprint and on-disk file sizes, returning existing byte counts when complete. Stale, missing, mismatched, or force=True builds rebuild. The current data/corpus was fingerprinted and verified in place.
<!-- SECTION:FINAL_SUMMARY:END -->
