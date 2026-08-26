---
id: TASK-005.10.03
title: Create and build data/corpus-12g expanded corpus
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-26 01:36'
updated_date: '2026-08-26 04:28'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies:
  - TASK-005.10.01
modified_files:
  - configs/kestrel/corpus.yaml
parent_task_id: TASK-005.10
priority: high
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the expanded 12GiB corpus config and build data/corpus-12g. This corpus is the first serious pretraining data set for Kestrel-150M and an overtrained or capped data set for Kestrel-50M.

Files:
- create configs/kestrel/corpus-12g.yaml
- create configs/kestrel/corpus-12g-smoke.yaml
- build data/corpus-12g-smoke
- build data/corpus-12g
- do not modify data/corpus

Required configs/kestrel/corpus-12g.yaml fields:
- total_bytes: 13212000000
- seed: 0
- output_dir: data/corpus-12g
- output_format: jsonl
- tokenizer_path: null
- val_fraction: 0.01
- test_fraction: 0.0
- min_component_fill: 0.9
- components:
  - name: web, fraction 0.85, source type hf, dataset HuggingFaceTB/smollm-corpus, config fineweb-edu-dedup, text_field text
  - name: code, fraction 0.10, source type hf, dataset codeparrot/codeparrot-train-more-filtering, text_field content
  - name: synthetic, fraction 0.05, source type hf, dataset HuggingFaceTB/smollm-corpus, config cosmopedia-v2, text_field text

Smoke config:
- Same sources and fractions.
- total_bytes: 268435456
- output_dir: data/corpus-12g-smoke
- val_fraction: 0.01
- min_component_fill: 0.9

Steps:
1. Add both YAML configs.
2. Verify both load into CorpusConfig under strict Pydantic rules.
3. Run the smoke build: uv run python scripts/build_corpus.py --config configs/kestrel/corpus-12g-smoke.yaml
4. Inspect smoke manifests and sample documents from web, code, and synthetic.
5. Run the full build: uv run python scripts/build_corpus.py --config configs/kestrel/corpus-12g.yaml
6. Run the full build again and verify it prints the idempotent skip message.
7. Record final manifest totals in task notes.

Quantitative targets:
- data/corpus-12g/train/manifest.json total_estimated_token_count >= 2900000000
- total raw bytes across train and val within 5% of 13212000000
- web bytes within 10% of 11230200000
- code bytes within 10% of 1321200000
- synthetic bytes within 10% of 660600000
- val raw bytes within 2% of 1% of total raw bytes

Gotchas:
- Use text_field content for codeparrot/codeparrot-train-more-filtering, not code.
- Do not use HuggingFaceTB/smollm-corpus python-edu as a direct text source; it requires S3 blob downloads.
- tokenizer_path stays null so the build does not spend time tokenizing 12GiB of text.
- val_fraction is 0.01, not 0.1, because validation uses only a small fixed number of batches.
- If min_component_fill triggers, stop and inspect source exhaustion instead of silently lowering the target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 configs/kestrel/corpus-12g.yaml and configs/kestrel/corpus-12g-smoke.yaml load into CorpusConfig
- [x] #2 Smoke build writes data/corpus-12g-smoke with train and val manifests
- [x] #3 Full build writes data/corpus-12g with train and val manifests
- [x] #4 Second full build run skips because the existing corpus is complete
- [x] #5 Train manifest total_estimated_token_count is at least 2900000000
- [x] #6 Component byte counts are within 10% of the 85/10/5 targets
- [x] #7 Existing data/corpus is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create configs/kestrel/corpus-12g.yaml and configs/kestrel/corpus-12g-smoke.yaml. 2. Verify both configs load strictly. 3. Build the 256MiB smoke corpus. 4. Inspect smoke manifests and sample documents. 5. Build the full 12GiB corpus. 6. Verify idempotent skip, manifest totals, component byte targets, and data/corpus unchanged. 7. Record manifest totals in task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created corpus-12g.yaml and corpus-12g-smoke.yaml; both load under strict CorpusConfig. Smoke build completed: train 265,860,139 bytes / 66,465,034 estimated tokens; val 2,617,194 bytes / 654,298 estimated tokens. Sample documents verified for web, code, and synthetic. Full 12GiB build started in background as PID 50714 with log data/corpus-12g-build.log. Existing data/corpus file hashes recorded before the full build.

Full data/corpus-12g build completed. Final totals: train 13,077,577,494 bytes / 3,269,394,373 estimated tokens; val 134,429,714 bytes / 33,607,428 estimated tokens; total 13,212,007,208 bytes. Component totals: web 11,230,201,013; code 1,321,204,982; synthetic 660,601,213. Val fraction 1.017%. Second build printed 'corpus already complete' and skipped. Exception for AC #7: the builder did not modify data/corpus, but the user manually removed data/corpus after the build to free disk space.

Consolidated corpus config in place: configs/kestrel/corpus.yaml now contains the 12GiB corpus settings with output_dir data/corpus-12g. Removed corpus-12g.yaml and corpus-12g-smoke.yaml. Renaming data/corpus-12g to data/corpus is deferred until the active pretrain process is killed.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 02:16
---
2026-08-26: Changed val_fraction from 0.1 to 0.01 to preserve training data; validation only uses eval_iters batches.
---

created: 2026-08-26 04:23
---
Consolidating corpus configs in place: configs/kestrel/corpus.yaml becomes the 12GiB corpus config; corpus-12g.yaml and corpus-12g-smoke.yaml are removed. data/corpus-12g rename deferred until active run is killed.
---

created: 2026-08-26 04:28
---
Future-note: configs/kestrel/corpus.yaml intentionally points to data/corpus-12g for now. CorpusConfig.default output_dir is data/corpus, which is the canonical target. After the active pretrain run is killed, run: mv data/corpus-12g data/corpus; update configs/kestrel/corpus.yaml output_dir to data/corpus; update data/corpus-12g references in tests/test_pretrain.py; make check.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built and verified data/corpus-12g and consolidated the corpus config into configs/kestrel/corpus.yaml. Final corpus: 13.212GiB raw, 3.269B estimated train tokens, 1.017% val, idempotent rebuild skip confirmed.
<!-- SECTION:FINAL_SUMMARY:END -->
