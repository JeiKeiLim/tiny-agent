---
id: TASK-005.10.03
title: Create and build data/corpus-12g expanded corpus
status: To Do
assignee: []
created_date: '2026-08-26 01:36'
updated_date: '2026-08-26 02:16'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies:
  - TASK-005.10.01
modified_files:
  - configs/kestrel/corpus-12g.yaml
  - configs/kestrel/corpus-12g-smoke.yaml
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
- [ ] #1 configs/kestrel/corpus-12g.yaml and configs/kestrel/corpus-12g-smoke.yaml load into CorpusConfig
- [ ] #2 Smoke build writes data/corpus-12g-smoke with train and val manifests
- [ ] #3 Full build writes data/corpus-12g with train and val manifests
- [ ] #4 Second full build run skips because the existing corpus is complete
- [ ] #5 Train manifest total_estimated_token_count is at least 2900000000
- [ ] #6 Component byte counts are within 10% of the 85/10/5 targets
- [ ] #7 Existing data/corpus is unchanged
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 02:16
---
2026-08-26: Changed val_fraction from 0.1 to 0.01 to preserve training data; validation only uses eval_iters batches.
---
<!-- COMMENTS:END -->
