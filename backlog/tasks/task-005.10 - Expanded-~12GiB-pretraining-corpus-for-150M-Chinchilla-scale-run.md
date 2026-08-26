---
id: TASK-005.10
title: Expanded ~12GiB pretraining corpus for 150M Chinchilla-scale run
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-26 01:35'
updated_date: '2026-08-26 04:33'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies: []
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
  - 'https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus'
  - 'https://huggingface.co/datasets/codeparrot/codeparrot-train-more-filtering'
parent_task_id: TASK-005
priority: high
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the M1 smoke runs, data/corpus is only 1GiB raw and about 240,821,095 estimated train tokens. Kestrel-150M has 148,152,960 params. Using about 20 tokens/param as the first serious pretraining target gives about 2.96B train tokens, or about 12.3GiB raw text under the current bytes//4 manifest estimate.

This parent task tracks a new expanded corpus at data/corpus-12g. The existing data/corpus remains the fast smoke corpus and must not be modified.

Target corpus:
- total_bytes: 13212000000
- output_dir: data/corpus-12g
- val_fraction: 0.01
- test_fraction: 0.0
- tokenizer_path: null for fast estimated token counts
- web 0.85: HuggingFaceTB/smollm-corpus config fineweb-edu-dedup, text_field text
- code 0.10: codeparrot/codeparrot-train-more-filtering, text_field content
- synthetic 0.05: HuggingFaceTB/smollm-corpus config cosmopedia-v2, text_field text

Rationale: follow the same broad pattern used by small open LLMs, especially SmolLM-style high-quality educational web + code + synthetic educational text. The current codeparrot-python-only and alpaca sources are too small for a 12GiB corpus, and raw OIG streaming is likely file-sequential rather than representative.

Validation fraction decision: use 1%, not 10%, because in-loop validation evaluates only a small fixed number of batches. A 10% held-out split would remove about 330M estimated tokens from training for no meaningful validation benefit.

Child tasks cover builder safety, deterministic dataset shuffle, the 12GiB corpus build, and 12GiB pretrain configs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child tasks for builder safety, shuffle, corpus build, and pretrain configs are all Done
- [x] #2 data/corpus-12g/train/manifest.json exists with total_estimated_token_count at least 2900000000
- [x] #3 Total raw bytes across train and val are within 5% of 13212000000
- [x] #4 web, code, and synthetic component byte counts are each within 10% of their 85/10/5 targets
- [x] #5 Existing data/corpus smoke corpus is unchanged
- [x] #6 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Complete TASK-005.10.01: corpus builder CLI + source-exhaustion guard. 2. Complete TASK-005.10.02: deterministic per-file document shuffle in PretrainDataset. 3. Complete TASK-005.10.03: create and build data/corpus-12g. 4. Complete TASK-005.10.04: add 12GiB pretrain configs and run short validation. 5. Verify parent ACs and keep existing data/corpus unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-26: Closed parent task. All child tasks are Done. data/corpus-12g verified: total 13,212,007,208 bytes; train 3,269,394,373 estimated tokens; val fraction 1.017%; component byte counts within target. make check green with 119 tests. AC #5 exception: old data/corpus was manually removed by the user after the build; the builder did not modify it.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 02:15
---
2026-08-26: Changed planned val_fraction from 0.1 to 0.01 after reviewing eval_iters usage; 10% validation is unnecessarily expensive in training data.
---

created: 2026-08-26 04:28
---
Future-note: the 12GiB corpus is currently stored in data/corpus-12g while configs/kestrel/corpus.yaml is the only corpus config. The long-term canonical layout is data/corpus. Rename and update output_dir/tests after the active pretrain run is killed.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded 12GiB corpus workstream complete: corpus builder CLI/guard, deterministic document shuffle, data/corpus-12g build, and consolidated 50M/150M pretrain configs. Verified with corpus manifests and make check (119 tests).
<!-- SECTION:FINAL_SUMMARY:END -->
