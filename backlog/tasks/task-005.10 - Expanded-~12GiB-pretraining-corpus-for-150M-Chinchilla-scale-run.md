---
id: TASK-005.10
title: Expanded ~12GiB pretraining corpus for 150M Chinchilla-scale run
status: To Do
assignee: []
created_date: '2026-08-26 01:35'
updated_date: '2026-08-26 02:15'
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
- [ ] #1 Child tasks for builder safety, shuffle, corpus build, and pretrain configs are all Done
- [ ] #2 data/corpus-12g/train/manifest.json exists with total_estimated_token_count at least 2900000000
- [ ] #3 Total raw bytes across train and val are within 5% of 13212000000
- [ ] #4 web, code, and synthetic component byte counts are each within 10% of their 85/10/5 targets
- [ ] #5 Existing data/corpus smoke corpus is unchanged
- [ ] #6 make check is green
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 02:15
---
2026-08-26: Changed planned val_fraction from 0.1 to 0.01 after reviewing eval_iters usage; 10% validation is unnecessarily expensive in training data.
---
<!-- COMMENTS:END -->
