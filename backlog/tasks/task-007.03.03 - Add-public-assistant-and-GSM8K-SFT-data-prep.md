---
id: TASK-007.03.03
title: Add public assistant and GSM8K SFT data prep
status: To Do
assignee: []
created_date: '2026-08-31 01:20'
labels:
  - sft
  - data
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.01
parent_task_id: TASK-007.03
priority: high
ordinal: 43000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement data prep for the public assistant and GSM8K math slices.

Depends on:
- TASK-007.03.01

Files:
- src/kestrel/data/sft_prepare.py or src/kestrel/data/sft_prepare_public.py
- src/kestrel/data/sft_prepare_gsm8k.py
- scripts/run_prepare_sft_public.py if a CLI entry point is needed
- tests/data/test_sft_prepare_public.py
- tests/data/test_sft_prepare_gsm8k.py
- configs/kestrel/50m/sft_data.yaml if config-driven data prep is introduced

Scope:
- Public assistant source: Smol-SmolTalk train.
- Sample 22,500 rows deterministically by seed.
- Convert conversations to the Kestrel logical SFT schema.
- Filter rows by max rendered token length.
- Tag rows with source=assistant_public.
- GSM8K source: GSM8K train split, about 7,473 rows.
- Convert each problem/solution into user problem + assistant CoT/final answer.
- Tag rows with source=gsm8k_math.
- Write per-source JSONL files under data/sft/raw/.
- Write a small manifest with source name, row count, seed, dataset ID, and hash.

Constraints:
- Unit tests must not require network access; use tiny local fixtures.
- Network download paths may be tested separately or manually.
- Keep the trainer independent of Hugging Face dataset internals.

Acceptance:
- assistant_public.jsonl contains the requested count or available fixture count in tests.
- gsm8k_math.jsonl contains chat/CoT rows with source tag.
- Manifest records counts/seeds/hashes.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Smol-SmolTalk rows are converted to the Kestrel logical schema
- [ ] #2 GSM8K rows are converted to user problem + assistant CoT/final answer rows
- [ ] #3 Output JSONL files include source tags and manifest metadata
- [ ] #4 Unit tests use local fixtures and do not require network access
- [ ] #5 make check passes
<!-- AC:END -->
