---
id: TASK-007.03.12
title: Make SFT source prep target written rows after filtering
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 23:31'
updated_date: '2026-08-31 23:35'
labels:
  - sft
  - data
  - bug
  - implementation
milestone: m-2
dependencies: []
modified_files:
  - src/kestrel/data/sft_prepare.py
  - src/kestrel/data/sft_prepare_public.py
  - src/kestrel/data/sft_prepare_gsm8k.py
  - configs/kestrel/sft_data.yaml
  - tests/data/test_sft_prepare_public.py
  - README.md
parent_task_id: TASK-007.03
priority: high
ordinal: 52000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix SFT raw-source prep so target_rows means rows written after conversion/context filtering, not raw candidates inspected.

Context: assistant_public requested 22500 rows but only 12104 were written because prepare_rows() reservoir-samples target_rows raw Smol-SmolTalk rows first and filters afterward. About 10396 candidates were filtered by context_length=1024, causing a large assistant deficit in the SFT mixture.

Files:
- src/kestrel/data/sft_prepare.py
- src/kestrel/data/sft_prepare_public.py
- src/kestrel/data/sft_prepare_gsm8k.py
- configs/kestrel/sft_data.yaml
- tests/data/test_sft_prepare_public.py
- tests/data/test_sft_prepare_gsm8k.py

Scope:
- prepare_rows() should stream raw rows, convert/filter each row, and continue until target_rows valid rows are collected or the source/max_candidate_rows cap exhausts.
- target_rows should remain manifest.requested_rows and should represent desired written_rows.
- candidate_rows should count raw rows inspected.
- filtered_rows should count inspected rows that fail conversion or context filtering.
- Add optional assistant.max_candidate_rows safety cap and set it in configs/kestrel/sft_data.yaml.
- Use seeded streaming shuffle in assistant/GSM8K loaders where practical so early-stop sampling remains representative.
- GSM8K may remain short by 27 rows; do not change the GSM8K target just to force 7500.

Verification:
- make check passes.
- Re-running assistant prep can collect 22500 written assistant_public rows when the source has enough valid rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 prepare_rows collects target_rows valid rows when enough valid rows exist after filtering
- [x] #2 prepare_rows stops after target_rows valid rows and records inspected candidate/filtered counts
- [x] #3 max_candidate_rows caps inspected raw rows and records deficit if the cap is reached before target
- [x] #4 assistant config and committed YAML support max_candidate_rows with a safe cap
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update prepare_rows() in src/kestrel/data/sft_prepare.py to stream raw rows, convert/filter each row, collect the first target_rows valid rows, count inspected candidate_rows and filtered_rows, and support max_candidate_rows.
2. Add max_candidate_rows to AssistantSourceConfig and pass it through prepare_assistant().
3. Add seeded streaming shuffle to load_smol_rows() and load_gsm8k_rows() so early-stop sampling is representative for a fixed seed.
4. Update configs/kestrel/sft_data.yaml with assistant.max_candidate_rows: 100000.
5. Add/adjust tests for continue-past-filtered-rows, early stop after target, max_candidate_rows cap, source exhaustion deficit, and committed config value.
6. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed prepare_rows() to stream/convert/filter until target_rows valid rows are collected, added assistant.max_candidate_rows, seeded streaming shuffle for assistant/GSM8K loaders, committed assistant.max_candidate_rows=100000, updated README, and added tests. make check passes with 306 tests. Re-ran assistant prep: candidate_rows=41702, filtered_rows=19202, written_rows=22500. Rebuilt mixture: total_rows=49973/50000 with only gsm8k_math deficit_rows=27.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed SFT source prep so target_rows means valid written rows after conversion/context filtering. prepare_rows() now streams until target valid rows are collected, counts inspected candidates/filtered rows, and supports assistant.max_candidate_rows. Added seeded streaming shuffle to assistant/GSM8K loaders, committed assistant.max_candidate_rows=100000, updated README, and added tests. Verified with make check: 306 tests passed. Re-ran assistant prep: 22500/22500 written from 41702 candidates; rebuilt mixture: 49973/50000 rows with only the expected 27-row GSM8K deficit.
<!-- SECTION:FINAL_SUMMARY:END -->
