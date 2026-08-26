---
id: TASK-005.11
title: Fix repeated PretrainDataset validation iteration returning no batches
status: Done
assignee: []
created_date: '2026-08-26 02:14'
updated_date: '2026-08-26 02:22'
labels:
  - bug
  - pretraining
  - data
dependencies: []
modified_files:
  - src/kestrel/data/pretrain_dataset.py
  - tests/test_pretrain_dataset.py
parent_task_id: TASK-005
priority: high
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pretrain validation prints 'val inf' after the first validation pass. Root cause introduced in 439195d: PretrainDataset stores per-iteration file iterators on shared _Source objects. When the trainer stops validation early after eval_iters batches, the abandoned __iter__ generator closes those source iterators. The next validation pass reuses the closed iterators, treats every source as exhausted, yields zero batches, and estimate_val_loss() silently returns inf.

Root-cause resolution: make PretrainDataset a proper reusable iterable. Each __iter__ call must own its temporary file iterators locally instead of mutating shared _Source state. Close only the local iterators when that iteration ends. Do not revert the document-aware pipeline and do not re-instantiate the whole dataset for each validation pass.

Scope decision: keep the fix minimal. Do not change trainer.py or add empty-validation guard behavior in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PretrainDataset can be iterated, stopped early, and iterated again without losing the remaining input
- [x] #2 Regression test covers repeated iteration for both a single JSONL file and a multi-file corpus directory
- [x] #3 No trainer.py behavior changes are included
- [x] #4 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Revert trainer.py to the committed behavior. 2. Keep the local per-iteration iterator fix in PretrainDataset.__iter__ and remove shared iterator state from _Source. 3. Add regression tests that iterate a single-file PretrainDataset and a multi-file directory PretrainDataset, stop after one batch, then iterate again and assert another batch is produced. 4. Run make check and fix all format/lint/type/test failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reverted src/kestrel/train/trainer.py to committed behavior. Kept root-cause fix in src/kestrel/data/pretrain_dataset.py: _Source no longer stores iterator; PretrainDataset.__iter__ uses local iterators and closes only those local iterators. Added repeated-iteration regression tests for single-file and multi-file inputs. make check passed: ruff, format, mypy, 109 pytest tests.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 02:22
---
2026-08-26: User confirmed root-cause fix should be local iterator ownership, not a shared-state reset or dataset re-instantiation. Trainer guard removed from scope to keep the change minimal.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed repeated validation iteration by making PretrainDataset iteration state local to each __iter__ call. No trainer behavior changes were made. Verified with repeated-iteration regression tests and make check (109 tests passed).
<!-- SECTION:FINAL_SUMMARY:END -->
