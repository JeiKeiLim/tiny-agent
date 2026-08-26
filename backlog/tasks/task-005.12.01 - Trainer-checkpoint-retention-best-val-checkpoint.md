---
id: TASK-005.12.01
title: Trainer checkpoint retention + best-val checkpoint
status: To Do
assignee: []
created_date: '2026-08-26 04:14'
labels:
  - training
  - checkpoint
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/train/trainer.py
  - tests/test_trainer.py
parent_task_id: TASK-005.12
priority: high
ordinal: 30000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add bounded checkpoint retention to the shared trainer and write a best-validation checkpoint.

Files:
- modify src/kestrel/train/trainer.py
- modify tests/test_trainer.py
- update pretrain YAML only if the new fields should be made explicit (optional)

Current behavior:
- trainer.train() writes output_dir/step_NNNNNN every save_every steps and output_dir/final at the end.
- best_val is only returned in TrainResult; no output_dir/best checkpoint exists.
- old step directories are never deleted.

Config changes on TrainerConfig:
- keep_latest_checkpoints: int | None = 3
  - None disables pruning.
  - int must be >= 1.
  - Pruning keeps only the N newest step_NNNNNN directories.
- keep_best_checkpoint: bool = True
  - When true, write output_dir/best whenever validation loss improves.

Behavior:
- best checkpoint is written only on a strict improvement in val_loss.
- final is always written at the end and never pruned.
- Pruning may only delete directories matching step_[0-9]+ directly under output_dir.
- best, final, and unrecognized directories must never be deleted.
- If keep_latest_checkpoints is None, existing retain-all behavior remains.
- This subtask may write best as a weights-only checkpoint; TASK-005.12.02 will upgrade all checkpoints to full resumable state.

Tests:
- After 5 periodic checkpoints with keep_latest_checkpoints=2, only the two newest step directories plus final remain.
- keep_latest_checkpoints=None retains all step directories.
- A best checkpoint is written when val loss improves and survives pruning.
- Pruning never deletes best or final.
- Strict config validation rejects keep_latest_checkpoints=0 and non-integer values.

Gate: make check green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TrainerConfig supports keep_latest_checkpoints and keep_best_checkpoint under strict validation
- [ ] #2 Only N latest step_NNNNNN checkpoints are retained when keep_latest_checkpoints is set
- [ ] #3 best and final checkpoints are never pruned
- [ ] #4 A best checkpoint is written when validation loss improves
- [ ] #5 make check is green
<!-- AC:END -->
