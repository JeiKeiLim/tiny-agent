---
id: TASK-005.12
title: Trainer checkpoint resume + retention policy
status: Done
assignee: []
created_date: '2026-08-26 04:13'
updated_date: '2026-08-30 23:14'
labels:
  - training
  - checkpoint
  - infrastructure
milestone: m-1
dependencies: []
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
modified_files:
  - src/kestrel/train/trainer.py
  - src/kestrel/train/pretrain.py
  - scripts/run_pretrain.py
parent_task_id: TASK-005
priority: high
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Long 12GiB pretrain runs need pause/resume and bounded checkpoint storage.

Current behavior (as of the 50M 12GiB pretrain config):
- src/kestrel/train/trainer.py saves checkpoints every save_every steps and a final checkpoint.
- src/kestrel/model/io.py save() writes only model weights to weights.npz.
- train() always starts at step 0 with a freshly constructed AdamW optimizer.
- scripts/run_pretrain.py has no --resume flag and PretrainConfig has no resume field.
- best_val_loss is tracked in memory but no best checkpoint is written.
- All step_NNNNNN checkpoint directories are retained indefinitely.

This parent task tracks two related trainer capabilities:
1. Checkpoint retention: keep only N latest step checkpoints plus a best-validation checkpoint and the final checkpoint.
2. Resume: restart a paused or crashed run from a full training checkpoint, restoring model weights, optimizer state, step count, LR schedule horizon, best validation loss, and dataset position.

The existing 50M run uses save_every: 2000. The user will manage existing disk usage manually; this work prevents future long runs from silently filling the disk and makes interruption safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child tasks for checkpoint retention and resumable checkpoints are Done
- [x] #2 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Complete TASK-005.12.01: checkpoint retention + best-val checkpoint. 2. Complete TASK-005.12.02: full resumable training checkpoints. 3. Keep make check green after each subtask. 4. Do not modify the currently running 50M pretrain process; these changes apply to future runs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-28: Both child tasks are Done. make check passed on 2026-08-28: ruff check clean, ruff format clean, mypy clean, 165 pytest tests passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Trainer checkpoint resume + retention is complete via TASK-005.12.01 and TASK-005.12.02. Full checkpoints are self-describing and resumable; retention keeps bounded step checkpoints plus best/final. make check is green.
<!-- SECTION:FINAL_SUMMARY:END -->
