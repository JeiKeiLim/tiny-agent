---
id: TASK-005.08.03
title: 'Model/trainer: document-aware attention, position reset, auto num_steps'
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-26 00:25'
updated_date: '2026-08-26 00:51'
labels:
  - model
  - trainer
  - pretraining
milestone: m-1
dependencies:
  - TASK-005.08.02
documentation:
  - doc-003
modified_files:
  - src/kestrel/model/kestrel.py
  - src/kestrel/train/trainer.py
  - src/kestrel/train/pretrain.py
  - tests/test_model_kestrel.py
  - tests/test_trainer.py
  - tests/test_pretrain.py
  - configs/kestrel/50m/pretrain.yaml
  - configs/kestrel/150m/pretrain.yaml
parent_task_id: TASK-005.08
priority: high
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the model and trainer respect document boundaries and support auto dataset-exhaustion step horizon.

Depends on TASK-005.08.02.

Current problem:
- Kestrel attention is causal only and does not know document boundaries.
- RoPE positions continue across packed documents.
- Trainer assumes dataset yields only input and target.
- Trainer uses num_steps both as stop cap and LR schedule horizon.

Outcome:
- Kestrel.__call__ accepts optional doc_ids of shape (B, T).
- If doc_ids is None, behavior remains normal causal attention.
- If doc_ids is provided, attention is allowed only when key_pos <= query_pos and doc_ids[key_pos] == doc_ids[query_pos].
- RoPE positions reset to 0 at every doc_id change.
- Trainer consumes (input, target, doc_ids) batches and passes doc_ids to model for train and validation.
- TrainerConfig.num_steps <= 0 means run until dataset exhaustion.
- When num_steps <= 0, pretrain derives schedule_steps from train_dataset.estimated_steps().
- LR schedule uses schedule_steps, while the loop still stops on dataset exhaustion.

Files to modify:
- src/kestrel/model/kestrel.py
- src/kestrel/train/trainer.py
- src/kestrel/train/pretrain.py
- tests/test_model_kestrel.py
- tests/test_trainer.py
- tests/test_pretrain.py

Implementation notes:
- At seq_len 1024/2048, a dense boolean or additive mask is acceptable.
- Do not add third-party varlen attention for M1.
- Generation should remain unchanged when doc_ids is None or all zeros.
- Log the derived schedule_steps when auto mode is used.

Acceptance targets:
- Unit test proves a token cannot attend to a previous token with a different doc_id.
- Unit test proves positions reset when doc_id changes.
- Unit test proves normal generation/check_model path still works without doc_ids.
- Unit test proves num_steps <= 0 uses dataset estimated_steps for LR schedule.
- make check is green.

Reference: doc-003.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A token cannot attend to a previous token with a different doc_id
- [x] #2 RoPE positions reset when doc_id changes
- [x] #3 Normal generation/check_model path still works without doc_ids
- [x] #4 num_steps <= 0 uses dataset estimated_steps for LR schedule
- [x] #5 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add optional doc_ids to Kestrel forward. 2. Build document-aware causal attention mask. 3. Reset RoPE positions at doc_id changes. 4. Update trainer to pass doc_ids and support schedule_steps separate from stop cap. 5. Update pretrain to derive schedule_steps from estimated_steps when num_steps <= 0. 6. Add model/trainer/pretrain tests. 7. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added optional doc_ids to Kestrel, same-document causal attention mask, document-reset RoPE positions, trainer doc_ids pass-through, and auto num_steps <= 0 with schedule_steps from dataset.estimated_steps(). Updated 50M/150M pretrain configs to num_steps=0. make check green: 101 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Model and trainer are document-aware: Kestrel accepts doc_ids, restricts attention to same-document causal tokens, resets RoPE positions at document boundaries, and preserves the normal causal path when doc_ids is None. Trainer passes doc_ids through train/validation and supports dataset-exhaustion runs with an estimated LR schedule horizon.
<!-- SECTION:FINAL_SUMMARY:END -->
