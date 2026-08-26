---
id: TASK-005.12.02
title: Resume pretraining from a full training checkpoint
status: To Do
assignee: []
created_date: '2026-08-26 04:14'
labels:
  - training
  - checkpoint
  - data
milestone: m-1
dependencies:
  - TASK-005.12.01
modified_files:
  - src/kestrel/train/trainer.py
  - src/kestrel/train/pretrain.py
  - scripts/run_pretrain.py
  - src/kestrel/data/pretrain_dataset.py
  - tests/test_trainer.py
  - tests/test_pretrain.py
  - tests/test_pretrain_dataset.py
parent_task_id: TASK-005.12
priority: high
ordinal: 31000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add true pause/resume support for pretraining. Today a checkpoint contains only model weights, so restarting from it loses optimizer moments, step count, LR schedule position, best validation loss, and dataset position.

Files:
- create or modify src/kestrel/train/checkpoint.py for full train-state save/load helpers
- modify src/kestrel/model/io.py only if weights saving/loading needs to be reused
- modify src/kestrel/train/trainer.py
- modify src/kestrel/train/pretrain.py
- modify scripts/run_pretrain.py
- modify src/kestrel/data/pretrain_dataset.py
- modify tests/test_trainer.py, tests/test_pretrain.py, tests/test_pretrain_dataset.py

Checkpoint format:
Each resumable checkpoint directory contains:
- weights.npz: model weights
- optimizer.npz: flattened MLX optimizer state arrays
- state.json: metadata including format version, step, best_val_loss, schedule_steps/horizon, corpus config fingerprint, model config fingerprint, trainer batch_size/seq_len/seed/total_tokens, and dataset state

Resume API:
- Add PretrainConfig.resume: str | None = None.
- Add scripts/run_pretrain.py --resume CHECKPOINT_DIR, which overrides config.resume.
- pretrain() loads model weights, optimizer state, trainer state, and dataset state before continuing.
- The loop continues from saved step + 1 and uses the original schedule horizon.
- best_val_loss is restored so a later validation improvement is compared against the true best.
- Incompatible checkpoints must raise a clear ValueError (wrong model shape, batch_size, seq_len, corpus fingerprint, seed, total_tokens, or missing state.json).

Dataset resume design:
- Do not use restart-from-beginning fast-forward for the 12GiB single-pass run; it would be slow and obscures exact data position.
- Make the PretrainDataset iteration checkpointable without changing its deterministic document order or domain mixing behavior.
- Introduce an iterator object or equivalent API exposing state_dict() and load_state_dict().
- Dataset state must include:
  - random.Random scheduler state
  - active source indices
  - emitted tokens per source and emitted_total
  - next_doc_id
  - current packing buffer tokens and doc ids
  - per-source shuffled-document iterator position (source index, shuffled offset position, exhausted flag)
- On restore, shuffled offsets are recomputed deterministically from the corpus file and per-source seed; only the position is restored, not the full offset array.
- doc_ids must continue from the restored next_doc_id so document-aware attention semantics remain consistent.

Optimizer resume:
- Use MLX optimizer.state (a mutable dict) and tree_flatten/tree_map to save and restore arrays.
- Restore state into a freshly constructed AdamW with the same hyperparameters before the next update.
- Include optimizer step/moment arrays in optimizer.npz.

Behavior and gotchas:
- Existing weights-only checkpoints are not resumable and must be rejected clearly.
- final and best checkpoints should also be resumable once this task lands.
- Resume must work after a SIGINT/crash at the last completed checkpoint, not only after a clean stop.
- The corpus builder may run first on resume; it should skip because the corpus is complete.
- History in TrainResult can begin at the resumed process, but result.num_steps must be the final global step count.

Tests:
- Unit test: PretrainDataset iterator state round-trips; after saving state following N batches, a restored iterator yields the same next batch as the original iterator.
- Unit test: trainer saves optimizer.npz and state.json alongside weights.npz.
- Integration test: train a tiny model for 5 steps, stop, resume to 10 steps, assert global num_steps=10, finite loss, and final checkpoint written.
- Integration test: resuming restores optimizer state (saved moment arrays are nonzero and loaded into the new optimizer).
- Config test: incompatible checkpoint metadata raises ValueError.
- CLI test: run_pretrain.py accepts --resume and passes it into pretrain().

Quantitative targets:
- Resume from step N continues at step N+1 with zero repeated optimizer updates.
- Dataset state round-trip is exact for at least 100 tiny batches in tests.
- make check is green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PretrainConfig and run_pretrain.py support resuming from a checkpoint directory
- [ ] #2 Checkpoint includes model weights, optimizer state, step, best val loss, schedule horizon, and dataset state
- [ ] #3 Resuming continues from saved step + 1 with the original LR schedule horizon
- [ ] #4 Dataset position round-trips exactly without restarting the corpus from the beginning
- [ ] #5 Incompatible checkpoints fail with a clear validation error
- [ ] #6 make check is green
<!-- AC:END -->
