---
id: TASK-005.12.02
title: Resume pretraining from a full training checkpoint
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-26 04:14'
updated_date: '2026-08-26 05:24'
labels:
  - training
  - checkpoint
  - data
milestone: m-1
dependencies:
  - TASK-005.12.01
modified_files:
  - src/kestrel/train/checkpoint.py
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
- [x] #1 PretrainConfig and run_pretrain.py support resuming from a checkpoint directory
- [x] #2 Checkpoint includes model weights, optimizer state, step, best val loss, schedule horizon, and dataset state
- [x] #3 Resuming continues from saved step + 1 with the original LR schedule horizon
- [x] #4 Dataset position round-trips exactly without restarting the corpus from the beginning
- [x] #5 Incompatible checkpoints fail with a clear validation error
- [x] #6 make check is green
- [x] #7 CLI supports --resume CHECKPOINT_DIR, where the argument is a step/best/final checkpoint directory, not the parent output_dir
- [x] #8 Each full checkpoint stores raw + resolved config snapshots and artifact hashes so the run configuration survives later YAML edits
- [x] #9 Resume validates the current config against the checkpoint config snapshot and rejects mismatches with a clear error
- [x] #10 The live run appends run.jsonl to output_dir, and each checkpoint includes a run.jsonl snapshot up to that checkpoint
- [x] #11 Checkpoint writes are crash-safe via temp-directory + rename or an equivalent state-last strategy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refactor PretrainDataset into a resumable iterator with state_dict/load_state_dict and exact round-trip tests.
2. Add full-checkpoint helpers: weights.npz, optimizer.npz, state.json, config snapshots, run.jsonl snapshot, atomic temp+rename, and strict resume validation.
3. Update trainer.py to write live run.jsonl, save full step/best/final checkpoints, support resume, and continue from saved step + 1.
4. Update pretrain.py and scripts/run_pretrain.py for PretrainConfig.resume and --resume CHECKPOINT_DIR.
5. Add unit/integration tests and make make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-26 pre-implementation design decisions (compaction-safe):
- CLI usage: uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain.yaml --resume checkpoints/pretrain/50m/step_000010. --resume takes the checkpoint directory itself and overrides PretrainConfig.resume.
- Checkpoint layout: step_NNNNNN/, best/, and final/ should all be full resumable checkpoints containing weights.npz, optimizer.npz, state.json, config snapshots, and a run.jsonl snapshot.
- Config snapshots: store raw pretrain/model/corpus YAML text plus resolved JSON dumps under the checkpoint config/ directory. state.json should also store machine-checkable hashes/metadata for tokenizer, corpus manifest/config, model config, pretrain config, batch_size, seq_len, seed, total_tokens, optimizer settings, schedule_steps, step, best/last losses, and dataset_state.
- Resume compatibility: compare current CLI config against the checkpoint snapshot. Reject missing state.json/optimizer.npz, wrong model shape, wrong tokenizer/corpus fingerprint, wrong batch_size/seq_len/seed/total_tokens, or incompatible optimizer settings. Existing weights-only checkpoints must fail clearly.
- Logging: write a live append-only output_dir/run.jsonl while training. At checkpoint time, copy the current run.jsonl into the checkpoint as a snapshot. Pruning may delete old step_NNNNNN directories, but must never delete the root run.jsonl, best, or final.
- Crash safety: preferred checkpoint write strategy is temp directory + rename, or at minimum write state.json last so a partial checkpoint is not resumable.
- Dataset resume: refactor PretrainDataset into an iterator object with state_dict()/load_state_dict(). State must include scheduler RNG state, active source indices, emitted tokens per source, emitted_total, next_doc_id, current token/doc buffers, partial batch rows, and per-source shuffled offset positions. Do not store full shuffled offset arrays; recompute them deterministically from corpus file + seed and restore only the position.
- Optimizer resume: flatten MLX optimizer.state, save arrays to optimizer.npz, restore into a freshly constructed AdamW with matching hyperparameters. Verify m/v arrays are restored.
- Trainer changes: use manual next() iteration, accept start_step / existing optimizer / initial best_val_loss / schedule horizon, continue from saved step + 1, and return global final step count.
- Suggested implementation order: 1) resumable PretrainDataset iterator + exact round-trip tests, 2) checkpoint save/load helpers + config/hash validation, 3) trainer full-checkpoint/resume support, 4) pretrain.py + CLI wiring, 5) integration tests and make check.

2026-08-26 implementation complete:
- Added src/kestrel/train/checkpoint.py with CheckpointContext, save_full_checkpoint, read_checkpoint_state, load_optimizer_state, and sha256_file.
- Refactored PretrainDataset into PretrainDatasetIterator with state_dict/load_state_dict. The iterator drains the existing token buffer before fetching another document, preserving the original batch/scheduler semantics.
- Trainer now writes live output_dir/run.jsonl, saves full step/best/final checkpoints using temp-directory + rename, supports ResumeState, and continues from saved step + 1.
- pretrain.py adds PretrainConfig.resume, checkpoint context with raw + resolved config snapshots and artifact hashes, strict resume validation, optimizer restore, and dataset iterator restore.
- run_pretrain.py adds --resume CHECKPOINT_DIR and passes the pretrain config path so raw pretrain.yaml can be snapshotted.
- make check green: 140 tests passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented true pause/resume for pretraining. Full checkpoints now contain weights.npz, optimizer.npz, state.json, config snapshots, and a run.jsonl snapshot; the CLI supports --resume CHECKPOINT_DIR. Verified with dataset state round-trip tests, trainer resume tests, pretrain crash/resume integration tests, CLI tests, and make check (140 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
