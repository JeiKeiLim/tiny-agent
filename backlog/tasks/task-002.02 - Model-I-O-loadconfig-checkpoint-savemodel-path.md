---
id: TASK-002.02
title: 'Model I/O: load(config, checkpoint) + save(model, path)'
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-24 00:15'
updated_date: '2026-08-24 01:03'
labels: []
milestone: m-0
dependencies:
  - TASK-002.01
parent_task_id: TASK-002
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement src/kestrel/model/io.py: load(config, checkpoint=None) factory for Kestrel models (random-init when checkpoint is None, else load from a Kestrel checkpoint) and save(model, path). Checkpoints use the checkpoints/<phase>/<name>/ directory convention (MLX-native mx.save_weights / mx.load_weights). This is the Kestrel model factory ONLY — the Qwen3/pretrained loader is a separate Track B build (step 7) and does NOT route through this load().
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 load(config) returns a randomly-initialized Kestrel model
- [x] #2 save(model, path) + load(config, path) round-trips a checkpoint (weights identical after reload)
- [x] #3 Checkpoints are written to the checkpoints/<phase>/<name>/ directory convention
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
New file src/kestrel/model/io.py. Depends on TASK-002.01 (Kestrel model). MLX-native.

- save(model, path): path is a DIRECTORY following the checkpoints/<phase>/<name>/ convention (e.g. checkpoints/pretrain/kestrel-50m/). Create the dir, then mx.save_weights(path, dict of the model parameters).
- load(config, checkpoint=None): build Kestrel(config). If checkpoint is None, return the random-init model. Else load weights from the checkpoint dir (mx.load_weights) into the model and return it.
- Kestrel-only factory: NO pretrained/Qwen3 branch (that is step 7, model/pretrained.py, a separate loader).

Tests: tests/test_model_io.py. Save a model (tiny config or 50M) to a tmp dir, reload via load(config, path), assert weights identical after reload (mx.array_equal per param, or equal forward output on a fixed input). Also assert load(config) with no checkpoint returns a fresh random-init model.

Gate: make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decisions/gotchas:
- Tied embedding is stored once (the embed weight) - there is no separate lm_head to save/load.
- mx.save_weights writes to a directory (weights.npz inside); mx.load_weights reads from the same dir.
- Round-trip check: compare each param array with mx.array_equal, or compare a fixed-input forward pass before/after reload.
- Checkpoint path example: checkpoints/pretrain/kestrel-50m/ (phase=pretrain, name=kestrel-50m). The <phase> and <name> come from the calling phase config, not hardcoded in io.py.
- reference: plan doc-001 §6 (thin model interface, checkpoint handoff).

MLX 0.32.1 gotcha: mx.save_weights / mx.load_weights do NOT exist in mlx.core (verified via dir(mx)/hasattr). Available: mx.savez(file, **arrays), mx.load(file), mx.save_safetensors(file, dict), and the nn.Module.save_weights(file)/load_weights(file_or_weights) methods. Solution: use model.save_weights(dir/"weights.npz") + model.load_weights(dir/"weights.npz") (nn.Module methods wrap mx.savez/mx.load, flatten/unflatten params, strict=True validates exact match). Still yields the checkpoints/<phase>/<name>/weights.npz dir convention. Tied embedding: only embed.weight is saved (no lm_head), so round-trip is clean. mypy: model.save_weights/load_weights type-check fine (Kestrel base is Any via mlx.nn follow_imports=skip).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented src/kestrel/model/io.py: save(model, path) creates the checkpoints/<phase>/<name>/ dir and writes weights.npz; load(config, checkpoint=None) builds Kestrel(config) and, when checkpoint is given, strictly loads weights.npz (else random-init). Kestrel-only factory (no pretrained/Qwen3 branch). Tests in tests/test_model_io.py (3): random-init finite weights, checkpoint dir convention, save/load round-trip with per-param mx.array_equal. Verified on real 50M config: 202.7 MB weights.npz, reload matches. make check green (mypy 24 files, 40 tests).
<!-- SECTION:FINAL_SUMMARY:END -->
