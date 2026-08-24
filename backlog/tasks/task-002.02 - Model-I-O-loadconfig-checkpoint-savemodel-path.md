---
id: TASK-002.02
title: 'Model I/O: load(config, checkpoint) + save(model, path)'
status: To Do
assignee: []
created_date: '2026-08-24 00:15'
updated_date: '2026-08-24 00:17'
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
- [ ] #1 load(config) returns a randomly-initialized Kestrel model
- [ ] #2 save(model, path) + load(config, path) round-trips a checkpoint (weights identical after reload)
- [ ] #3 Checkpoints are written to the checkpoints/<phase>/<name>/ directory convention
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
<!-- SECTION:NOTES:END -->
