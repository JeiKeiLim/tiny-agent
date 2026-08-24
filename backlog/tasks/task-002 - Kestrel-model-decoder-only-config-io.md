---
id: TASK-002
title: Kestrel model (decoder-only) + config + io
status: To Do
assignee: []
created_date: '2026-08-21 06:44'
updated_date: '2026-08-24 00:27'
labels: []
milestone: m-0
dependencies:
  - TASK-001
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Kestrel decoder-only transformer per plan §9: pre-norm RMSNorm, RoPE, SwiGLU, GQA, tied embeddings, no biases, dropout 0. Split into sub-tasks: TASK-002.01 (model/kestrel.py — the transformer + count_params), TASK-002.02 (model/io.py — load(config, checkpoint=None) + save(model, path)), and TASK-002.03 (scripts/check_model.py — manual smoke-test CLI, built on the load() factory so it can load a trained checkpoint later).

Scope notes (decided 2026-08-24):
- model/config.py (ModelConfig) + configs/kestrel/{50m,150m}/model.yaml already exist (TASK-001) — the config half is done.
- load() is the Kestrel model factory ONLY (random-init / from-Kestrel-checkpoint). The Qwen3/pretrained loader is a SEPARATE Track B build (step 7, model/pretrained.py) and does NOT route through this load().
- Checkpoints use the checkpoints/<phase>/<name>/ directory convention (MLX-native).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Instantiate Kestrel-50M and -150M from config; param counts land near ~50M / ~150M
- [ ] #2 Forward pass on random token IDs returns logits of shape (B, T, vocab) with finite loss
- [ ] #3 model/io.py load(config) + save(model, path) round-trips a checkpoint
<!-- AC:END -->
