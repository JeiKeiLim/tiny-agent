---
id: TASK-002
title: Kestrel model (decoder-only) + config + io
status: To Do
assignee: []
created_date: '2026-08-21 06:44'
updated_date: '2026-08-21 07:12'
labels: []
milestone: m-0
dependencies:
  - TASK-001
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Kestrel decoder-only transformer per §9: RMSNorm, RoPE, SwiGLU, GQA, tied embeddings, no biases, dropout 0. Include model/config.py (50M: 15L/512H/8Q/2KV/1408F; 150M: 32L/640H/10Q/2KV/1728F; vocab 16384, ctx 2048) and model/io.py (load(config) factory: random-init / from-checkpoint / pretrained + save(model, path)).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Instantiate Kestrel-50M and -150M from config; param counts land near ~50M / ~150M
- [ ] #2 Forward pass on random token IDs returns logits of shape (B, T, vocab) with finite loss
- [ ] #3 model/io.py load(config) + save(model, path) round-trips a checkpoint
<!-- AC:END -->
