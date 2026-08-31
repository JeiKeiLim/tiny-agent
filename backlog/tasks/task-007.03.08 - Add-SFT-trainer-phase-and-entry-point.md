---
id: TASK-007.03.08
title: Add SFT trainer phase and entry point
status: To Do
assignee: []
created_date: '2026-08-31 01:20'
labels:
  - sft
  - training
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.02
parent_task_id: TASK-007.03
priority: high
ordinal: 48000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the SFT training phase and command entry point.

Depends on:
- TASK-007.03.02

Files:
- src/kestrel/train/sft.py
- scripts/run_sft.py
- configs/kestrel/50m/sft.yaml
- configs/kestrel/150m/sft.yaml
- tests/train/test_sft.py

Scope:
- Add strict SFT config models for:
  - model checkpoint to load, initially checkpoints/pretrain/50m/final for 50M
  - dataset: SFTDatasetConfig
  - optimizer hyperparameters
  - epochs or max_steps
  - gradient accumulation if needed
  - checkpoint directory
  - logging/run.jsonl behavior
- Reuse the shared trainer where practical.
- Train with input/target/loss_mask from SFTDataset.
- Support checkpoint save/resume consistent with existing checkpoint invariants.
- Write run.jsonl or equivalent training log.
- 50M config is the first execution target.
- 150M config must exist and be structurally valid, but 150M is not the first run.

Defaults to propose:
- context_length: 1024 for first validation runs
- epochs: 1 for smoke, configurable up to 2 or 3
- lower SFT learning rate than pretrain
- checkpoint every N steps and at epoch end

Acceptance:
- scripts/run_sft.py can load a pretrain checkpoint and run a tiny dummy SFT dataset in tests.
- Checkpoint resume works for SFT.
- 50M and 150M configs load through the strict config loader.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SFT trainer reads SFTDataset batches and applies loss masking
- [ ] #2 SFT checkpointing and resume follow existing checkpoint invariants
- [ ] #3 50M and 150M SFT configs are strict and loadable
- [ ] #4 Tiny dummy SFT training run passes in tests
- [ ] #5 make check passes
<!-- AC:END -->
