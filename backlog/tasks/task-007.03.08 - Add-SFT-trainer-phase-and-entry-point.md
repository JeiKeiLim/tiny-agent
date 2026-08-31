---
id: TASK-007.03.08
title: Add SFT trainer phase and entry point
status: Done
assignee:
  - '@Jongkuk Lim'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 01:55'
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
- [x] #1 SFT trainer reads SFTDataset batches and applies loss masking
- [x] #2 SFT checkpointing and resume follow existing checkpoint invariants
- [x] #3 50M and 150M SFT configs are strict and loadable
- [x] #4 Tiny dummy SFT training run passes in tests
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend SFTDataset with a resumable SFTDatasetIterator (iterator/load_iterator/state_dict/load_state_dict) and add epochs to SFTDatasetConfig so SFT checkpoints can satisfy the existing resume invariants.
2. Extend the shared trainer with TrainerConfig.use_loss_mask. When enabled, the third batch element is a loss mask aligned to input positions; the loss uses logits[:, :-1], target[:, :-1], and mask[:, 1:] with a masked mean.
3. Add src/kestrel/train/sft.py with strict SFTConfig (model, checkpoint, dataset, optional val_dataset, trainer, resume), shape validation between dataset and trainer, checkpoint context hashes, and resume validation consistent with pretrain.
4. Add scripts/run_sft.py mirroring run_pretrain.py, including --resume override.
5. Add configs/kestrel/50m/sft.yaml and configs/kestrel/150m/sft.yaml with context_length 1024, use_loss_mask true, lower SFT LR, and 50M as the first execution target.
6. Add tests/train/test_sft.py covering strict config, YAML loading, masked loss, tiny end-to-end SFT, resume no-op, incompatible resume rejection, simulated-crash resume, and CLI resume override.
7. Update tests/data/test_sft_dataset.py for the new resumable iterator and epochs behavior.
8. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added SFTConfig and sft() in src/kestrel/train/sft.py, scripts/run_sft.py, and strict 50M/150M SFT YAML configs.
Extended TrainerConfig with use_loss_mask. The shared trainer now computes a masked mean over logits[:, :-1], target[:, :-1], and input-aligned mask[:, 1:].
Extended SFTDatasetConfig with epochs and added SFTDatasetIterator with iterator/load_iterator/state_dict/load_state_dict so SFT checkpoints carry resumable dataset state.
SFT resume validation checks initial checkpoint, model config, dataset config, training-relevant trainer fields, tokenizer hash, and dataset input hashes.
Added tests/train/test_sft.py and extended tests/data/test_sft_dataset.py.
Validation: make check passed with 214 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the M2 SFT training phase: strict SFT config, masked SFT loss in the shared trainer, resumable SFT dataset iteration, run_sft.py, 50M/150M configs, and tests covering end-to-end tiny SFT plus checkpoint resume.
<!-- SECTION:FINAL_SUMMARY:END -->
