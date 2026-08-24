---
id: TASK-005.03
title: 'Shared trainer (train/trainer.py) - optimizer, step loop, checkpoint'
status: To Do
assignee: []
created_date: '2026-08-24 01:55'
updated_date: '2026-08-24 06:34'
labels: []
milestone: m-1
dependencies: []
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
parent_task_id: TASK-005
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The reusable training engine shared by pretrain/SFT/RL. Owns optimizer setup, the step loop, gradient clipping, logging, in-loop validation, and checkpointing. Each phase plugs in its own dataset + loss. M1 = full fine-tuning only (PEFT selection is a later Track B extension - leave a hook, do not build it).

Files to create:
- src/kestrel/train/trainer.py
- src/kestrel/train/config.py  (TrainerConfig)
- tests/test_trainer.py

Design (Trainer class or train() fn):
- inputs: model (mlx nn.Module), dataset (iterable of (input, target) batches), val_dataset (iterable of (input, target) batches, held-out), TrainerConfig, optional loss fn (default next-token cross-entropy).
- optimizer: mlx.optimizers.AdamW(lr, ...) with weight_decay.
- LR schedule: linear warmup over warmup_steps, then decay (cosine or linear) to a min.
- step loop over num_steps:
  - logits = model(input)   [Kestrel.forward returns logits (B, T, V)]
  - loss = mlx.nn.losses.cross_entropy(logits[:, :-1], target[:, :-1], reduction='mean')  # next-token shift; matches scripts/check_model.py
  - loss.backward(); optimizer.clip_gradients(model, grad_clip); optimizer.step()
  - log train loss every log_every steps.
  - every eval_every steps: estimate val loss = mean over eval_iters val batches (nanoGPT estimate_loss pattern); log it alongside train loss; track best val loss.
  - save checkpoint every save_every steps + at end (via kestrel.model.io.save); also save the best-val checkpoint.
- returns a small history [(step, train_loss, val_loss_or_None)] + final checkpoint path.

TrainerConfig fields (Pydantic, strict): lr, weight_decay, batch_size, seq_len, num_steps, warmup_steps, grad_clip, save_every, log_every, eval_every, eval_iters, output_dir.

Quantitative targets:
- on a tiny model + synthetic dataset, runs num_steps steps; loss finite every step and final < initial.
- val loss computed + logged every eval_every steps (finite; tracks train loss).
- a checkpoint is written and reloads via kestrel.model.io.load() (weights match).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Trainer runs num_steps on a tiny model + synthetic (input, target) batches; loss finite every step and final < initial
- [ ] #2 LR schedule: warmup ramps lr from 0 to peak over warmup_steps (assert a few schedule values)
- [ ] #3 checkpoint saved every save_every steps and at the end; reloads via kestrel.model.io.load() with matching weights
- [ ] #4 tests/test_trainer.py uses a TINY model (vocab ~400, 2 layers, hidden 64) + synthetic random dataset so it runs fast; make check green
- [ ] #5 in-loop validation: takes a val_dataset (iterable of (input,target) batches); every eval_every steps computes val loss (mean over a bounded number of val batches) and logs it; tracks best val loss
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADDED (2026-08-24, from 005.01 split work): the corpus now has a held-out val slice (data/corpus/val). The trainer must compute in-loop validation loss: add eval_every to TrainerConfig; take a val_dataset; every eval_every steps compute val loss (mean over up to N val batches, N bounded) and log it alongside train loss; track best val loss (optionally save best checkpoint). Standard LLM practice - detects overfitting + picks the stopping point.
<!-- SECTION:NOTES:END -->
