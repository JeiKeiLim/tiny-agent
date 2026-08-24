---
id: TASK-005.03
title: 'Shared trainer (train/trainer.py) - optimizer, step loop, checkpoint'
status: Done
assignee:
  - 7477cb22-9a4d-4bfc-9c19-64c3784d2b3a
created_date: '2026-08-24 01:55'
updated_date: '2026-08-24 07:15'
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
- [x] #1 Trainer runs num_steps on a tiny model + synthetic (input, target) batches; loss finite every step and final < initial
- [x] #2 LR schedule: warmup ramps lr from 0 to peak over warmup_steps (assert a few schedule values)
- [x] #3 checkpoint saved every save_every steps and at the end; reloads via kestrel.model.io.load() with matching weights
- [x] #4 tests/test_trainer.py uses a TINY model (vocab ~400, 2 layers, hidden 64) + synthetic random dataset so it runs fast; make check green
- [x] #5 in-loop validation: takes a val_dataset (iterable of (input,target) batches); every eval_every steps computes val loss (mean over a bounded number of val batches) and logs it; tracks best val loss
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADDED (2026-08-24, from 005.01 split work): the corpus now has a held-out val slice (data/corpus/val). The trainer must compute in-loop validation loss: add eval_every to TrainerConfig; take a val_dataset; every eval_every steps compute val loss (mean over up to N val batches, N bounded) and log it alongside train loss; track best val loss (optionally save best checkpoint). Standard LLM practice - detects overfitting + picks the stopping point.

Implemented (2026-08-24): src/kestrel/train/trainer.py (TrainerConfig strict Pydantic; TrainResult; lr_at warmup+cosine; _clip_grads global-norm; _batch_loss; estimate_val_loss; train loop) + tests/test_trainer.py (5 tests). make check green (68 tests, +5). Real smoke: tiny Kestrel (vocab 16384, 1.1M params) on data/corpus/train+val -> train loss 10.11->8.38, val loss 8.39->6.68 (monotonic), checkpoint reload matches in-memory.

KEY DECISIONS / GOTCHAS:
- Loss: cross_entropy(logits[:, :-1], target[:, :-1], reduction='mean') - matches scripts/check_model.py. model(x) returns logits directly (NOT a (logits,cache) tuple).
- MLX idiom (verified empirically, mlx 0.32.1): opt = optim.AdamW(learning_rate=lr, weight_decay=wd) [hyperparams only, NOT the model]; value, grad = mx.value_and_grad(loss_fn)(model); opt.update(model, grad) [update takes (model, gradients)].
- @mx.compile OMITTED (deviation from original design): MLX compile cache keys on arg shapes and captures the param arrays at trace time, so in-place opt.update changes are invisible to the cached graph (stale params). Verified: compiled step did NOT decrease loss (4.86->4.93) while the eager step did (5.40->3.10). Passing model as a compiled arg also fails (compile flattens a module arg to a param dict -> 'dict not callable'). No benefit at 50M scale, so the step runs eagerly. Documented in the module docstring.
- Grad clip is GLOBAL-NORM on the gradients (not value-clip on params as the draft design loosely said).
- TINY test model is vocab=64/hidden=32/2 layers (smaller than the AC's illustrative ~400/64) for speed; test runs ~1s.
- mypy: MLX .item() returns scalar (int|float|complex) -> cast(float, ...); value_and_grad grad is Any; tree_flatten returns list|dict -> cast to list[tuple[str,Any]] and accumulate norm with an mx.array (builtin sum over arrays is ill-typed).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built the shared MLX trainer (train/trainer.py): TrainerConfig (strict Pydantic: lr, weight_decay, batch_size, seq_len, num_steps, warmup_steps, grad_clip, save_every, log_every, eval_every, eval_iters, output_dir), TrainResult (final_loss, num_steps, best_val_loss, history of (step, train_loss, val_loss|None)), lr_at (linear warmup 0->lr over warmup_steps then cosine decay to 0), _clip_grads (global-norm on gradients), _batch_loss (next-token CE cross_entropy(logits[:,:-1], target[:,:-1]), matches check_model), estimate_val_loss (mean over up to eval_iters val batches), train(model, dataset, val_dataset, config) -> TrainResult (AdamW step loop, in-loop val loss every eval_every, best-val tracking, checkpoint every save_every + final). tests/test_trainer.py (5 tests; TINY model vocab 64/hidden 32/2 layers + synthetic counting batches). Verified: make check green (68 tests) + real smoke (tiny Kestrel vocab 16384 on data/corpus/train+val: train loss ~10.4->8.9, val loss 8.5->6.9 monotonic, checkpoint reload matches in-memory). DECISION: @mx.compile omitted - MLX compile cache captures param arrays at trace time so in-place optimizer.update is invisible to the cached graph (stale params; verified: compiled step did NOT decrease loss, eager did); no benefit at 50M scale. MLX idiom (0.32.1): optim.AdamW(learning_rate=lr, weight_decay=wd); value, grad = mx.value_and_grad(loss_fn)(model); opt.update(model, grad).
<!-- SECTION:FINAL_SUMMARY:END -->
