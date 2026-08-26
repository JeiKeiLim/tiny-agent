---
id: TASK-005.10.04
title: Add 12GiB pretrain configs and short validation runs
status: To Do
assignee: []
created_date: '2026-08-26 01:36'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies:
  - TASK-005.10.02
  - TASK-005.10.03
modified_files:
  - configs/kestrel/150m/pretrain-12g.yaml
  - configs/kestrel/50m/pretrain-12g.yaml
  - tests/test_pretrain.py
parent_task_id: TASK-005.10
priority: high
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add pretrain configs that use the expanded data/corpus-12g corpus and verify that the document-aware pretrain stack integrates with it.

Files:
- create configs/kestrel/150m/pretrain-12g.yaml
- create configs/kestrel/50m/pretrain-12g.yaml
- modify tests/test_pretrain.py if YAML load coverage is needed
- optionally create a tiny 12g smoke pretrain config if a short real-data run is needed

150M full-corpus config:
- model: configs/kestrel/150m/model.yaml
- tokenizer: checkpoints/tokenizer/tokenizer.json
- corpus: configs/kestrel/corpus-12g.yaml
- total_tokens: null
- trainer:
  - lr: 0.0003
  - weight_decay: 0.1
  - batch_size: 4
  - seq_len: 1024
  - num_steps: 0
  - warmup_steps: 500
  - grad_clip: 1.0
  - save_every: 2000
  - log_every: 100
  - eval_every: 1000
  - eval_iters: 10
  - output_dir: checkpoints/pretrain/150m-12g

50M Chinchilla-capped config:
- model: configs/kestrel/50m/model.yaml
- tokenizer: checkpoints/tokenizer/tokenizer.json
- corpus: configs/kestrel/corpus-12g.yaml
- total_tokens: 1013504000
- trainer:
  - lr: 0.0003
  - weight_decay: 0.1
  - batch_size: 8
  - seq_len: 1024
  - num_steps: 0
  - warmup_steps: 500
  - grad_clip: 1.0
  - save_every: 2000
  - log_every: 100
  - eval_every: 1000
  - eval_iters: 10
  - output_dir: checkpoints/pretrain/50m-12g

Validation:
1. Verify both YAML files load into PretrainConfig under strict Pydantic rules.
2. Build PretrainDataset for data/corpus-12g/train using the 150M shape and verify estimated_steps is close to train_manifest_tokens / (4 * 1024).
3. Build PretrainDataset for data/corpus-12g/train using the 50M capped shape and verify estimated_steps is close to 1013504000 / (8 * 1024).
4. Run a short smoke pretrain of at most 100 steps, either on data/corpus-12g-smoke or on a tiny local corpus, and verify finite loss plus a final checkpoint.
5. Do not start the full 150M run in this task.

Expected estimates:
- 150M full corpus: about 725000 steps for roughly 2.97B train tokens at batch_size 4 and seq_len 1024.
- 50M capped: about 123718 steps for 1013504000 tokens at batch_size 8 and seq_len 1024.

Gate: make check green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 configs/kestrel/150m/pretrain-12g.yaml loads into PretrainConfig
- [ ] #2 configs/kestrel/50m/pretrain-12g.yaml loads into PretrainConfig
- [ ] #3 150M estimated_steps matches the 12g train manifest within 5%
- [ ] #4 50M estimated_steps matches the 1013504000 token cap within 5%
- [ ] #5 A short smoke pretrain run completes with finite loss and writes a checkpoint
- [ ] #6 make check is green
<!-- AC:END -->
