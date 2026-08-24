---
id: TASK-005.05
title: >-
  Pretrain loop + entry point (train/pretrain.py, run_pretrain.py,
  50m/pretrain.yaml)
status: To Do
assignee: []
created_date: '2026-08-24 01:56'
updated_date: '2026-08-24 08:12'
labels: []
milestone: m-1
dependencies:
  - TASK-005.01
  - TASK-005.02
  - TASK-005.03
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
parent_task_id: TASK-005
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire corpus -> dataset -> trainer into a runnable pretrain phase with a CLI entry point and a 50M config. This is the integration point that makes pretrain a single command (build-order step 1 of doc-001 section 6).

Files to create:
- src/kestrel/train/pretrain.py  (PretrainConfig + pretrain(config))
- scripts/run_pretrain.py  (CLI, mirrors scripts/check_model.py)
- configs/kestrel/50m/pretrain.yaml
- tests/test_pretrain.py

Design:
- PretrainConfig (Pydantic, strict): model config path (50m/model.yaml), tokenizer path (checkpoints/tokenizer/tokenizer.json), corpus config (corpus.yaml), total_tokens (validation target ~50M), and trainer settings (embed a TrainerConfig). The corpus is split into train/val by the corpus builder (005.01); PretrainConfig derives the train/val dirs from the corpus output_dir (e.g. data/corpus/train, data/corpus/val).
- pretrain(config): load model via kestrel.model.io.load -> corpus.builder.build -> build a TRAIN PretrainDataset (input=<corpus_out>/train) + a VAL PretrainDataset (input=<corpus_out>/val) -> train.trainer(model, train_ds, val_ds, trainer_cfg) -> save final checkpoint.
- scripts/run_pretrain.py: argparse --config, load PretrainConfig, call pretrain().

configs/kestrel/50m/pretrain.yaml fields:
- model: configs/kestrel/50m/model.yaml
- tokenizer: checkpoints/tokenizer/tokenizer.json
- corpus: configs/kestrel/corpus.yaml
- total_tokens: ~50_000_000  (validation target)
- trainer: { lr, weight_decay, batch_size, seq_len: 2048, num_steps, warmup_steps, grad_clip, save_every, log_every, eval_every, eval_iters, output_dir: checkpoints/pretrain/50m/ }

Quantitative targets:
- a TINY end-to-end pretrain (tiny model + tiny local corpus + few steps) runs, train loss decreases, val loss computed, checkpoint written.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pretrain(config) runs end-to-end on a TINY config (tiny model + tiny local corpus + few steps): loss finite and decreasing, checkpoint written to output_dir
- [ ] #2 scripts/run_pretrain.py --config <path> runs the same path via CLI (argparse, mirrors check_model.py)
- [ ] #3 configs/kestrel/50m/pretrain.yaml loads into PretrainConfig (strict) with seq_len 2048 and total_tokens ~50M
- [ ] #4 tests/test_pretrain.py uses a TINY config (tiny model + tiny local corpus + few steps) so it runs fast; make check green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADDED (2026-08-24, from 005.01 split work): the corpus is now split into data/corpus/train + data/corpus/val. pretrain(config) must build BOTH a train dataset (input=data/corpus/train) and a val dataset (input=data/corpus/val) and pass both to the trainer (train for the step loop, val for in-loop val loss). PretrainConfig should derive the train/val dirs from the corpus output_dir (or take them explicitly).

PLAN UPDATE (2026-08-24): total_tokens is a CAP on training tokens, not a fixed target - set it small for the fast TINY test (AC #1/#4) and to the full corpus (~275M) or null (= run until the corpus is exhausted) for the real single-pass validation run (005.06). The run is SINGLE-PASS by design (matches modern LLM pretraining: Chinchilla ~20 tokens/param, LLaMA 'each token used once'; no multi-epoch). TrainerConfig.betas now defaults to (0.9, 0.95) (beta2=0.95, the LLaMA/modern default) - no need to set it in the YAML.
<!-- SECTION:NOTES:END -->
