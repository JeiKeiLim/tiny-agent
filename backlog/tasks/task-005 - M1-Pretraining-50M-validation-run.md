---
id: TASK-005
title: M1 - Pretraining (50M validation run)
status: Done
assignee: []
created_date: '2026-08-24 01:54'
updated_date: '2026-08-30 23:14'
labels: []
milestone: m-1
dependencies: []
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the Track A pretraining stack and validate it with a short (~50M-token) run on Kestrel-50M, confirming the forward-pass loss decreases and generate() yields coherent text. This is build-order step 1 of plan doc-001 §6.

Scope (subtasks):
- 005.01 corpus/ — pluggable weighted corpus builder (raw text)
- 005.02 data/pretrain_dataset.py — tokenize text -> (input, target) batches
- 005.03 train/trainer.py — shared trainer (optimizer, step loop, checkpoint)
- 005.04 model/generate.py — minimal autoregressive generate() for the coherent-text check
- 005.05 train/pretrain.py + scripts/run_pretrain.py + 50m/pretrain.yaml — pretrain loop + entry point
- 005.06 validation run — ~50M tokens on 50M, verify loss down + coherent text

Decisions locked (2026-08-24): 50M-first (150M is a later manual pass); reuse the existing 1GB tokenizer sample (data/tokenizer_train/) via a 'local' corpus source (no new download); build the full pluggable corpus/ now; build a minimal generate() now; 150M stays 32Lx640; '1B' = 1B tokens of data (not a model size). Full ~1B-token run is a separate follow-up milestone.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-28: All M1 subtasks are Done. The 50M pretrain validation run completed 1.0135B tokens, final checkpoint is at checkpoints/pretrain/50m/final, external validation loss is 3.1443 on 302,808 held-out tokens, and make check is green. M1 is ready to be marked complete.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 01:36
---
2026-08-26: M1 smoke pipeline is complete enough to move to a serious data budget. The expanded ~12GiB / ~2.97B-token corpus work is tracked under TASK-005.10.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M1 pretraining is complete. The pretrain stack, document-aware corpus/dataset, trainer resume/retention, generation, evaluation, and 50M 1B-token validation run are all done. The final 50M checkpoint is adequate as a pipeline-validation and SFT-base artifact.
<!-- SECTION:FINAL_SUMMARY:END -->
