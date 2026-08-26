---
id: TASK-005.10.04
title: Add 12GiB pretrain configs and short validation runs
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-26 01:36'
updated_date: '2026-08-26 04:24'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies:
  - TASK-005.10.02
  - TASK-005.10.03
modified_files:
  - configs/kestrel/50m/pretrain.yaml
  - configs/kestrel/150m/pretrain.yaml
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
- [x] #1 150M estimated_steps matches the 12g train manifest within 5%
- [x] #2 50M estimated_steps matches the 1013504000 token cap within 5%
- [x] #3 A short smoke pretrain run completes with finite loss and writes a checkpoint
- [x] #4 make check is green
- [x] #5 Existing configs/kestrel/50m/pretrain.yaml loads with the 12GiB corpus and 1013504000 token cap
- [x] #6 Existing configs/kestrel/150m/pretrain.yaml loads with the 12GiB corpus and no token cap
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update existing configs/kestrel/50m/pretrain.yaml in place to use configs/kestrel/corpus-12g.yaml, total_tokens 1013504000, batch_size 8, seq_len 1024, save_every 20000. 2. Update existing configs/kestrel/150m/pretrain.yaml in place to use configs/kestrel/corpus-12g.yaml, total_tokens null, batch_size 4, seq_len 1024, save_every 50000. 3. Remove the temporary pretrain-12g.yaml files. 4. Update tests to load the in-place 50M/150M pretrain configs and keep skipif estimated-step tests for data/corpus-12g. 5. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated existing 50M and 150M pretrain configs in place instead of adding pretrain-12g.yaml files. 50M uses corpus-12g with total_tokens=1013504000 and save_every=20000. 150M uses corpus-12g with total_tokens=null and save_every=50000. Added YAML load tests and skipif estimated-step tests. make check passed with 119 tests.

Reverted save_every to 2000 in both 50M and 150M pretrain configs per user override; user will handle checkpoint disk usage separately. make check passed with 119 tests.

Updated 50M/150M pretrain configs to reference configs/kestrel/corpus.yaml after removing corpus-12g.yaml. make check passed with 119 tests.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 04:09
---
2026-08-26: User confirmed data/corpus-12g is the only corpus going forward, so the existing 50M/150M pretrain configs should be updated in place instead of adding separate pretrain-12g.yaml files.
---

created: 2026-08-26 04:10
---
2026-08-26: User overrode disk-saving checkpoint spacing: restore save_every=2000 in both 50M and 150M pretrain configs.
---

created: 2026-08-26 04:23
---
Updating 50M/150M pretrain configs to reference configs/kestrel/corpus.yaml after corpus config consolidation.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the existing 50M/150M pretrain configs in place to use data/corpus-12g, with a Chinchilla token cap for 50M and a full single-pass 150M config. save_every remains 2000 per user override. Verified config loading, estimated steps, tiny smoke pretrain, and make check (119 tests passed).
<!-- SECTION:FINAL_SUMMARY:END -->
