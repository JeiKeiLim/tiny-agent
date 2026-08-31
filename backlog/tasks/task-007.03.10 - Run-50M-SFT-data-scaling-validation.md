---
id: TASK-007.03.10
title: Run 50M SFT data-scaling validation
status: To Do
assignee: []
created_date: '2026-08-31 01:21'
labels:
  - sft
  - validation
  - experiment
milestone: m-2
dependencies:
  - TASK-007.03.07
  - TASK-007.03.08
  - TASK-007.03.09
parent_task_id: TASK-007.03
priority: high
ordinal: 50000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the M2 50M SFT validation experiment and record the result.

Depends on:
- TASK-007.03.07
- TASK-007.03.08
- TASK-007.03.09

Scope:
- Build or verify the default 50k SFT mixture.
- Run pretrain-only baseline eval on checkpoints/pretrain/50m/final.
- Run SFT smoke/validation runs:
  - 5k max_examples
  - 20k max_examples
  - 50k max_examples
- Use context_length 1024 for first runs unless a config decision says otherwise.
- Record:
  - checkpoint paths
  - dataset manifest paths
  - eval scorecard JSON paths
  - train/validation loss summaries
  - tool/math/assistant metric summaries
  - any failures or debugging notes
- Update Backlog notes or a M2 results doc with the scorecard.
- Update README.md or AGENTS.md only if commands, layout, checkpoint format, or pipeline stages changed.

Success criteria:
- 5k smoke run produces parseable outputs and a complete eval scorecard.
- SFT checkpoints improve over pretrain baseline on at least GSM8K final-answer accuracy and tool-call validity.
- Unseen tool/schema accuracy is non-trivial.
- Pretrain perplexity does not explode.
- 20k/50k show improvement or a clear plateau relative to 5k.

Acceptance:
- All runs are reproducible from configs and manifests.
- make check passes before handing off results.
- Backlog contains the final M2 validation scorecard or a link to it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pretrain-only baseline eval is recorded
- [ ] #2 5k, 20k, and 50k SFT runs produce checkpoints and eval scorecards
- [ ] #3 SFT improves tool-call validity and GSM8K final-answer accuracy over baseline or documents why not
- [ ] #4 Unseen tool/schema eval is included
- [ ] #5 make check passes
<!-- AC:END -->
