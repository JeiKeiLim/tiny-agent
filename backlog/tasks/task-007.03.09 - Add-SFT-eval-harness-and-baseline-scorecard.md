---
id: TASK-007.03.09
title: Add SFT eval harness and baseline scorecard
status: To Do
assignee: []
created_date: '2026-08-31 01:21'
labels:
  - sft
  - eval
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.04
  - TASK-007.03.08
parent_task_id: TASK-007.03
priority: high
ordinal: 49000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement SFT evaluation and the pretrain-vs-SFT scorecard.

Depends on:
- TASK-007.03.04
- TASK-007.03.08

Files:
- src/kestrel/eval/sft.py
- src/kestrel/eval/tool_calling.py if separate tool metrics are useful
- scripts/run_eval_sft.py
- configs/kestrel/50m/eval_sft.yaml
- tests/eval/test_sft_eval.py

Scope:
- Evaluate a checkpoint on:
  - assistant sanity set from held-out Smol-SmolTalk or equivalent
  - GSM8K test subset
  - local tool seen-schema eval set
  - local tool unseen-schema eval set
  - no-call eval set
  - missing-info eval set
  - small held-out pretrain perplexity set
- Tool metrics:
  - valid JSON rate
  - schema-valid argument rate
  - correct tool selection rate
  - argument exact/partial accuracy
  - no-call correctness
- Math metric:
  - final numeric answer exact match
- Assistant metric:
  - automatic sanity checks first: non-empty, no tool call when tools absent, no obvious repetition
  - optional manual or LLM-judge review later
- Loss is not the primary success metric; record it only as sanity if available.
- Output a JSON scorecard comparing:
  - pretrain-only baseline
  - SFT 5k
  - SFT 20k
  - SFT 50k

Acceptance:
- Eval runs inference-only and does not modify checkpoints.
- Tool eval includes unseen tool names/schemas.
- Scorecard JSON includes per-metric results and checkpoint paths.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eval harness runs on pretrain and SFT checkpoints
- [ ] #2 Tool metrics include valid JSON, schema validity, tool selection, and unseen-schema accuracy
- [ ] #3 Math metric uses final-answer exact match
- [ ] #4 Scorecard compares pretrain-only baseline against SFT checkpoints
- [ ] #5 make check passes
<!-- AC:END -->
