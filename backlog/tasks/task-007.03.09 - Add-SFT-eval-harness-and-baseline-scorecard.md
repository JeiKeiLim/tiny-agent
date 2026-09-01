---
id: TASK-007.03.09
title: Add SFT eval harness and baseline scorecard
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 01:21'
updated_date: '2026-09-01 00:24'
labels:
  - sft
  - eval
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.04
  - TASK-007.03.08
  - TASK-007.03.13
modified_files:
  - src/kestrel/data/sft_chat.py
  - src/kestrel/eval/sft.py
  - src/kestrel/eval/tool_calling.py
  - scripts/run_eval_sft.py
  - configs/kestrel/50m/eval_sft.yaml
  - tests/data/test_sft_chat.py
  - tests/data/test_sft_dataset.py
  - tests/data/test_sft_prepare_tool.py
  - tests/data/test_sft_prepare_eval.py
  - tests/data/test_sft_public_tool.py
  - tests/eval/test_sft_eval.py
  - README.md
  - AGENTS.md
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
- [x] #1 Eval harness runs on pretrain and SFT checkpoints
- [x] #2 Tool metrics include valid JSON, schema validity, tool selection, and unseen-schema accuracy
- [x] #3 Math metric uses final-answer exact match
- [x] #4 Scorecard compares pretrain-only baseline against SFT checkpoints
- [x] #5 make check passes
- [x] #6 render_sft() includes compact tool definitions in a loss-masked system block when row.tools is non-empty
- [x] #7 SFT training data and SFT eval prompts use the same tool-aware renderer
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend src/kestrel/data/sft_chat.py render_sft() to prepend a compact loss-masked system block containing row.tools when tools are present. 2. Update SFT chat/dataset tests for tool-aware rendering. 3. Add src/kestrel/eval/tool_calling.py for generated tool-call parsing and tool metrics. 4. Add src/kestrel/eval/sft.py with strict eval config, held-out row loading, prompt construction, greedy generation, assistant/math/tool metrics, optional pretrain perplexity, and JSON scorecard output. 5. Add scripts/run_eval_sft.py CLI with --config, --max-rows, --output, and --skip-perplexity. 6. Add configs/kestrel/50m/eval_sft.yaml for pretrain plus expected 5k/20k/50k SFT checkpoints. 7. Add tests/eval/test_sft_eval.py covering tool-aware rendering, inference-only eval, tool/math metrics, scorecard comparison, and CLI behavior. 8. Update README.md and AGENTS.md, then run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SFT scorecard eval bundle is prepared by TASK-007.03.13 under data/sft/eval. The eval harness should consume that held-out bundle instead of the training mixture.

Scope expanded with user approval: the existing SFT renderer ignored row.tools, so tool schemas were absent from prompts. This task now includes a minimal tool-aware renderer fix before building the SFT eval harness.

Implemented tool-aware render_sft(), src/kestrel/eval/sft.py, src/kestrel/eval/tool_calling.py, scripts/run_eval_sft.py, configs/kestrel/50m/eval_sft.yaml, tests/eval/test_sft_eval.py, and README/AGENTS updates. make check passes with 326 tests. A real one-row pretrain smoke with max_tokens=16 produced a valid scorecard; tool/math scores are zero for the pretrain checkpoint, which is expected before SFT. Verified current raw and eval tool rows remain within the 1024-token SFT context after the renderer change (max observed: tool_public 983, tool_local 872, tool_eval_seen 848).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the inference-only SFT eval harness and pretrain-vs-SFT scorecard. It evaluates held-out assistant, GSM8K, local tool seen/unseen/no-call/missing-info rows, and optional pretrain perplexity, then writes a JSON scorecard with per-checkpoint metrics. Also fixed the SFT chat renderer to expose row.tools as a compact loss-masked system block so tool schemas are present in training and eval prompts. Added run_eval_sft.py, configs/kestrel/50m/eval_sft.yaml, eval tests, and README/AGENTS updates. Verified with make check: 326 tests passed.
<!-- SECTION:FINAL_SUMMARY:END -->
