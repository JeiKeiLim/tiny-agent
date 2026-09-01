---
id: TASK-007.03.14
title: Fix SFT inference prompt and special-token decoding
status: Done
assignee:
  - '@Jongkuk Lim'
created_date: '2026-09-01 04:45'
updated_date: '2026-09-01 22:12'
labels:
  - bug
  - sft
  - eval
  - inference
milestone: m-2
dependencies: []
references:
  - data/sft/eval/scorecard.json
  - src/kestrel/model/generate.py
  - src/kestrel/eval/sft.py
  - src/kestrel/data/chat.py
  - src/kestrel/data/sft_chat.py
  - backlog/tasks/task-007.03.10 - Run-50M-SFT-data-scaling-validation.md
modified_files:
  - src/kestrel/model/generate.py
  - src/kestrel/data/sft_chat.py
  - src/kestrel/data/chat.py
  - src/kestrel/eval/sft.py
  - src/kestrel/eval/pretrain.py
  - scripts/check_model.py
  - tests/test_generate.py
  - tests/data/test_sft_chat.py
  - tests/data/test_chat.py
  - tests/eval/test_sft_eval.py
  - README.md
  - AGENTS.md
  - >-
    backlog/docs/research/sft-drift-pretrain-perplexity/doc-008 -
    SFT-Drift-Pretrain-Perplexity-and-Tool-Call-Failure-Research.md
parent_task_id: TASK-007.03
priority: high
ordinal: 57000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix two SFT inference/eval bugs that make the current SFT scorecard invalid.

## Context

The 50M 50k SFT run completed at `checkpoints/sft/50m/final` (step 6246). The first scorecard at `data/sft/eval/scorecard.json` showed tool seen/unseen `valid_json_rate = 0.0` for both pretrain and SFT. Manual probing shows the SFT checkpoint can emit the `tool_call` special token when the prompt includes the assistant role prefix, but the current eval/inference path does not measure that correctly.

This task is a bug fix and re-measurement task. It is not a retraining task.

## Root causes

1. `generate()` decodes generated token IDs with `tokenizer.decode(generated)`, which defaults to `skip_special_tokens=True`.
   - Affected code: `src/kestrel/model/generate.py:78` and `src/kestrel/model/generate.py:105`.
   - This removes `tool_call`, `tool_call_end`, `im_start`, `im_assistant`, and other special tokens from the returned text.
   - The tool-call parser requires the literal `tool_call` marker in the returned text: `src/kestrel/eval/tool_calling.py:27`.
   - Therefore tool metrics can be forced to zero even when the model emits the correct special token.

2. Eval and chat prompts do not append the assistant completion prefix.
   - Affected code: `src/kestrel/eval/sft.py:299` in `_prompt_text()` and `src/kestrel/data/chat.py:14` in `build_chat_prompt()`.
   - Training loss-masks the assistant role marker: `src/kestrel/data/sft_chat.py:133`.
   - The model is therefore trained to generate assistant content/tool-call payload given a prompt that already ends with:
     `im_start` + newline + `im_assistant` + newline.
   - Current eval prompts end after the previous user/tool message and do not provide that assistant prefix.

## Goal

Make inference and eval use the same prompt contract as training, and preserve special tokens in generated text by default so tool-call parsing can see `tool_call` / `tool_call_end`.

## Scope

- Update `generate()` to preserve special tokens by default and expose an explicit `skip_special_tokens` option.
- Apply the same decode behavior to both the KV-cache path and the no-cache fallback path.
- Add or use a prompt helper that appends the assistant completion prefix for inference/eval.
- Use that helper in SFT eval prompt construction and interactive chat prompt construction.
- Add regression tests.
- Re-run the SFT scorecard on the existing pretrain and SFT checkpoints.
- Record before/after results in task notes.
- Update README/AGENTS if user-visible inference behavior or commands changed.

## Non-goals

- Do not retrain the model.
- Do not change the SFT training loss mask.
- Do not change the tool-call format.
- Do not attempt to fix model-quality issues such as repetition, weak math, or imperfect JSON termination. If those remain after fair measurement, track them as follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generate() preserves tokenizer special tokens in returned text by default
- [x] #2 generate() supports skip_special_tokens=True for callers that want clean text
- [x] #3 KV-cache and no-cache generation paths use the same special-token decode behavior
- [x] #4 SFT eval prompts end with im_start + newline + im_assistant + newline after the last prior message
- [x] #5 interactive chat prompts end with im_start + newline + im_assistant + newline
- [x] #6 tests cover special-token preservation, prompt suffix behavior, and tool-call parsing with generated special tokens
- [x] #7 a manual temp=0 probe over 5 tool_seen and 5 tool_unseen cases records at least one raw tool_call marker in each category
- [x] #8 a corrected scorecard is generated for checkpoints/pretrain/50m/final and checkpoints/sft/50m/final
- [x] #9 task notes record before/after scorecard deltas and any remaining model-quality issues
- [x] #10 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression tests for special-token decoding and assistant-prefix prompts.
2. Update `src/kestrel/model/generate.py` to decode with `skip_special_tokens=False` by default and add an explicit `skip_special_tokens` parameter.
3. Add a completion-prompt helper in `src/kestrel/data/sft_chat.py` or `src/kestrel/data/chat.py` that appends `im_start\nim_assistant\n` to a rendered prior-message prompt.
4. Use the helper in `src/kestrel/eval/sft.py:_prompt_text()` and `src/kestrel/data/chat.py:build_chat_prompt()`.
5. Run `make check`.
6. Re-run `uv run python scripts/run_eval_sft.py --config configs/kestrel/50m/eval_sft.yaml`.
7. Run a small manual probe over direct tool eval cases and record raw special-token behavior.
8. Update README/AGENTS if needed and record before/after scorecard deltas in task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Buggy scorecard values from `data/sft/eval/scorecard.json`:

- pretrain:
  - assistant non_empty 0.645, no_tool_call 1.0, no_repetition 0.535
  - math exact 0.008
  - tool seen/unseen valid_json 0.0, schema_valid 0.0, selection 0.0, argument_exact 0.0
  - no_call correct 0.204
  - missing_info correct 0.0
  - perplexity 24.14

- sft_50k:
  - assistant non_empty 0.825, no_tool_call 1.0, no_repetition 0.555
  - math exact 0.020
  - tool seen/unseen valid_json 0.0, schema_valid 0.0, selection 0.0, argument_exact 0.0
  - no_call correct 0.240
  - missing_info correct 0.172
  - perplexity 50.08

Training run facts:

- checkpoint: `checkpoints/sft/50m/final`
- final step: 6246
- final train loss: 1.6147
- first 50-step avg train loss: 2.3905
- last 50-step avg train loss: 1.8438
- best in-loop val: 1.8335 at step 6000
- in-loop val was not held-out because the SFT config had no `val_dataset`

Manual probe facts:

- tokenizer special token IDs observed:
  - `im_start` = 0
  - `im_assistant` = 4
  - `tool_call` = 5
  - `tool_call_end` = 6
- Without assistant prefix, direct tool prompts produced weak outputs and no useful tool-call structure.
- With `im_start\nim_assistant\n` appended, direct tool probes began with token ID 5 (`tool_call`) in the first 2 tool_seen and first 2 tool_unseen cases.
- Some outputs still failed JSON termination and repeated `}`; one unseen case emitted `tool_call_end` followed by a newline.
- This indicates the current zero tool metrics are at least partly an eval/inference measurement bug, not necessarily pure model failure.

Corrected scorecard after fix:

- pretrain:
  - assistant non_empty 0.630, no_repetition 0.510
  - math exact 0.002
  - tool seen/unseen valid_json 0.0
  - no_call correct 0.140
  - missing_info correct 0.0
  - pretrain-val perplexity 24.141

- sft_50k:
  - assistant non_empty 1.000, no_repetition 0.520
  - math exact 0.012
  - tool seen/unseen valid_json 0.0
  - no_call correct 1.000
  - missing_info correct 1.000
  - pretrain-val perplexity 50.084

Manual 20-case tool probe:

- 20/20 prompts ended with assistant prefix
- 20/20 raw outputs contained `tool_call`
- 0/20 valid JSON
- dominant failure: missing opening brace / missing newline after `tool_call` / repeated closing braces
- repetition penalty 1.1-1.3 improved `tool_call_end` emission but did not produce valid JSON

Research findings saved to `doc-008 - SFT Drift, Pretrain Perplexity, and Tool-Call Failure Research`.

`make check` passes with 368 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the SFT eval/chat prompt contract and special-token decoding. generate() now preserves tokenizer special tokens by default and supports skip_special_tokens=True. SFT eval and chat prompts append the assistant completion prefix. Added regression tests, updated display-only callers, updated README/AGENTS, generated the corrected scorecard, and saved research analysis as doc-008. Verified with make check: 368 tests passed.
<!-- SECTION:FINAL_SUMMARY:END -->
