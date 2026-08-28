---
id: TASK-005.04.02
title: Add repetition_penalty to generate() and check_model.py
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-28 00:15'
updated_date: '2026-08-28 00:20'
labels:
  - decoding
  - debug
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/model/generate.py
  - scripts/check_model.py
  - tests/test_generate.py
parent_task_id: TASK-005.04
priority: medium
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small 50M pretrain generations often fall into exact repetition loops, especially at temp=0.0 or temp=0.1. Add an optional decoding-side repetition_penalty to generate() and expose it in scripts/check_model.py for manual model testing.

Scope is repetition_penalty only. presence_penalty and frequency_penalty are intentionally deferred to a later serving/decoding task.

Design decision:
- Apply the penalty to previously generated token IDs only, not prompt tokens, so prompt conditioning is not unexpectedly altered.
- Use the common Hugging Face-style repetition penalty rule:
  - for each previously generated token id, if its current logit is positive, divide by repetition_penalty
  - if its current logit is negative, multiply by repetition_penalty
- repetition_penalty=1.0 must be an exact no-op.
- repetition_penalty must be >= 1.0; raise ValueError otherwise.
- Apply the penalty to the final-position logits each generation step before argmax or sampling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generate() accepts repetition_penalty: float = 1.0
- [x] #2 repetition_penalty=1.0 leaves generation behavior unchanged
- [x] #3 repetition_penalty > 1.0 penalizes previously generated token IDs before argmax/sampling
- [x] #4 repetition_penalty works with both temp=0.0 and temp>0
- [x] #5 repetition_penalty < 1.0 raises ValueError
- [x] #6 scripts/check_model.py exposes --repetition-penalty with default 1.0
- [x] #7 tests/test_generate.py covers no-op, repeated-token suppression, temp interaction, and invalid input
- [x] #8 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update src/kestrel/model/generate.py:
   - add repetition_penalty: float = 1.0 parameter
   - validate repetition_penalty >= 1.0
   - on each step, if repetition_penalty != 1.0 and generated is non-empty, copy the final-position logits and apply the penalty to unique generated token IDs
   - use the penalized logits for both temp=0.0 argmax and temp>0 sampling
2. Update scripts/check_model.py:
   - add --repetition-penalty float flag with default 1.0
   - pass it through to generate()
   - update usage/docstring if helpful
3. Add tests in tests/test_generate.py:
   - repetition_penalty=1.0 is a no-op
   - repetition_penalty > 1.0 changes behavior in a scripted repeated-token case
   - penalty works with temp=0.0 and temp>0
   - invalid repetition_penalty < 1.0 raises ValueError
4. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This is a decoding-side debug control only. It does not change training, checkpoints, or model quality.

Manual verification can use:
uv run python scripts/check_model.py --checkpoint checkpoints/pretrain/50m/best --generate --max-tokens 64 --temp 0.1 --repetition-penalty 1.2

Expected result is less exact looping than --repetition-penalty 1.0, but output quality can still be limited by the 50M pretrain model.

Implementation gotcha:
- Keep the default behavior unchanged for existing callers.
- Do not penalize prompt tokens in this task.
- The generated list already exists in generate(); use its unique token IDs.
- MLX logits modification should be simple and testable; short generations make a unique-ID loop acceptable.

Implemented repetition_penalty in generate() with generated-token-only HF-style logit penalty. Added --repetition-penalty to scripts/check_model.py. Added tests for no-op, temp=0 suppression, prompt non-penalty, temp>0 categorical input, and invalid input. make check passed with 165 tests. Manual 50M best checkpoint: temp=0.1, max_tokens=32, penalty=1.0 produced an exact repeated sentence; penalty=1.2 produced non-repeating text.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added optional HF-style repetition_penalty to generate() and --repetition-penalty to scripts/check_model.py. The penalty is generated-token-only, disabled at 1.0, validated to be >= 1.0, and applied before greedy or sampled decoding. Verified with 5 new generate tests, make check (165 passed), and a 50M best-checkpoint generation where penalty 1.2 reduced exact sentence repetition compared with 1.0.
<!-- SECTION:FINAL_SUMMARY:END -->
