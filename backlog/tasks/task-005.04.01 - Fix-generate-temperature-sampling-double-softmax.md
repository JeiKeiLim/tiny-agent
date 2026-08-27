---
id: TASK-005.04.01
title: Fix generate() temperature sampling double softmax
status: Done
assignee: []
created_date: '2026-08-27 09:04'
updated_date: '2026-08-27 09:05'
labels:
  - bug
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/model/generate.py
  - tests/test_generate.py
parent_task_id: TASK-005.04
priority: high
ordinal: 34000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-greedy generation is much more random than configured because generate() applies softmax before calling mx.random.categorical(). MLX categorical expects unnormalized logits and applies softmax internally, so the current code samples from softmax(softmax(logits / temp)). This makes even temp=0.1 collapse into gibberish while temp=0 greedy output can look coherent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generate() passes temperature-scaled logits to mx.random.categorical() for temp > 0
- [x] #2 tests/test_generate.py contains a regression test asserting the categorical input equals captured logits / temp
- [x] #3 make check passes
- [x] #4 Non-greedy generation at low temperature is no longer double-softmaxed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Change src/kestrel/model/generate.py sampling to pass last / temp directly to mx.random.categorical().
2. Add a regression test in tests/test_generate.py that captures the model logits and asserts mx.random.categorical receives logits / temp, not softmax probabilities.
3. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause identified from scripts/check_model.py --generate: temp=0.0 produced English-like text, but temp=0.1 produced almost complete gibberish. MLX docs state mx.random.categorical takes unnormalized log probabilities.

Fixed src/kestrel/model/generate.py to pass last / temp directly to mx.random.categorical(). Added regression test asserting categorical receives captured logits / temp. make check passed with 160 tests. Manual check_model.py --checkpoint checkpoints/pretrain/50m/best --generate --max-tokens 64 --temp 0.1 now produces coherent repetitive text instead of gibberish.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed generate() temperature sampling by passing temperature-scaled logits directly to mx.random.categorical() instead of softmax probabilities. Added a regression test in tests/test_generate.py. Verified with make check (160 tests passed) and a live check_model.py generation at temp=0.1, which now produces coherent repetitive text instead of gibberish.
<!-- SECTION:FINAL_SUMMARY:END -->
