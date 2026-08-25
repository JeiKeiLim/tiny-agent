---
id: TASK-005.07
title: Fix causal attention + query-chunked SDPA for seq_len=2048
status: Done
assignee: []
created_date: '2026-08-24 23:54'
updated_date: '2026-08-24 23:58'
labels:
  - model
  - performance
  - bug
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/model/kestrel.py
  - tests/test_model_kestrel.py
parent_task_id: TASK-005
priority: high
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Kestrel attention causal and trainable at seq_len=2048.

Bug: Attention.__call__ in src/kestrel/model/kestrel.py calls mx.fast.scaled_dot_product_attention(q,k,v,scale=scale) with no mask. MLX SDPA mask default is None (not causal), so the decoder can attend to future tokens. This invalidates pretraining and generation.

Performance: full T=2048 SDPA in training mode falls back to an O(T^2) materialization and hangs/OOMs on M4 Pro 48GB. A query-chunked SDPA path is feasible: split q into chunks, call SDPA per chunk against full k/v, concatenate. Earlier chunks use a boolean causal mask (key_pos <= query_pos); the final chunk can use mask='causal' because MLX causal mask is lower-right aligned.

Spike results (15-layer 50M shape, B=8, float32): T=2048 chunk=1024: 11.44s/step, peak 33.34GB, ~1.4k tok/s; T=1024: 1.28s/step, peak 12.79GB, ~6.4k tok/s. Correctness vs full causal SDPA: max_abs_diff=0.

Implementation:
- add causal_sdpa(q,k,v,*,scale,chunk_size=1024) to src/kestrel/model/kestrel.py
- Attention.__call__ must use causal_sdpa instead of raw SDPA
- keep GQA native (do not tile k/v)
- no config change required for M1

Gotchas:
- mask='causal' is lower-right aligned; only valid for a query chunk when that chunk ends at T.
- boolean mask True means attend; verified against mask='causal'.
- do not pre-tile k/v for GQA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Attention is causal: changing tokens after position i does not change logits at positions <= i (unit test)
- [x] #2 causal_sdpa with chunk_size < T matches mx.fast.scaled_dot_product_attention(mask='causal') within 1e-5 (unit test with small shapes)
- [x] #3 make check is green
- [x] #4 TASK-005.06 notes record the T=1024 and T=2048 step-time/peak-memory benchmarks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add causal_sdpa helper to src/kestrel/model/kestrel.py.
2. Switch Attention.__call__ to causal_sdpa.
3. Add tests to tests/test_model_kestrel.py.
4. Run make check.
5. Append benchmark notes to TASK-005.06.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Validation: make check green (78 tests passed). Added causal_sdpa helper and causal unit tests; switched Attention to causal query-chunked SDPA.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed missing causal masking and made seq_len=2048 trainable via query-chunked causal SDPA. Verified with unit tests and make check; recorded T=1024/T=2048 benchmarks in TASK-005.06.
<!-- SECTION:FINAL_SUMMARY:END -->
