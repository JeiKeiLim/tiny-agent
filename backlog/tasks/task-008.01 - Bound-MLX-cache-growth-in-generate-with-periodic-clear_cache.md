---
id: TASK-008.01
title: Bound MLX cache growth in generate() with periodic clear_cache
status: Done
assignee:
  - '@opencode'
created_date: '2026-09-01 00:42'
updated_date: '2026-09-01 00:45'
labels:
  - model
  - inference
  - memory
  - bug
dependencies: []
modified_files:
  - src/kestrel/model/generate.py
  - scripts/check_model.py
  - src/kestrel/eval/sft.py
  - tests/test_generate.py
  - README.md
  - AGENTS.md
parent_task_id: TASK-008
priority: high
ordinal: 55000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Short-term mitigation for TASK-008. The current generate() in src/kestrel/model/generate.py has no KV cache and re-runs the full sequence each token. Each step allocates/frees differently sized buffers; MLX retains freed buffers in its allocator cache. Measured 512-token generation on checkpoints/pretrain/50m/final: active ~0.19GiB, peak active ~0.53GiB, cache ~18.26GiB. Calling mx.clear_cache() every 64 generated tokens kept cache near 0 and peak active ~0.52GiB. This task adds a configurable periodic mx.clear_cache() call to bound memory. It is explicitly a bandaid, not the root KV-cache fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generate() accepts clear_cache_every: int = 64 and calls mx.clear_cache() after every N generated tokens when N > 0
- [x] #2 scripts/check_model.py exposes --clear-cache-every and passes it to generate()
- [x] #3 SFTEvalConfig.generation includes clear_cache_every and scripts/run_eval_sft.py passes it through evaluate_sft()
- [x] #4 Tests verify clear_cache is called on the configured cadence and disabled when clear_cache_every == 0
- [x] #5 make check passes
- [x] #6 Manual 512-token 50M probe reports mx.get_cache_memory() under 1GiB with clear_cache_every=64
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add clear_cache_every: int = 64 to generate() in src/kestrel/model/generate.py and call mx.clear_cache() after every N appended generated tokens when N > 0. 2. Add --clear-cache-every CLI flag to scripts/check_model.py and pass it through. 3. Add clear_cache_every to SFTGenerationConfig in src/kestrel/eval/sft.py and pass it from _generate_output. 4. Update tests/test_generate.py and tests/eval/test_sft_eval.py to verify cadence/disable behavior. 5. Run make check. 6. Run a real 512-token 50M probe and record mx.get_cache_memory() result.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Use mx.clear_cache(), not deprecated mx.metal.clear_cache(). Do not call every token; default 64 was validated. This does not fix O(T^2) generation compute.

Implemented clear_cache_every in generate(), check_model.py, and SFTGenerationConfig. Added cadence/disable/default tests. make check passes with 330 tests. Real 512-token 50M probe with clear_cache_every=64: active 0.189GiB, cache 0.031GiB, peak 0.520GiB. check_model.py 64-token smoke with --clear-cache-every 16 succeeded.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a configurable periodic mx.clear_cache() call to the no-KV-cache generate() path to bound MLX allocator-cache growth. generate() defaults to clear_cache_every=64, scripts/check_model.py exposes --clear-cache-every, and SFT eval generation passes the same setting from SFTGenerationConfig. Added tests for cadence, disabled mode, default, and invalid input. Verified with make check: 330 tests passed. A real 512-token 50M probe went from ~18.26GiB retained cache without clearing to ~0.031GiB retained cache with clear_cache_every=64.
<!-- SECTION:FINAL_SUMMARY:END -->
