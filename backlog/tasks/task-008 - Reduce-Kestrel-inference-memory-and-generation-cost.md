---
id: TASK-008
title: Reduce Kestrel inference memory and generation cost
status: Done
assignee: []
created_date: '2026-09-01 00:42'
updated_date: '2026-09-01 01:10'
labels:
  - model
  - inference
  - memory
  - performance
dependencies: []
priority: high
ordinal: 54000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Kestrel inference currently uses a no-KV-cache generate() path that re-runs the full sequence every token. This causes O(T^2) compute and large MLX allocator-cache growth. Measured on the 50M pretrain checkpoint: 512-token greedy generation used ~0.19GiB active MLX memory and ~0.53GiB peak active memory, but retained ~18.26GiB in mx.get_cache_memory(); mx.clear_cache() released the cache. This makes scripts/check_model.py --generate and scripts/run_eval_sft.py appear to use 20-44GiB system memory. This parent tracks a short-term allocator-cache mitigation and the root-cause KV-cache generation fix.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both subtasks are complete. TASK-008.01 bounds temporary MLX allocator cache with clear_cache_every. TASK-008.02 removes the root O(T^2) no-cache generation behavior for Kestrel models via KV-cache prefill/decode. Real 50M 512-token benchmark: cached generation matched no-cache output, ran 4.25x faster, and used 0.019GiB MLX cache vs 18.93GiB for the no-cache path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced Kestrel inference memory and generation cost. generate() now uses KV-cache prefill/decode for Kestrel models, falls back to no-cache generation for plain callable models, and can optionally release temporary MLX allocator cache with clear_cache_every. The 50M 512-token benchmark shows 4.25x faster generation and MLX cache reduced from 18.93GiB to 0.019GiB.
<!-- SECTION:FINAL_SUMMARY:END -->
