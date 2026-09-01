---
id: TASK-008.02
title: Implement KV-cache generation for Kestrel
status: To Do
assignee: []
created_date: '2026-09-01 00:42'
labels:
  - model
  - inference
  - performance
dependencies: []
parent_task_id: TASK-008
priority: high
ordinal: 56000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Root-cause fix for TASK-008. Replace or augment the no-KV-cache generate() path so Kestrel inference performs one prompt prefill and then single-token decode steps using cached key/value tensors. Current src/kestrel/model/generate.py re-runs the full sequence every token, causing O(T^2) compute and MLX allocator-cache growth. The model is src/kestrel/model/kestrel.py; Attention uses grouped-query attention with n_heads=8 and n_kv_heads=2 for 50M, RoPE, and mx.fast.scaled_dot_product_attention. Generation does not use doc_ids, so the cache path can initially support only the ordinary causal generation path. Likely files: src/kestrel/model/kestrel.py, src/kestrel/model/generate.py, optionally src/kestrel/model/cache.py, tests/test_generate.py, tests/test_model_kestrel.py, scripts/check_model.py, src/kestrel/eval/sft.py. The training forward path, checkpoint format, and document-aware pretrain evaluation must remain unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cached greedy generation produces the same output as the existing no-cache greedy generate() for a small model and fixed prompt
- [ ] #2 Prompt is processed once in a prefill step, then each new token runs a single-token decode step
- [ ] #3 KV cache memory scales with prompt_len + max_tokens, not with the sum of every intermediate sequence length
- [ ] #4 512-token 50M generation completes with mx.get_cache_memory() under 1GiB without relying on periodic mx.clear_cache()
- [ ] #5 Cached generation is at least 3x faster than the no-cache path for 512 tokens from a short prompt on the 50M checkpoint
- [ ] #6 Sampling, stop token, and repetition_penalty behavior remain supported
- [ ] #7 make check passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design decision: preallocate or grow KV cache in chunks, not one token at a time, to avoid the same allocator size-class churn. For single-token decode, query length is 1 and all cached keys are prior to the query, so an explicit causal mask should not be needed. Verify mx.fast.scaled_dot_product_attention works with q shape (B, H, 1, D) and k/v shape (B, n_kv_heads, T, D). Do not break doc_ids training forward.
<!-- SECTION:NOTES:END -->
