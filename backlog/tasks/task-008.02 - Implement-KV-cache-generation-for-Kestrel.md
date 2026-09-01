---
id: TASK-008.02
title: Implement KV-cache generation for Kestrel
status: Done
assignee: []
created_date: '2026-09-01 00:42'
updated_date: '2026-09-01 01:10'
labels:
  - model
  - inference
  - performance
dependencies: []
modified_files:
  - src/kestrel/model/cache.py
  - src/kestrel/model/kestrel.py
  - src/kestrel/model/generate.py
  - tests/test_model_cache.py
  - tests/test_model_kestrel.py
  - tests/test_generate.py
  - README.md
  - AGENTS.md
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
- [x] #1 Cached greedy generation produces the same output as the existing no-cache greedy generate() for a small model and fixed prompt
- [x] #2 Prompt is processed once in a prefill step, then each new token runs a single-token decode step
- [x] #3 KV cache memory scales with prompt_len + max_tokens, not with the sum of every intermediate sequence length
- [x] #4 512-token 50M generation completes with mx.get_cache_memory() under 1GiB without relying on periodic mx.clear_cache()
- [x] #5 Cached generation is at least 3x faster than the no-cache path for 512 tokens from a short prompt on the 50M checkpoint
- [x] #6 Sampling, stop token, and repetition_penalty behavior remain supported
- [x] #7 make check passes
- [x] #8 Kestrel exposes inference-only prefill/decode methods while the training __call__ path and checkpoint format remain unchanged
- [x] #9 generate() uses the cached path for Kestrel models and falls back to the existing no-cache path for plain callable models
- [x] #10 KV cache is preallocated or grown in chunks, not concatenated one token at a time
<!-- AC:END -->



## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/model/cache.py with a small KVCache type storing k/v and current position, with preallocation or chunk growth. 2. Extend the Kestrel forward path with optional per-layer caches; model(x) with no cache must remain the training path. 3. Add Kestrel.prefill(prompt, reserve=max_tokens) and Kestrel.decode(token, caches) inference methods. 4. In Attention, keep cache=None behavior unchanged; when cache is provided, compute q/k/v for current tokens, apply RoPE at the current position, write/append k/v, and attend current q against cached k/v. 5. Update generate() to use prefill/decode when available and fall back to the legacy no-cache loop for scripted/plain callable models. 6. Add tests for prefill logits equivalence, decode logits equivalence, cached vs legacy greedy equality, stop/sampling/repetition behavior, and cache growth. 7. Update README/AGENTS and run make check. 8. Run a real 50M 512-token benchmark to verify memory and at least 3x speedup.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design decision: preallocate or grow KV cache in chunks, not one token at a time, to avoid the same allocator size-class churn. For single-token decode, query length is 1 and all cached keys are prior to the query, so an explicit causal mask should not be needed. Verify mx.fast.scaled_dot_product_attention works with q shape (B, H, 1, D) and k/v shape (B, n_kv_heads, T, D). Do not break doc_ids training forward.

Architecture decision: avoid a separate CachedKestrel wrapper that duplicates model logic. Add optional cache plumbing inside Kestrel/Attention, with cache=None preserving the existing training path. The public inference API should be logits, caches = model.prefill(prompt_ids, reserve=max_tokens) followed by logits, caches = model.decode(next_token_ids, caches). generate() can detect prefill/decode and use the cached path, while plain callable test models use the legacy path. No cross-prompt prefix caching is required; each eval row gets a fresh cache and discards it after generation. The win is avoiding reprocessing the prompt and previously generated tokens at every decode step within the same row. 50M KV cache is roughly 15KB per token, so a 2048-token cache is only about 30MB. Initial cache path can support batch_size=1 and doc_ids=None only.

Implemented KVCache in src/kestrel/model/cache.py with preallocation and chunk growth. Added optional cache plumbing to Attention/TransformerBlock/Kestrel, with Kestrel.prefill(reserve=...) and Kestrel.decode(). generate() now dispatches to the cached path when a model exposes prefill/decode and falls back to the no-cache path for plain callables. Training __call__, doc_ids, checkpoint format, and SFT eval config remain unchanged. make check passes with 347 tests. Real 512-token 50M pretrain benchmark: cached output matched no-cache output; no-cache 6.337s vs cached 1.491s (4.25x); no-cache MLX cache 18.93GiB vs cached 0.019GiB; cached peak 0.207GiB with clear_cache_every=0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added KV-cache generation for Kestrel. Kestrel now supports inference-only prefill and single-token decode with per-layer KV caches, and generate() uses that path automatically for Kestrel models while preserving a no-cache fallback for plain callable models. The training forward path, doc_ids path, and checkpoint format are unchanged. Verified with unit tests, cached/no-cache equivalence, and a real 50M 512-token benchmark showing 4.25x speedup and MLX cache under 0.02GiB without periodic clear_cache.
<!-- SECTION:FINAL_SUMMARY:END -->
