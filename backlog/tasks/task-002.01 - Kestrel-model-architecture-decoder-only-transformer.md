---
id: TASK-002.01
title: Kestrel model architecture (decoder-only transformer)
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-24 00:15'
updated_date: '2026-08-24 00:55'
labels: []
milestone: m-0
dependencies: []
parent_task_id: TASK-002
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Kestrel decoder-only transformer in src/kestrel/model/kestrel.py per plan §9: pre-norm RMSNorm, RoPE, SwiGLU FFN, GQA attention, tied embeddings, no biases, dropout 0. __call__ returns logits of shape (B, T, vocab). Use mx.fast.scaled_dot_product_attention (fused) so GQA (n_kv_heads < n_heads) is native and the later 8-16k context extension stays memory-feasible. Tied embeddings: use the embedding matrix as the unembedding (logits = h @ W_embed.T) — no separate lm_head weight. Include a count_params(model) helper. ModelConfig + both model.yaml already exist (TASK-001).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Instantiate Kestrel-50M and Kestrel-150M from their configs; count_params lands near ~50M / ~150M (asserted, not hand-waved)
- [x] #2 Forward pass on random token IDs returns logits of shape (B, T, vocab) with finite cross-entropy loss
- [x] #3 GQA exercised (n_kv_heads=2 < n_heads) via fused attention; no biases anywhere; dropout 0
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
New file src/kestrel/model/kestrel.py. Config already exists: ModelConfig in src/kestrel/model/config.py, loaded from configs/kestrel/50m/model.yaml and configs/kestrel/150m/model.yaml via load_config (kestrel.common.config). Config fields: name, vocab_size=16384, context_length=2048, n_layers, n_heads, n_kv_heads, hidden_size, intermediate_size, rope_theta=10000.0, tie_embeddings=True, dropout=0.0. head_dim = hidden_size // n_heads (64 for both).

Components (all nn.Module, bias=False):
- RMSNorm(hidden): weight (hidden,); y = x * rsqrt(mean(x^2)+eps) * weight.
- RoPE: standard rotary, positions 0..context_length-1, base=rope_theta.
- Attention: q_proj (hidden -> n_heads*head_dim), k_proj and v_proj (hidden -> n_kv_heads*head_dim), o_proj (n_heads*head_dim -> hidden). Reshape q to (B, n_heads, T, head_dim), k/v to (B, n_kv_heads, T, head_dim); call mx.fast.scaled_dot_product_attention(q, k, v) which natively handles GQA (n_kv < n_q).
- FeedForward (SwiGLU): gate_proj and up_proj (hidden -> intermediate), down_proj (intermediate -> hidden); out = down(silu(gate(x)) * up(x)).
- TransformerBlock: attn_norm (RMSNorm) + Attention + ffn_norm (RMSNorm) + FeedForward, pre-norm with residuals.
- Kestrel(nn.Module): embed = nn.Embedding(vocab_size, hidden_size); n_layers TransformerBlocks; final RMSNorm. Forward: h = embed(x) -> blocks -> final_norm -> logits = h @ embed.weight.T (tied, no separate lm_head). Returns (B, T, vocab).
- count_params(model) -> int: sum(p.size for p in model.parameters()).

Init: linears and embedding normal(0, 0.02).

Tests: tests/test_model_kestrel.py. Load both real configs with load_config; assert count_params within ~5% of 50M / 150M (expected ~50.7M / ~148M). Forward on random int32 ids (B=1, T=small) -> logits shape (B, T, vocab) and finite cross-entropy loss (shifted targets). Optionally a tiny config for a fast check.

mypy is strict and MLX is untyped: add targeted type: ignore comments where needed (as in tokenizer/train.py). Gate: make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decisions/gotchas:
- GQA is native to mx.fast.scaled_dot_product_attention (k/v have fewer heads than q) - do NOT manually repeat/expand KV heads.
- Tied embeddings via logits = h @ embed.weight.T (no lm_head weight) -> clean param count and clean save/load (weight stored once).
- Expected param counts (hand-computed, match plan doc-001 §9): 50M ~50.7M, 150M ~148M. Assert within tolerance, not exact.
- The model returns logits only; cross-entropy loss is computed by the caller (test now, trainer later), not inside the model.
- head_dim is derived (hidden_size // n_heads), not a config field.

RESULT: make check green. count_params: 50m=50,675,200 (50.7M), 150m=148,152,960 (148.2M) - both within 5% and matching plan §9. Tiny untrained CE loss 4.81 (~ln 64=4.16), logits (B,T,vocab).

MLX 0.32 gotchas discovered (apply to 002.02 io + 002.03 check script too):
- mlx.core HAS type stubs (typed); mlx.nn has NO stubs. Added mypy override in pyproject.toml: [[tool.mypy.overrides]] module = "mlx.nn" / follow_imports = "skip" (else spurious attr-defined/name-defined for nn.Module, nn.Linear, ...). Kept # type: ignore[misc] on each nn.Module subclass (disallow_subclassing_any) and [no-any-return] where a nn.* layer result is returned.
- No mx.polar / mx.view_as_complex / mx.view_as_real in the core stubs -> RoPE uses the real cos/sin (rotate-half) formulation, not complex.
- mx.fast.scaled_dot_product_attention REQUIRES the scale= kwarg (float); pass 1.0/sqrt(head_dim).
- No nn.ModuleList -> assign a plain list (nn.Module is a dict subclass; lists/dicts of submodules are auto-traversed by parameters()).
- model.parameters() returns a NESTED dict (not a flat list) -> count_params uses mlx.utils.tree_flatten; there is NO model.named_parameters() -> get names via tree_flatten(model.parameters()).
- Cross-entropy is mlx.nn.losses.cross_entropy (NOT mx.loss); default reduction="none", pass reduction="mean" for a scalar.
- RMSNorm weight init to ones (mx.ones); linears/embedding use MLX default init.

Refinement (2026-08-24): forward + GQA tests now load the REAL 50M config (FIFTY_M_CONFIG) instead of a toy _tiny_config() - dropped. Rationale (user principle): unit tests should be as close to real usage as possible. The real config is fast enough (~1s suite), and it exercises the production shape (head_dim=64, GQA 8:2) rather than a 2-layer/head_dim-16 multi-query toy. make check still green (37 tests).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Kestrel decoder-only transformer in src/kestrel/model/kestrel.py (RMSNorm, cos/sin RoPE, GQA attention via fused mx.fast.scaled_dot_product_attention, SwiGLU FFN, pre-norm blocks, tied embeddings, count_params) + tests/test_model_kestrel.py. Verified: 50m=50,675,200 (50.7M) / 150m=148,152,960 (148.2M) params within 5% of target, forward pass returns (B,T,vocab) with finite CE loss, GQA (n_kv<n_heads) + no biases. Added mypy follow_imports=skip override for mlx.nn (no stubs) in pyproject.toml. make check green (37 tests).
<!-- SECTION:FINAL_SUMMARY:END -->
