---
id: TASK-002.01
title: Kestrel model architecture (decoder-only transformer)
status: To Do
assignee: []
created_date: '2026-08-24 00:15'
updated_date: '2026-08-24 00:17'
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
- [ ] #1 Instantiate Kestrel-50M and Kestrel-150M from their configs; count_params lands near ~50M / ~150M (asserted, not hand-waved)
- [ ] #2 Forward pass on random token IDs returns logits of shape (B, T, vocab) with finite cross-entropy loss
- [ ] #3 GQA exercised (n_kv_heads=2 < n_heads) via fused attention; no biases anywhere; dropout 0
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
<!-- SECTION:NOTES:END -->
