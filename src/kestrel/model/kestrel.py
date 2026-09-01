"""Kestrel decoder-only transformer (plan §9).

A small GPT-style model: pre-norm RMSNorm, rotary position embeddings (RoPE),
SwiGLU feed-forward, causal grouped-query attention (GQA) via
``mx.fast.scaled_dot_product_attention``, and tied input/output embeddings.
No biases, dropout 0. ``Kestrel.__call__`` returns logits of shape
 ``(B, T, vocab_size)``; the cross-entropy loss is computed by the caller.

When ``doc_ids`` of shape ``(B, T)`` is provided, attention is restricted to
tokens with the same document id, and RoPE positions reset to 0 at every
document boundary. Without ``doc_ids``, the model uses ordinary causal
attention.

``Kestrel.prefill`` and ``Kestrel.decode`` provide an inference-only KV-cache
path. The training ``__call__`` path, checkpoint format, and document-aware
forward path remain unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from kestrel.model.cache import KVCache
from kestrel.model.config import ModelConfig


def _freqs_cis_at(dim: int, positions: mx.array, theta: float) -> tuple[mx.array, mx.array]:
    """Rotary tables at explicit positions.

    For ``positions`` of shape ``(T,)`` the tables have shape ``(T, dim//2)``.
    For ``positions`` of shape ``(B, T)`` the tables have shape
    ``(B, 1, T, dim//2)`` so they broadcast over attention heads.
    """
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))
    angles = positions[..., None] * freqs
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    if cos.ndim == 3:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    return cos, sin


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    """Rotary (cos, sin) tables for positions ``0..end-1``, each of shape (end, dim//2)."""
    return _freqs_cis_at(dim, mx.arange(end), theta)


def document_positions(doc_ids: mx.array) -> mx.array:
    """Per-position offsets inside each document for ``doc_ids`` of shape ``(B, T)``."""
    B, T = doc_ids.shape
    first = mx.ones((B, 1), dtype=mx.bool_)
    changed = mx.not_equal(doc_ids[:, 1:], doc_ids[:, :-1])
    starts = mx.concatenate([first, changed], axis=1)
    arange = mx.arange(T)[None, :]
    start_idx = mx.where(starts, arange, mx.zeros((B, T), dtype=mx.int32))
    last_start = mx.cummax(start_idx, axis=1)
    return arange - last_start


def apply_rotary_emb(
    xq: mx.array, xk: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply rotary embeddings to query/key tensors of shape (B, H, T, D)."""

    def _apply(x: mx.array) -> mx.array:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(xq), _apply(xk)


def _causal_mask(query_start: int, query_end: int, key_len: int) -> mx.array:
    query_pos = mx.arange(query_start, query_end)[:, None]
    key_pos = mx.arange(key_len)[None, :]
    return (key_pos <= query_pos)[None, None, :, :]


def _document_mask(doc_ids: mx.array, query_start: int, query_end: int, key_len: int) -> mx.array:
    """Boolean attention mask for causal, same-document attention.

    The mask is true only when ``key_pos <= query_pos`` and the key/query tokens
    share the same document id. Shape is ``(B, 1, T_q, T_k)``.
    """
    query_doc = doc_ids[:, query_start:query_end][:, None, :, None]
    key_doc = doc_ids[:, :key_len][:, None, None, :]
    same_doc = query_doc == key_doc
    return _causal_mask(query_start, query_end, key_len) & same_doc


def causal_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    chunk_size: int = 1024,
    doc_ids: mx.array | None = None,
) -> mx.array:
    """Causal or document-aware GQA scaled-dot-product attention.

    Uses the fused causal path when ``doc_ids`` is None and the query sequence
    fits in one chunk. Otherwise processes query chunks against the full
    key/value sequence with an explicit boolean mask.
    """
    T = q.shape[2]
    if doc_ids is None and chunk_size >= T:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

    outs: list[mx.array] = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        qc = q[:, :, start:end, :]
        if doc_ids is None and end == T:
            out = mx.fast.scaled_dot_product_attention(qc, k, v, scale=scale, mask="causal")
        else:
            mask = (
                _causal_mask(start, end, T)
                if doc_ids is None
                else _document_mask(doc_ids, start, end, T)
            )
            out = mx.fast.scaled_dot_product_attention(qc, k, v, scale=scale, mask=mask)
        outs.append(out)
    return mx.concatenate(outs, axis=2)


class RMSNorm(nn.Module):  # type: ignore[misc]
    """Root-mean-square layer norm (no bias, no mean centering)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class Attention(nn.Module):  # type: ignore[misc]
    """Causal grouped-query attention with RoPE (query-chunked SDPA, native GQA)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        doc_ids: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, T = x.shape[0], x.shape[1]
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rotary_emb(q, k, cos, sin)
        scale = 1.0 / (self.head_dim**0.5)
        if cache is None:
            out = causal_sdpa(q, k, v, scale=scale, doc_ids=doc_ids)
        else:
            if doc_ids is not None:
                msg = "KV-cache generation does not support doc_ids"
                raise ValueError(msg)
            prefix = cache.length
            cache.write(k, v)
            keys = cache.keys()
            values = cache.values()
            if prefix == 0:
                out = mx.fast.scaled_dot_product_attention(
                    q, keys, values, scale=scale, mask="causal" if T > 1 else None
                )
            elif T == 1:
                out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale)
            else:
                key_len = cache.length
                query_pos = mx.arange(prefix, key_len)[:, None]
                key_pos = mx.arange(key_len)[None, :]
                mask = (key_pos <= query_pos)[None, None, :, :]
                out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)  # type: ignore[no-any-return]


class FeedForward(nn.Module):  # type: ignore[misc]
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        return self.down_proj(gate * mx.sigmoid(gate) * self.up_proj(x))  # type: ignore[no-any-return]


class TransformerBlock(nn.Module):  # type: ignore[misc]
    """Pre-norm block: attention + feed-forward with residuals."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        doc_ids: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, doc_ids, cache)
        return x + self.ffn(self.ffn_norm(x))


class Kestrel(nn.Module):  # type: ignore[misc]
    """Kestrel decoder-only transformer. Returns logits (B, T, vocab)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.n_heads
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.final_norm = RMSNorm(config.hidden_size)

    def __call__(self, x: mx.array, doc_ids: mx.array | None = None) -> mx.array:
        if doc_ids is None:
            cos, sin = precompute_freqs_cis(self.head_dim, x.shape[1], self.config.rope_theta)
        else:
            positions = document_positions(doc_ids)
            cos, sin = _freqs_cis_at(self.head_dim, positions, self.config.rope_theta)
        return self._forward(x, cos, sin, doc_ids, None)

    def prefill(self, x: mx.array, reserve: int = 0) -> tuple[mx.array, list[KVCache]]:
        """Run a full prompt forward pass and return logits plus fresh KV caches."""
        if x.ndim != 2:
            msg = f"x must have shape (B, T), got {x.shape}"
            raise ValueError(msg)
        if x.shape[1] == 0:
            msg = "prefill requires a non-empty prompt"
            raise ValueError(msg)
        if reserve < 0:
            msg = f"reserve must be >= 0, got {reserve}"
            raise ValueError(msg)
        caches = [
            KVCache(
                x.shape[0],
                self.config.n_kv_heads,
                self.head_dim,
                x.shape[1] + reserve,
                dtype=self.embed.weight.dtype,
            )
            for _ in self.layers
        ]
        cos, sin = precompute_freqs_cis(self.head_dim, x.shape[1], self.config.rope_theta)
        logits = self._forward(x, cos, sin, None, caches)
        return logits, caches

    def decode(self, x: mx.array, caches: list[KVCache]) -> tuple[mx.array, list[KVCache]]:
        """Run a single-token decode step against existing KV caches."""
        if x.ndim != 2 or x.shape[1] != 1:
            msg = f"decode requires x with shape (B, 1), got {x.shape}"
            raise ValueError(msg)
        if not caches or len(caches) != len(self.layers):
            msg = f"decode requires {len(self.layers)} caches, got {len(caches)}"
            raise ValueError(msg)
        position = caches[0].length
        cos, sin = _freqs_cis_at(self.head_dim, mx.array([position]), self.config.rope_theta)
        logits = self._forward(x, cos, sin, None, caches)
        return logits, caches

    def _forward(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        doc_ids: mx.array | None,
        caches: list[KVCache] | None,
    ) -> mx.array:
        h = self.embed(x)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, doc_ids, caches[i] if caches is not None else None)
        h = self.final_norm(h)
        return mx.matmul(h, self.embed.weight.T)


def count_params(model: nn.Module) -> int:
    """Total number of scalar parameters in ``model``."""
    flat = tree_flatten(model.parameters())
    return sum(p.size for _, p in flat)  # type: ignore[str-unpack, union-attr]
