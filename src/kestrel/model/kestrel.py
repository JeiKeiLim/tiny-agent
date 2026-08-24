"""Kestrel decoder-only transformer (plan §9).

A small GPT-style model: pre-norm RMSNorm, rotary position embeddings (RoPE),
SwiGLU feed-forward, grouped-query attention (GQA) via fused
``mx.fast.scaled_dot_product_attention``, and tied input/output embeddings.
No biases, dropout 0. ``Kestrel.__call__`` returns logits of shape
``(B, T, vocab_size)``; the cross-entropy loss is computed by the caller.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from kestrel.model.config import ModelConfig


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    """Rotary (cos, sin) tables for positions ``0..end-1``, each of shape (end, dim//2)."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))
    angles = mx.outer(mx.arange(0, end), freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rotary_emb(
    xq: mx.array, xk: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply rotary embeddings to query/key tensors of shape (B, H, T, D)."""

    def _apply(x: mx.array) -> mx.array:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(xq), _apply(xk)


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
    """Grouped-query attention with RoPE (fused SDPA, native GQA)."""

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

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T = x.shape[0], x.shape[1]
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rotary_emb(q, k, cos, sin)
        scale = 1.0 / (self.head_dim**0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
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

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin)
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

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        cos, sin = precompute_freqs_cis(self.head_dim, x.shape[1], self.config.rope_theta)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.final_norm(h)
        return mx.matmul(h, self.embed.weight.T)


def count_params(model: nn.Module) -> int:
    """Total number of scalar parameters in ``model``."""
    flat = tree_flatten(model.parameters())
    return sum(p.size for _, p in flat)  # type: ignore[str-unpack, union-attr]
