"""KV cache for incremental Kestrel generation."""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    """Preallocated per-layer key/value cache with chunk growth.

    The cache stores keys and values for every token already processed by the
    corresponding attention layer. ``write`` appends a block of tokens at the
    current position; ``keys`` and ``values`` return the populated prefix.
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        capacity: int,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        if n_kv_heads < 1:
            msg = f"n_kv_heads must be >= 1, got {n_kv_heads}"
            raise ValueError(msg)
        if head_dim < 1:
            msg = f"head_dim must be >= 1, got {head_dim}"
            raise ValueError(msg)
        if capacity < 0:
            msg = f"capacity must be >= 0, got {capacity}"
            raise ValueError(msg)
        self._batch_size = batch_size
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._capacity = capacity
        self._length = 0
        shape = (batch_size, n_kv_heads, capacity, head_dim)
        self._k = mx.zeros(shape, dtype=dtype)
        self._v = mx.zeros(shape, dtype=dtype)

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._capacity

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_capacity = max(needed, self._capacity * 2, 8)
        extra = new_capacity - self._capacity
        pad_shape = (self._batch_size, self._n_kv_heads, extra, self._head_dim)
        self._k = mx.concatenate([self._k, mx.zeros(pad_shape, dtype=self._k.dtype)], axis=2)
        self._v = mx.concatenate([self._v, mx.zeros(pad_shape, dtype=self._v.dtype)], axis=2)
        self._capacity = new_capacity

    def write(self, k: mx.array, v: mx.array) -> None:
        """Append key/value tensors of shape ``(B, n_kv_heads, T, head_dim)``."""
        if k.shape != v.shape:
            msg = f"k and v shapes must match, got {k.shape} and {v.shape}"
            raise ValueError(msg)
        if k.ndim != 4:
            msg = f"k must have shape (B, n_kv_heads, T, head_dim), got {k.shape}"
            raise ValueError(msg)
        expected = (self._batch_size, self._n_kv_heads, k.shape[2], self._head_dim)
        if k.shape != expected:
            msg = f"k shape must be {expected}, got {k.shape}"
            raise ValueError(msg)
        t = k.shape[2]
        if t == 0:
            return
        self._ensure_capacity(self._length + t)
        self._k[:, :, self._length : self._length + t, :] = k
        self._v[:, :, self._length : self._length + t, :] = v
        self._length += t

    def keys(self) -> mx.array:
        return self._k[:, :, : self._length, :]

    def values(self) -> mx.array:
        return self._v[:, :, : self._length, :]
