"""Tests for the KV cache used by Kestrel inference."""

import mlx.core as mx
import pytest

from kestrel.model.cache import KVCache


def test_kv_cache_write_and_read() -> None:
    cache = KVCache(batch_size=2, n_kv_heads=3, head_dim=4, capacity=5)
    k = mx.ones((2, 3, 2, 4))
    v = mx.full((2, 3, 2, 4), 2.0)

    cache.write(k, v)

    assert cache.length == 2
    assert cache.keys().shape == (2, 3, 2, 4)
    assert cache.values().shape == (2, 3, 2, 4)
    assert bool(mx.all(cache.keys() == 1.0).item())
    assert bool(mx.all(cache.values() == 2.0).item())


def test_kv_cache_appends_blocks() -> None:
    cache = KVCache(batch_size=1, n_kv_heads=1, head_dim=2, capacity=4)
    cache.write(mx.ones((1, 1, 2, 2)), mx.zeros((1, 1, 2, 2)))
    cache.write(mx.full((1, 1, 1, 2), 3.0), mx.zeros((1, 1, 1, 2)))

    assert cache.length == 3
    keys = cache.keys()
    assert bool(mx.all(keys[:, :, :2, :] == 1.0).item())
    assert bool(mx.all(keys[:, :, 2:, :] == 3.0).item())


def test_kv_cache_grows_when_capacity_is_insufficient() -> None:
    cache = KVCache(batch_size=1, n_kv_heads=1, head_dim=2, capacity=2)
    cache.write(mx.ones((1, 1, 2, 2)), mx.zeros((1, 1, 2, 2)))

    cache.write(mx.full((1, 1, 1, 2), 5.0), mx.zeros((1, 1, 1, 2)))

    assert cache.length == 3
    assert cache.capacity >= 3
    assert bool(mx.all(cache.keys()[:, :, 2:, :] == 5.0).item())


def test_kv_cache_rejects_mismatched_k_and_v() -> None:
    cache = KVCache(batch_size=1, n_kv_heads=1, head_dim=2, capacity=2)
    with pytest.raises(ValueError, match="shapes must match"):
        cache.write(mx.ones((1, 1, 1, 2)), mx.zeros((1, 1, 2, 2)))


def test_kv_cache_rejects_wrong_batch_or_heads() -> None:
    cache = KVCache(batch_size=1, n_kv_heads=1, head_dim=2, capacity=2)
    with pytest.raises(ValueError, match="k shape must be"):
        cache.write(mx.ones((2, 1, 1, 2)), mx.ones((2, 1, 1, 2)))


def test_kv_cache_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        KVCache(batch_size=1, n_kv_heads=1, head_dim=2, capacity=-1)
