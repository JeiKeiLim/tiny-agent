"""Tests for the Kestrel decoder-only transformer (TASK-002.01)."""

import mlx.core as mx
import pytest
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten

from kestrel.common.config import load_config
from kestrel.model.config import ModelConfig
from kestrel.model.kestrel import Kestrel, causal_sdpa, count_params, document_positions

MODEL_50M = "configs/kestrel/50m/model.yaml"
MODEL_150M = "configs/kestrel/150m/model.yaml"
REAL_CONFIGS = [
    (MODEL_50M, 50_000_000),
    (MODEL_150M, 150_000_000),
]


@pytest.mark.parametrize("path,expected", REAL_CONFIGS)
def test_param_count_near_expected(path: str, expected: int) -> None:
    config = load_config(path, ModelConfig)
    model = Kestrel(config)
    n = count_params(model)
    assert abs(n - expected) / expected < 0.05, f"{config.name}: {n} not within 5% of {expected}"


def test_forward_shape_and_finite_loss() -> None:
    config = load_config(MODEL_50M, ModelConfig)
    model = Kestrel(config)
    B, T = 1, 8
    x = mx.random.randint(0, config.vocab_size, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, config.vocab_size)
    loss = cross_entropy(logits[:, :-1], x[:, 1:], reduction="mean")
    assert mx.isfinite(loss).item()


def test_gqa_and_no_biases() -> None:
    config = load_config(MODEL_50M, ModelConfig)
    assert config.n_kv_heads < config.n_heads
    model = Kestrel(config)
    names = [name for name, _ in tree_flatten(model.parameters())]
    assert names, "model has no parameters"
    assert not any("bias" in name for name in names)


def test_causal_sdpa_chunked_matches_full() -> None:
    mx.random.seed(0)
    B, N, Nkv, T, D = 2, 4, 2, 16, 8
    scale = D**-0.5
    q = mx.random.normal((B, N, T, D))
    k = mx.random.normal((B, Nkv, T, D))
    v = mx.random.normal((B, Nkv, T, D))

    full = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    chunked = causal_sdpa(q, k, v, scale=scale, chunk_size=8)

    diff = mx.max(mx.abs(full - chunked)).item()
    assert diff < 1e-5, f"chunked causal SDPA mismatch: {diff}"


def test_attention_is_causal() -> None:
    mx.random.seed(1)
    config = ModelConfig(
        name="tiny",
        vocab_size=32,
        context_length=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        hidden_size=16,
        intermediate_size=32,
        rope_theta=10000.0,
    )
    model = Kestrel(config)

    prefix = mx.random.randint(0, config.vocab_size, (1, 5))
    future_a = mx.random.randint(0, config.vocab_size, (1, 3))
    future_b = mx.random.randint(0, config.vocab_size, (1, 3))
    x_a = mx.concatenate([prefix, future_a], axis=1)
    x_b = mx.concatenate([prefix, future_b], axis=1)

    logits_a = model(x_a)
    logits_b = model(x_b)

    diff = mx.max(mx.abs(logits_a[:, :5] - logits_b[:, :5])).item()
    assert diff < 1e-4, f"future tokens changed earlier logits: {diff}"


def _tiny_doc_config() -> ModelConfig:
    return ModelConfig(
        name="tiny",
        vocab_size=32,
        context_length=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        hidden_size=16,
        intermediate_size=32,
        rope_theta=10000.0,
    )


def test_document_positions_reset_at_doc_id_changes() -> None:
    doc_ids = mx.array([[0, 0, 1, 1, 1, 2]])
    assert document_positions(doc_ids).tolist() == [[0, 1, 0, 1, 2, 0]]

    all_zeros = mx.zeros((1, 4), dtype=mx.int32)
    assert document_positions(all_zeros).tolist() == [[0, 1, 2, 3]]


def test_document_attention_blocks_previous_document() -> None:
    mx.random.seed(3)
    model = Kestrel(_tiny_doc_config())

    before = mx.random.randint(0, 32, (1, 2))
    after = mx.random.randint(0, 32, (1, 5))
    x_a = mx.concatenate([before, mx.array([[1]]), after], axis=1)
    x_b = mx.concatenate([before, mx.array([[2]]), after], axis=1)
    doc_ids = mx.array([[0, 0, 0, 1, 1, 1, 1, 1]])

    logits_a = model(x_a, doc_ids)
    logits_b = model(x_b, doc_ids)

    diff = mx.max(mx.abs(logits_a[:, 3:] - logits_b[:, 3:])).item()
    assert diff < 1e-4, f"previous document changed later document logits: {diff}"


def test_all_zero_doc_ids_match_causal_path() -> None:
    mx.random.seed(4)
    model = Kestrel(_tiny_doc_config())
    x = mx.random.randint(0, 32, (1, 8))

    logits_plain = model(x)
    logits_zero = model(x, mx.zeros((1, 8), dtype=mx.int32))

    diff = mx.max(mx.abs(logits_plain - logits_zero)).item()
    assert diff < 1e-3, f"all-zero doc_ids changed logits: {diff}"
