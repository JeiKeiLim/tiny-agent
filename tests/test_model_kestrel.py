"""Tests for the Kestrel decoder-only transformer (TASK-002.01)."""

import mlx.core as mx
import pytest
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten

from kestrel.common.config import load_config
from kestrel.model.config import ModelConfig
from kestrel.model.kestrel import Kestrel, count_params

REAL_CONFIGS = [
    ("configs/kestrel/50m/model.yaml", 50_000_000),
    ("configs/kestrel/150m/model.yaml", 150_000_000),
]


@pytest.mark.parametrize("path,expected", REAL_CONFIGS)
def test_param_count_near_expected(path: str, expected: int) -> None:
    config = load_config(path, ModelConfig)
    model = Kestrel(config)
    n = count_params(model)
    assert abs(n - expected) / expected < 0.05, f"{config.name}: {n} not within 5% of {expected}"


def _tiny_config() -> ModelConfig:
    # n_kv_heads < n_heads so GQA is exercised; hidden_size // n_heads is even.
    return ModelConfig(
        name="tiny",
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        hidden_size=32,
        intermediate_size=64,
    )


def test_forward_shape_and_finite_loss() -> None:
    config = _tiny_config()
    model = Kestrel(config)
    B, T = 1, 8
    x = mx.random.randint(0, config.vocab_size, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, config.vocab_size)
    loss = cross_entropy(logits[:, :-1], x[:, 1:], reduction="mean")
    assert mx.isfinite(loss).item()


def test_gqa_and_no_biases() -> None:
    config = _tiny_config()
    assert config.n_kv_heads < config.n_heads
    model = Kestrel(config)
    names = [name for name, _ in tree_flatten(model.parameters())]
    assert names, "model has no parameters"
    assert not any("bias" in name for name in names)
