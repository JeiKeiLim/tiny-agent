"""Tests for the shared trainer (TASK-005.03)."""

import math
from pathlib import Path

import mlx.core as mx
import pytest
from pydantic import ValidationError

from kestrel.model.config import ModelConfig
from kestrel.model.io import load
from kestrel.model.kestrel import Kestrel
from kestrel.train.trainer import TrainerConfig, TrainResult, lr_at, train

TINY = ModelConfig(
    name="tiny",
    vocab_size=64,
    context_length=16,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    hidden_size=32,
    intermediate_size=64,
)


def _batches(n: int, b: int, t: int, vocab: int, seed: int = 0) -> list[tuple[mx.array, mx.array]]:
    """Learnable counting batches: token at pos i is (start + i) % vocab, so next = cur + 1."""
    mx.random.seed(seed)
    out: list[tuple[mx.array, mx.array]] = []
    for _ in range(n):
        start = mx.random.randint(0, vocab, (b,))
        pos = mx.arange(t)
        x = ((start[:, None] + pos[None, :]) % vocab).astype(mx.int32)
        target = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
        out.append((x, target))
    return out


def _tiny_model() -> Kestrel:
    mx.random.seed(0)
    return Kestrel(TINY)


def test_train_loss_decreases(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(30, 2, 16, 64, seed=1)
    val_ds = _batches(5, 2, 16, 64, seed=2)
    cfg = TrainerConfig(
        num_steps=30,
        warmup_steps=5,
        eval_every=10,
        eval_iters=3,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path),
    )
    result = train(model, train_ds, val_ds, cfg)
    assert isinstance(result, TrainResult)
    assert result.num_steps == 30
    first = result.history[0][1]
    last = result.history[-1][1]
    assert math.isfinite(last)
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"


def test_trainer_config_strict() -> None:
    TrainerConfig(lr=1e-3)  # valid
    with pytest.raises(ValidationError):
        TrainerConfig(lr="fast")  # type: ignore[arg-type]  # mistyped
    with pytest.raises(ValidationError):
        TrainerConfig(bogus=1)  # type: ignore[call-arg]  # unknown key


def test_trainer_config_default_betas() -> None:
    assert TrainerConfig().betas == (0.9, 0.95)  # beta2=0.95 is the LLaMA/modern default
    TrainerConfig(betas=(0.9, 0.99))  # custom betas accepted


def test_lr_schedule() -> None:
    cfg = TrainerConfig(lr=1.0, warmup_steps=10, num_steps=50)
    assert lr_at(0, cfg) == 0.0  # starts at 0
    assert lr_at(10, cfg) == pytest.approx(1.0)  # peak at end of warmup
    assert lr_at(50, cfg) == pytest.approx(0.0, abs=1e-9)  # ~0 at the end
    warmup = [lr_at(s, cfg) for s in range(11)]
    assert warmup == sorted(warmup)  # monotonic increase during warmup
    decay = [lr_at(s, cfg) for s in range(10, 51)]
    assert decay == sorted(decay, reverse=True)  # monotonic decrease during decay


def test_checkpoint_reloads(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(12, 2, 16, 64, seed=3)
    val_ds = _batches(3, 2, 16, 64, seed=4)
    cfg = TrainerConfig(
        num_steps=10,
        warmup_steps=2,
        save_every=5,
        eval_every=100,
        log_every=100,
        output_dir=str(tmp_path),
    )
    train(model, train_ds, val_ds, cfg)
    reloaded = load(TINY, tmp_path / "final")
    x = train_ds[0][0]
    mx.eval(model.parameters(), reloaded.parameters())
    assert mx.allclose(model(x), reloaded(x))


def test_in_loop_val_loss(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(30, 2, 16, 64, seed=5)
    val_ds = _batches(5, 2, 16, 64, seed=6)
    cfg = TrainerConfig(
        num_steps=30,
        warmup_steps=5,
        eval_every=10,
        eval_iters=3,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path),
    )
    result = train(model, train_ds, val_ds, cfg)
    val_entries = [h[2] for h in result.history if h[2] is not None]
    assert len(val_entries) == 3  # val ran at steps 10, 20, 30
    assert all(math.isfinite(v) for v in val_entries)
    assert result.best_val_loss == min(val_entries)
