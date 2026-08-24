"""Shared training loop (TASK-005.03).

A minimal MLX trainer shared by every phase (pretrain / SFT / RL): AdamW +
linear-warmup/cosine-decay LR schedule + global-norm gradient clipping +
in-loop validation loss + periodic checkpointing.

The model is a ``Kestrel`` (``model(x) -> logits`` of shape ``(B, T, V)``); the
dataset yields ``(input, target)`` int32 batches of shape ``(batch_size,
seq_len)`` (see ``data/pretrain_dataset.py``). The next-token loss is
``cross_entropy(logits[:, :-1], target[:, :-1])`` (matches
``scripts/check_model.py``).

The step is run eagerly (no ``@mx.compile``): MLX's compile cache keys on
argument shapes and captures the parameter arrays at trace time, so the
in-place ``optimizer.update`` changes are invisible to the cached graph (stale
params, no learning). At our scale (50M) compile buys nothing, so eager is used.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten, tree_map
from pydantic import Field

from kestrel.common.config import BaseConfig
from kestrel.model.io import save as save_checkpoint


class TrainerConfig(BaseConfig):
    """Strict settings for the shared trainer (no unknown keys)."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 8
    seq_len: int = 2048
    num_steps: int = 1000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    save_every: int = 500
    log_every: int = 10
    eval_every: int = 100
    eval_iters: int = 5
    output_dir: str = "checkpoints/train"


class TrainResult(BaseConfig):
    """Summary of a training run.

    ``history`` holds one ``(step, train_loss, val_loss)`` triple per step;
    ``val_loss`` is ``None`` except on steps where validation ran.
    """

    final_loss: float
    num_steps: int
    best_val_loss: float | None = None
    history: list[tuple[int, float, float | None]] = Field(default_factory=list)


def lr_at(step: int, cfg: TrainerConfig) -> float:
    """LR at ``step``: linear warmup 0->lr over ``warmup_steps``, then cosine decay to 0."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    decay = max(1, cfg.num_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _clip_grads(grad: Any, max_norm: float) -> Any:
    """Global-norm clip: scale all grads so their combined L2 norm is <= ``max_norm``."""
    flat = cast(list[tuple[str, Any]], tree_flatten(grad))
    total_sq = mx.array(0.0)
    for _, g in flat:
        total_sq = total_sq + mx.sum(g * g)
    coef = mx.minimum(1.0, max_norm / (mx.sqrt(total_sq) + 1e-6))
    return tree_map(lambda g: g * coef, grad)


def _batch_loss(model: nn.Module, x: mx.array, target: mx.array) -> mx.array:
    """Next-token cross-entropy for one ``(input, target)`` batch."""
    logits = model(x)
    return cross_entropy(logits[:, :-1], target[:, :-1], reduction="mean")


def estimate_val_loss(
    model: nn.Module, val: Iterable[tuple[mx.array, mx.array]], iters: int
) -> float:
    """Mean next-token loss over up to ``iters`` validation batches."""
    total = 0.0
    n = 0
    for x, target in val:
        total += cast(float, _batch_loss(model, x, target).item())
        n += 1
        if n >= iters:
            break
    return total / n if n else float("inf")


def train(
    model: nn.Module,
    dataset: Iterable[tuple[mx.array, mx.array]],
    val_dataset: Iterable[tuple[mx.array, mx.array]],
    config: TrainerConfig,
) -> TrainResult:
    """Run the training loop and return a :class:`TrainResult`.

    Checkpoints are written to ``config.output_dir`` every ``save_every`` steps
    (``step_<n>``) and a ``final`` checkpoint at the end.
    """
    opt = optim.AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
    history: list[tuple[int, float, float | None]] = []
    best_val: float | None = None
    step = 0
    for x, target in dataset:
        opt.learning_rate = lr_at(step, config)

        def loss_fn(m: nn.Module, _x: mx.array = x, _t: mx.array = target) -> mx.array:
            return _batch_loss(m, _x, _t)

        value, grad = mx.value_and_grad(loss_fn)(model)
        opt.update(model, _clip_grads(grad, config.grad_clip))
        mx.eval(model.parameters())
        loss = cast(float, value.item())

        val_loss: float | None = None
        if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
            val_loss = estimate_val_loss(model, val_dataset, config.eval_iters)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
        if config.log_every > 0 and (step + 1) % config.log_every == 0:
            msg = f"step {step + 1}/{config.num_steps} loss {loss:.4f} lr {opt.learning_rate:.2e}"
            if val_loss is not None:
                msg += f" val {val_loss:.4f}"
            print(msg)
        if config.save_every > 0 and (step + 1) % config.save_every == 0:
            save_checkpoint(model, Path(config.output_dir) / f"step_{step + 1:06d}")

        history.append((step + 1, loss, val_loss))
        step += 1
        if step >= config.num_steps:
            break

    save_checkpoint(model, Path(config.output_dir) / "final")
    final_loss = history[-1][1] if history else float("inf")
    return TrainResult(
        final_loss=final_loss, num_steps=step, best_val_loss=best_val, history=history
    )
