"""Shared training loop (TASK-005.03).

A minimal MLX trainer shared by every phase (pretrain / SFT / RL): AdamW +
linear-warmup/cosine-decay LR schedule + global-norm gradient clipping +
in-loop validation loss + periodic checkpointing.

The model is a ``Kestrel`` (``model(x) -> logits`` of shape ``(B, T, V)``); the
dataset yields ``(input, target)`` or ``(input, target, doc_ids)`` int32 batches
of shape ``(batch_size, seq_len)`` (see ``data/pretrain_dataset.py``). The
next-token loss is ``cross_entropy(logits[:, :-1], target[:, :-1])`` (matches
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
    betas: tuple[float, float] = (0.9, 0.95)  # beta2=0.95 is the LLaMA/modern-LLM default
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
    schedule_steps: int | None = None
    history: list[tuple[int, float, float | None]] = Field(default_factory=list)


def lr_at(step: int, cfg: TrainerConfig, schedule_steps: int | None = None) -> float:
    """LR at ``step``: linear warmup 0->lr, then cosine decay to 0.

    The decay horizon is ``schedule_steps`` when provided, otherwise
    ``cfg.num_steps``. A non-positive horizon is clamped to 1 step.
    """
    horizon = schedule_steps if schedule_steps is not None and schedule_steps > 0 else cfg.num_steps
    horizon = max(1, horizon)
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    decay = max(1, horizon - cfg.warmup_steps)
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


def _unpack_batch(batch: tuple[mx.array, ...]) -> tuple[mx.array, mx.array, mx.array | None]:
    """Accept legacy ``(input, target)`` batches and document-aware 3-tuples."""
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    msg = f"expected a 2- or 3-tuple batch, got {len(batch)} elements"
    raise ValueError(msg)


def _batch_loss(
    model: nn.Module, x: mx.array, target: mx.array, doc_ids: mx.array | None = None
) -> mx.array:
    """Next-token cross-entropy for one pretraining batch."""
    logits = model(x, doc_ids)
    return cross_entropy(logits[:, :-1], target[:, :-1], reduction="mean")


def estimate_val_loss(model: nn.Module, val: Iterable[tuple[mx.array, ...]], iters: int) -> float:
    """Mean next-token loss over up to ``iters`` validation batches."""
    total = 0.0
    n = 0
    for batch in val:
        x, target, doc_ids = _unpack_batch(batch)
        total += cast(float, _batch_loss(model, x, target, doc_ids).item())
        n += 1
        if n >= iters:
            break
    return total / n if n else float("inf")


def train(
    model: nn.Module,
    dataset: Iterable[tuple[mx.array, ...]],
    val_dataset: Iterable[tuple[mx.array, ...]],
    config: TrainerConfig,
    schedule_steps: int | None = None,
) -> TrainResult:
    """Run the training loop and return a :class:`TrainResult`.

    Checkpoints are written to ``config.output_dir`` every ``save_every`` steps
    (``step_<n>``) and a ``final`` checkpoint at the end.

    ``config.num_steps > 0`` is a hard stop cap. ``config.num_steps <= 0`` runs
    until the dataset is exhausted; ``schedule_steps`` (or the dataset's
    ``estimated_steps()`` when available) sets the LR decay horizon.
    """
    auto_steps = config.num_steps <= 0
    if auto_steps and schedule_steps is None:
        estimated = getattr(dataset, "estimated_steps", None)
        if callable(estimated):
            schedule_steps = cast(int, estimated())
    if schedule_steps is None or schedule_steps <= 0:
        schedule_steps = config.num_steps
    horizon = max(1, schedule_steps)
    if auto_steps:
        print(f"num_steps <= 0: running until dataset exhaustion with schedule_steps={horizon}")

    opt = optim.AdamW(
        learning_rate=config.lr, betas=list(config.betas), weight_decay=config.weight_decay
    )
    history: list[tuple[int, float, float | None]] = []
    best_val: float | None = None
    steps_label = str(config.num_steps) if not auto_steps else str(horizon)
    step = 0
    for batch in dataset:
        x, target, doc_ids = _unpack_batch(batch)
        opt.learning_rate = lr_at(step, config, horizon)

        def loss_fn(
            m: nn.Module,
            _x: mx.array = x,
            _t: mx.array = target,
            _d: mx.array | None = doc_ids,
        ) -> mx.array:
            return _batch_loss(m, _x, _t, _d)

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
            msg = f"step {step + 1}/{steps_label} loss {loss:.4f} lr {opt.learning_rate:.2e}"
            if val_loss is not None:
                msg += f" val {val_loss:.4f}"
            print(msg)
        if config.save_every > 0 and (step + 1) % config.save_every == 0:
            save_checkpoint(model, Path(config.output_dir) / f"step_{step + 1:06d}")

        history.append((step + 1, loss, val_loss))
        step += 1
        if not auto_steps and step >= config.num_steps:
            break

    save_checkpoint(model, Path(config.output_dir) / "final")
    final_loss = history[-1][1] if history else float("inf")
    return TrainResult(
        final_loss=final_loss,
        num_steps=step,
        best_val_loss=best_val,
        schedule_steps=horizon if auto_steps else None,
        history=history,
    )
