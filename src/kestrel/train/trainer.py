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

import json
import math
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten, tree_map
from pydantic import Field

from kestrel.common.config import BaseConfig
from kestrel.train.checkpoint import CheckpointContext, save_full_checkpoint


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
    keep_latest_checkpoints: int | None = Field(default=3, ge=1)
    keep_best_checkpoint: bool = True


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


@dataclass
class ResumeState:
    """Training state restored from a full checkpoint before continuing."""

    step: int
    schedule_steps: int
    best_val_loss: float | None
    last_train_loss: float | None
    last_val_loss: float | None
    last_eval_step: int | None
    optimizer: optim.Optimizer
    checkpoint_dir: Path | None = None


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


_STEP_CHECKPOINT_RE = re.compile(r"^step_(\d+)$")


def _prune_step_checkpoints(output_dir: Path, keep_latest: int | None) -> None:
    """Delete old ``step_NNNNNN`` checkpoint directories, keeping the newest ones.

    Only immediate subdirectories matching ``step_<digits>`` are eligible.
    ``best``, ``final``, and unrecognized paths are never touched. Recency is
    determined by directory mtime with the step number as a tie-breaker, so a
    reused output directory containing stale high-step checkpoints does not
    cause fresh low-step checkpoints from a new run to be deleted.
    """
    if keep_latest is None or not output_dir.is_dir():
        return
    candidates: list[tuple[float, int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = _STEP_CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        candidates.append((path.stat().st_mtime, int(match.group(1)), path))
    candidates.sort(key=lambda item: (item[0], item[1]))
    for _, _, path in candidates[:-keep_latest]:
        shutil.rmtree(path)


def _dataset_state(dataset: Any) -> dict[str, Any] | None:
    """Return checkpointable dataset state when the iterator exposes it."""
    state_dict = getattr(dataset, "state_dict", None)
    if callable(state_dict):
        return cast(dict[str, Any], state_dict())
    return None


def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    path: Path,
    config: TrainerConfig,
    *,
    step: int,
    best_val_loss: float | None,
    last_train_loss: float | None,
    last_val_loss: float | None,
    last_eval_step: int | None,
    horizon: int,
    dataset: Any,
    context: CheckpointContext | None,
    run_log_path: Path,
) -> None:
    state = {
        "kind": path.name,
        "step": step,
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "last_eval_step": last_eval_step,
        "schedule_steps": horizon,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "optimizer": {
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "betas": list(config.betas),
        },
        "dataset_state": _dataset_state(dataset),
    }
    save_full_checkpoint(
        model,
        optimizer,
        path,
        state,
        context=context,
        run_log_path=run_log_path,
    )


def train(
    model: nn.Module,
    dataset: Iterable[tuple[mx.array, ...]],
    val_dataset: Iterable[tuple[mx.array, ...]],
    config: TrainerConfig,
    schedule_steps: int | None = None,
    *,
    resume: ResumeState | None = None,
    checkpoint_context: CheckpointContext | None = None,
) -> TrainResult:
    """Run the training loop and return a :class:`TrainResult`.

    Checkpoints are written to ``config.output_dir`` every ``save_every`` steps
    (``step_<n>``) and a ``final`` checkpoint at the end. A ``best`` checkpoint
    is written whenever validation loss strictly improves, and old step
    checkpoints are pruned according to ``keep_latest_checkpoints``.

    ``config.num_steps > 0`` is a hard stop cap. ``config.num_steps <= 0`` runs
    until the dataset is exhausted; ``schedule_steps`` (or the dataset's
    ``estimated_steps()`` when available) sets the LR decay horizon.
    """
    auto_steps = config.num_steps <= 0
    history: list[tuple[int, float, float | None]] = []
    opt: optim.Optimizer

    if resume is None:
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
        step = 0
        best_val: float | None = None
        last_train: float | None = None
        last_val: float | None = None
        last_eval_step: int | None = None
    else:
        horizon = max(1, resume.schedule_steps)
        opt = resume.optimizer
        step = resume.step
        best_val = resume.best_val_loss
        last_train = resume.last_train_loss
        last_val = resume.last_val_loss
        last_eval_step = resume.last_eval_step

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / "run.jsonl"
    steps_label = str(config.num_steps) if not auto_steps else str(horizon)
    train_iter = iter(dataset)

    with run_log_path.open("a", encoding="utf-8") as run_log:
        while True:
            if config.num_steps > 0 and step >= config.num_steps:
                break
            try:
                batch = next(train_iter)
            except StopIteration:
                break

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
                if math.isfinite(val_loss) and (best_val is None or val_loss < best_val):
                    best_val = val_loss
                    if config.keep_best_checkpoint:
                        _save_checkpoint(
                            model,
                            opt,
                            output_dir / "best",
                            config,
                            step=step + 1,
                            best_val_loss=best_val,
                            last_train_loss=loss,
                            last_val_loss=val_loss,
                            last_eval_step=step + 1,
                            horizon=horizon,
                            dataset=train_iter,
                            context=checkpoint_context,
                            run_log_path=run_log_path,
                        )
            if config.log_every > 0 and (step + 1) % config.log_every == 0:
                msg = f"step {step + 1}/{steps_label} loss {loss:.4f} lr {opt.learning_rate:.2e}"
                if val_loss is not None:
                    msg += f" val {val_loss:.4f}"
                print(msg)
            if config.save_every > 0 and (step + 1) % config.save_every == 0:
                _save_checkpoint(
                    model,
                    opt,
                    output_dir / f"step_{step + 1:06d}",
                    config,
                    step=step + 1,
                    best_val_loss=best_val,
                    last_train_loss=loss,
                    last_val_loss=val_loss,
                    last_eval_step=step + 1 if val_loss is not None else last_eval_step,
                    horizon=horizon,
                    dataset=train_iter,
                    context=checkpoint_context,
                    run_log_path=run_log_path,
                )
                _prune_step_checkpoints(output_dir, config.keep_latest_checkpoints)

            entry: dict[str, float | int] = {
                "step": step + 1,
                "train_loss": loss,
                "lr": float(opt.learning_rate),
            }
            if val_loss is not None:
                entry["val_loss"] = val_loss
            run_log.write(json.dumps(entry) + "\n")
            run_log.flush()

            history.append((step + 1, loss, val_loss))
            last_train = loss
            last_val = val_loss
            if val_loss is not None:
                last_eval_step = step + 1
            step += 1

        _save_checkpoint(
            model,
            opt,
            output_dir / "final",
            config,
            step=step,
            best_val_loss=best_val,
            last_train_loss=last_train,
            last_val_loss=last_val,
            last_eval_step=last_eval_step,
            horizon=horizon,
            dataset=train_iter,
            context=checkpoint_context,
            run_log_path=run_log_path,
        )

    if history:
        final_loss = history[-1][1]
    else:
        final_loss = last_train if last_train is not None else float("inf")
    return TrainResult(
        final_loss=final_loss,
        num_steps=step,
        best_val_loss=best_val,
        schedule_steps=horizon if (auto_steps or resume is not None) else None,
        history=history,
    )
