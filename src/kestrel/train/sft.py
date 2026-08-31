"""SFT phase (TASK-007.03.08): wire a pretrain checkpoint into masked SFT.

``sft(config)`` loads a Kestrel model from a pretrain checkpoint, builds
:class:`~kestrel.data.sft_dataset.SFTDataset` iterables for training and
validation, and hands them to the shared
:func:`~kestrel.train.trainer.train` loop with ``use_loss_mask=true``.

``SFTConfig`` references the model YAML, the initial pretrain checkpoint, and
the SFT dataset by path. ``resume`` is a full SFT checkpoint directory from
which to continue training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import mlx.optimizers as optim
from pydantic import Field

from kestrel.common.config import BaseConfig, load_config
from kestrel.data.sft_dataset import SFTDataset, SFTDatasetConfig
from kestrel.model.config import ModelConfig
from kestrel.model.io import load as load_model
from kestrel.train.checkpoint import (
    CheckpointContext,
    load_optimizer_state,
    read_checkpoint_state,
    sha256_file,
)
from kestrel.train.trainer import ResumeState, TrainerConfig, TrainResult, train


class SFTConfig(BaseConfig):
    """Strict settings for the SFT phase (no unknown keys)."""

    model: str
    checkpoint: str
    dataset: SFTDatasetConfig
    val_dataset: SFTDatasetConfig | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    resume: str | None = None


def _validate_shapes(config: SFTConfig) -> None:
    if not config.trainer.use_loss_mask:
        msg = "SFT trainer requires use_loss_mask: true"
        raise ValueError(msg)
    if config.dataset.context_length != config.trainer.seq_len:
        msg = "dataset.context_length must match trainer.seq_len"
        raise ValueError(msg)
    if config.dataset.batch_size != config.trainer.batch_size:
        msg = "dataset.batch_size must match trainer.batch_size"
        raise ValueError(msg)


def _sft_snapshot(config: SFTConfig) -> dict[str, Any]:
    """Resolved SFT config stored in checkpoints (``resume`` normalized)."""
    dump = config.model_dump()
    dump["resume"] = None
    return dump


def _training_relevant_model(config: ModelConfig) -> dict[str, Any]:
    dump = config.model_dump()
    dump.pop("name", None)
    return dump


def _training_relevant_trainer(config: TrainerConfig) -> dict[str, Any]:
    return {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "warmup_steps": config.warmup_steps,
        "grad_clip": config.grad_clip,
        "num_steps": config.num_steps,
        "use_loss_mask": config.use_loss_mask,
    }


def _checkpoint_context(
    config: SFTConfig,
    model_cfg: ModelConfig,
    config_path: str | Path | None,
) -> CheckpointContext:
    raw_configs: dict[str, str] = {}
    if config_path is not None:
        raw_configs["sft.yaml"] = Path(config_path).read_text(encoding="utf-8")
    raw_configs["model.yaml"] = Path(config.model).read_text(encoding="utf-8")

    artifact_hashes: dict[str, str] = {}
    tokenizer_path = Path(config.dataset.tokenizer_path)
    if tokenizer_path.is_file():
        artifact_hashes["tokenizer_sha256"] = sha256_file(tokenizer_path)
    if config_path is not None:
        artifact_hashes["sft_config_sha256"] = sha256_file(config_path)
    artifact_hashes["model_config_sha256"] = sha256_file(config.model)

    dataset_input = Path(config.dataset.input)
    if dataset_input.is_file():
        artifact_hashes["dataset_input_sha256"] = sha256_file(dataset_input)
    if config.val_dataset is not None:
        val_input = Path(config.val_dataset.input)
        if val_input.is_file():
            artifact_hashes["val_dataset_input_sha256"] = sha256_file(val_input)

    return CheckpointContext(
        raw_configs=raw_configs,
        resolved_configs={
            "sft": _sft_snapshot(config),
            "model": model_cfg.model_dump(),
            "dataset": config.dataset.model_dump(),
            "trainer": config.trainer.model_dump(),
        },
        artifact_hashes=artifact_hashes,
        extra_state={
            "initial_checkpoint": config.checkpoint,
            "seed": config.dataset.seed,
        },
    )


def _require_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"checkpoint is missing resolved config snapshot: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _validate_resume(
    checkpoint_dir: Path,
    config: SFTConfig,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    state = read_checkpoint_state(checkpoint_dir)
    if state.get("dataset_state") is None:
        msg = f"checkpoint has no resumable dataset state: {checkpoint_dir}"
        raise ValueError(msg)

    resolved_dir = checkpoint_dir / "config" / "resolved"
    sft_snap = _require_json(resolved_dir / "sft.json")
    model_snap = _require_json(resolved_dir / "model.json")
    dataset_snap = _require_json(resolved_dir / "dataset.json")
    trainer_snap = _require_json(resolved_dir / "trainer.json")

    model_snap.pop("name", None)
    if sft_snap.get("checkpoint") != config.checkpoint:
        msg = "resume initial checkpoint does not match the checkpoint snapshot"
        raise ValueError(msg)
    if _training_relevant_model(model_cfg) != model_snap:
        msg = "resume model config does not match the checkpoint snapshot"
        raise ValueError(msg)
    if config.dataset.model_dump() != dataset_snap:
        msg = "resume dataset config does not match the checkpoint snapshot"
        raise ValueError(msg)

    trainer_keys = tuple(_training_relevant_trainer(TrainerConfig()).keys())
    trainer_selected = {key: trainer_snap.get(key) for key in trainer_keys}
    if _training_relevant_trainer(config.trainer) != trainer_selected:
        msg = "resume trainer config does not match the checkpoint snapshot"
        raise ValueError(msg)

    expected_hashes = cast(dict[str, str], state.get("artifact_hashes", {}))
    tokenizer_path = Path(config.dataset.tokenizer_path)
    if not tokenizer_path.is_file():
        msg = f"tokenizer file not found: {tokenizer_path}"
        raise ValueError(msg)
    if expected_hashes.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        msg = "resume tokenizer does not match the checkpoint artifact hash"
        raise ValueError(msg)

    hash_targets = {
        "dataset_input_sha256": Path(config.dataset.input),
        "val_dataset_input_sha256": (
            Path(config.val_dataset.input) if config.val_dataset is not None else Path()
        ),
    }
    for key, path in hash_targets.items():
        if key in expected_hashes:
            if not path.is_file():
                msg = f"SFT dataset input not found: {path}"
                raise ValueError(msg)
            if expected_hashes[key] != sha256_file(path):
                msg = "resume SFT dataset input does not match the checkpoint hash"
                raise ValueError(msg)

    return state


def _train_dataset(config: SFTConfig) -> SFTDataset:
    return SFTDataset(config.dataset)


def _val_dataset(config: SFTConfig) -> SFTDataset:
    dataset_config = config.val_dataset if config.val_dataset is not None else config.dataset
    return SFTDataset(dataset_config)


def sft(config: SFTConfig, config_path: str | Path | None = None) -> TrainResult:
    """Run the SFT phase and return the :class:`TrainResult`.

    Loads weights from ``config.checkpoint`` (or from ``config.resume`` when
    set), trains with masked SFT batches, and writes periodic, best, and final
    checkpoints under ``config.trainer.output_dir``.
    """
    model_cfg = load_config(config.model, ModelConfig)
    _validate_shapes(config)
    context = _checkpoint_context(config, model_cfg, config_path)
    resume_dir = Path(config.resume) if config.resume is not None else None

    if resume_dir is not None:
        state = _validate_resume(resume_dir, config, model_cfg)
        model = load_model(model_cfg, resume_dir)
        optimizer = optim.AdamW(
            learning_rate=config.trainer.lr,
            betas=list(config.trainer.betas),
            weight_decay=config.trainer.weight_decay,
        )
        load_optimizer_state(optimizer, resume_dir)

        train_ds = _train_dataset(config)
        train_iter = train_ds.load_iterator(cast(dict[str, Any], state["dataset_state"]))
        val_ds = _val_dataset(config)

        output_dir = Path(config.trainer.output_dir)
        run_log = output_dir / "run.jsonl"
        if not run_log.exists() and (resume_dir / "run.jsonl").is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resume_dir / "run.jsonl", run_log)

        resume = ResumeState(
            step=int(state["step"]),
            schedule_steps=int(state["schedule_steps"]),
            best_val_loss=cast(float | None, state.get("best_val_loss")),
            last_train_loss=cast(float | None, state.get("last_train_loss")),
            last_val_loss=cast(float | None, state.get("last_val_loss")),
            last_eval_step=cast(int | None, state.get("last_eval_step")),
            optimizer=optimizer,
            checkpoint_dir=resume_dir,
        )
        return train(
            model,
            train_iter,
            val_ds,
            config.trainer,
            resume=resume,
            checkpoint_context=context,
        )

    model = load_model(model_cfg, config.checkpoint)
    train_ds = _train_dataset(config)
    train_iter = train_ds.iterator()
    val_ds = _val_dataset(config)
    schedule_steps = train_ds.estimated_steps() if config.trainer.num_steps <= 0 else None

    return train(
        model,
        train_iter,
        val_ds,
        config.trainer,
        schedule_steps=schedule_steps,
        checkpoint_context=context,
    )
