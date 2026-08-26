"""Pretrain phase (TASK-005.05): wire corpus -> dataset -> trainer.

The integration point that makes pretraining a single command (build-order step
1 of doc-001 section 6). ``pretrain(config)`` loads a randomly-initialized
Kestrel model, builds the corpus (train/val split), wraps each split in a
:class:`~kestrel.data.pretrain_dataset.PretrainDataset`, and hands them to the
shared :func:`~kestrel.train.trainer.train` loop (which saves the final
checkpoint).

``PretrainConfig`` references the model / tokenizer / corpus by path (each
loaded into its own strict config) and embeds a :class:`TrainerConfig`. The
dataset sequence length and batch size are taken from the trainer config so the
pretrain config is the single source of truth for the training shape.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import mlx.optimizers as optim
from pydantic import Field

from kestrel.common.config import BaseConfig, load_config
from kestrel.corpus.builder import build as build_corpus
from kestrel.corpus.config import CorpusConfig
from kestrel.data.pretrain_dataset import PretrainDataset, PretrainDatasetConfig
from kestrel.model.config import ModelConfig
from kestrel.model.io import load as load_model
from kestrel.train.checkpoint import (
    CheckpointContext,
    load_optimizer_state,
    read_checkpoint_state,
    sha256_file,
)
from kestrel.train.trainer import ResumeState, TrainerConfig, TrainResult, train


class PretrainConfig(BaseConfig):
    """Strict settings for the pretrain phase (no unknown keys).

    ``model`` / ``tokenizer`` / ``corpus`` are paths to their respective YAML /
    JSON artifacts. ``total_tokens`` caps the number of training tokens
    (``None`` = run until the corpus is exhausted, i.e. a single pass). The
    train/val data dirs are derived from the corpus ``output_dir``.

    ``resume`` is a full checkpoint directory (``step_<n>``, ``best``, or
    ``final``) from which to continue training.
    """

    model: str
    tokenizer: str
    corpus: str
    total_tokens: int | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    resume: str | None = None


def _dataset(
    config: PretrainConfig,
    corpus_cfg: CorpusConfig,
    split: str,
    total_tokens: int | None,
) -> PretrainDataset:
    """Build a :class:`PretrainDataset` for one split (``train`` or ``val``)."""
    return PretrainDataset(
        PretrainDatasetConfig(
            input=str(Path(corpus_cfg.output_dir) / split),
            tokenizer_path=config.tokenizer,
            context_length=config.trainer.seq_len,
            batch_size=config.trainer.batch_size,
            total_tokens=total_tokens,
            seed=corpus_cfg.seed,
        )
    )


def _pretrain_snapshot(config: PretrainConfig) -> dict[str, Any]:
    """Resolved pretrain config stored in checkpoints (``resume`` normalized)."""
    dump = config.model_dump()
    dump["resume"] = None
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
    }


def _training_relevant_model(config: ModelConfig) -> dict[str, Any]:
    dump = config.model_dump()
    dump.pop("name", None)
    return dump


def _training_relevant_corpus(config: CorpusConfig) -> dict[str, Any]:
    dump = config.model_dump()
    dump.pop("output_dir", None)
    return dump


def _checkpoint_context(
    config: PretrainConfig,
    model_cfg: ModelConfig,
    corpus_cfg: CorpusConfig,
    config_path: str | Path | None,
) -> CheckpointContext:
    raw_configs: dict[str, str] = {}
    if config_path is not None:
        raw_configs["pretrain.yaml"] = Path(config_path).read_text(encoding="utf-8")
    raw_configs["model.yaml"] = Path(config.model).read_text(encoding="utf-8")
    raw_configs["corpus.yaml"] = Path(config.corpus).read_text(encoding="utf-8")

    artifact_hashes: dict[str, str] = {}
    tokenizer_path = Path(config.tokenizer)
    if tokenizer_path.is_file():
        artifact_hashes["tokenizer_sha256"] = sha256_file(tokenizer_path)
    if config_path is not None:
        artifact_hashes["pretrain_config_sha256"] = sha256_file(config_path)
    artifact_hashes["model_config_sha256"] = sha256_file(config.model)
    artifact_hashes["corpus_config_sha256"] = sha256_file(config.corpus)

    for split in ("train", "val"):
        manifest = Path(corpus_cfg.output_dir) / split / "manifest.json"
        if manifest.is_file():
            artifact_hashes[f"corpus_{split}_manifest_sha256"] = sha256_file(manifest)

    return CheckpointContext(
        raw_configs=raw_configs,
        resolved_configs={
            "pretrain": _pretrain_snapshot(config),
            "model": model_cfg.model_dump(),
            "corpus": corpus_cfg.model_dump(),
            "trainer": config.trainer.model_dump(),
        },
        artifact_hashes=artifact_hashes,
        extra_state={
            "total_tokens": config.total_tokens,
            "seed": corpus_cfg.seed,
            "corpus_output_dir": corpus_cfg.output_dir,
        },
    )


def _require_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"checkpoint is missing resolved config snapshot: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _validate_resume(
    checkpoint_dir: Path,
    config: PretrainConfig,
    model_cfg: ModelConfig,
    corpus_cfg: CorpusConfig,
) -> dict[str, Any]:
    state = read_checkpoint_state(checkpoint_dir)
    if state.get("dataset_state") is None:
        msg = f"checkpoint has no resumable dataset state: {checkpoint_dir}"
        raise ValueError(msg)

    resolved_dir = checkpoint_dir / "config" / "resolved"
    pretrain_snap = _require_json(resolved_dir / "pretrain.json")
    model_snap = _require_json(resolved_dir / "model.json")
    corpus_snap = _require_json(resolved_dir / "corpus.json")
    trainer_snap = _require_json(resolved_dir / "trainer.json")

    model_snap.pop("name", None)
    corpus_snap.pop("output_dir", None)
    trainer_keys = tuple(_training_relevant_trainer(TrainerConfig()).keys())
    trainer_selected = {key: trainer_snap.get(key) for key in trainer_keys}

    if config.total_tokens != pretrain_snap.get("total_tokens"):
        msg = "resume total_tokens does not match the checkpoint snapshot"
        raise ValueError(msg)
    if _training_relevant_model(model_cfg) != model_snap:
        msg = "resume model config does not match the checkpoint snapshot"
        raise ValueError(msg)
    if _training_relevant_corpus(corpus_cfg) != corpus_snap:
        msg = "resume corpus config does not match the checkpoint snapshot"
        raise ValueError(msg)
    if _training_relevant_trainer(config.trainer) != trainer_selected:
        msg = "resume trainer config does not match the checkpoint snapshot"
        raise ValueError(msg)

    expected_hashes = cast(dict[str, str], state.get("artifact_hashes", {}))
    tokenizer_path = Path(config.tokenizer)
    if not tokenizer_path.is_file():
        msg = f"tokenizer file not found: {tokenizer_path}"
        raise ValueError(msg)
    if expected_hashes.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        msg = "resume tokenizer does not match the checkpoint artifact hash"
        raise ValueError(msg)

    for split in ("train", "val"):
        key = f"corpus_{split}_manifest_sha256"
        manifest = Path(corpus_cfg.output_dir) / split / "manifest.json"
        if key in expected_hashes:
            if not manifest.is_file():
                msg = f"corpus manifest not found: {manifest}"
                raise ValueError(msg)
            if expected_hashes[key] != sha256_file(manifest):
                msg = f"resume {split} corpus manifest does not match the checkpoint hash"
                raise ValueError(msg)

    return state


def pretrain(config: PretrainConfig, config_path: str | Path | None = None) -> TrainResult:
    """Run the pretrain phase and return the :class:`TrainResult`.

    Loads a random-init model (or a checkpoint when ``config.resume`` is set),
    builds the corpus, and trains on the train split with in-loop validation on
    the val split. The trainer writes periodic, best, and final checkpoints
    under ``config.trainer.output_dir``.
    """
    model_cfg = load_config(config.model, ModelConfig)
    corpus_cfg = load_config(config.corpus, CorpusConfig)
    build_corpus(corpus_cfg)

    context = _checkpoint_context(config, model_cfg, corpus_cfg, config_path)
    resume_dir = Path(config.resume) if config.resume is not None else None

    if resume_dir is not None:
        state = _validate_resume(resume_dir, config, model_cfg, corpus_cfg)
        model = load_model(model_cfg, resume_dir)
        optimizer = optim.AdamW(
            learning_rate=config.trainer.lr,
            betas=list(config.trainer.betas),
            weight_decay=config.trainer.weight_decay,
        )
        load_optimizer_state(optimizer, resume_dir)

        train_ds = _dataset(config, corpus_cfg, "train", config.total_tokens)
        train_iter = train_ds.load_iterator(cast(dict[str, Any], state["dataset_state"]))
        val_ds = _dataset(config, corpus_cfg, "val", None)

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

    model = load_model(model_cfg)
    train_ds = _dataset(config, corpus_cfg, "train", config.total_tokens)
    train_iter = train_ds.iterator()
    val_ds = _dataset(config, corpus_cfg, "val", None)
    schedule_steps = train_ds.estimated_steps() if config.trainer.num_steps <= 0 else None

    return train(
        model,
        train_iter,
        val_ds,
        config.trainer,
        schedule_steps=schedule_steps,
        checkpoint_context=context,
    )
