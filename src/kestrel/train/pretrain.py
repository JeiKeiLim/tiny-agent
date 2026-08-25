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

from pathlib import Path

from pydantic import Field

from kestrel.common.config import BaseConfig, load_config
from kestrel.corpus.builder import build as build_corpus
from kestrel.corpus.config import CorpusConfig
from kestrel.data.pretrain_dataset import PretrainDataset, PretrainDatasetConfig
from kestrel.model.config import ModelConfig
from kestrel.model.io import load as load_model
from kestrel.train.trainer import TrainerConfig, TrainResult, train


class PretrainConfig(BaseConfig):
    """Strict settings for the pretrain phase (no unknown keys).

    ``model`` / ``tokenizer`` / ``corpus`` are paths to their respective YAML /
    JSON artifacts. ``total_tokens`` caps the number of training tokens
    (``None`` = run until the corpus is exhausted, i.e. a single pass). The
    train/val data dirs are derived from the corpus ``output_dir``.
    """

    model: str
    tokenizer: str
    corpus: str
    total_tokens: int | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)


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


def pretrain(config: PretrainConfig) -> TrainResult:
    """Run the pretrain phase and return the :class:`TrainResult`.

    Loads a random-init model, builds the corpus, and trains on the train split
    with in-loop validation on the val split. The trainer writes periodic and a
    final checkpoint under ``config.trainer.output_dir``.
    """
    model = load_model(load_config(config.model, ModelConfig))

    corpus_cfg = load_config(config.corpus, CorpusConfig)
    build_corpus(corpus_cfg)

    train_ds = _dataset(config, corpus_cfg, "train", config.total_tokens)
    val_ds = _dataset(config, corpus_cfg, "val", None)

    return train(model, train_ds, val_ds, config.trainer)
