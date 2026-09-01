"""Read-only pretrain checkpoint evaluation (TASK-005.13).

Evaluates a saved Kestrel checkpoint on a fixed corpus split using the same
tokenizer, sequence length, batch size, and document-aware batching as
training. Unlike the trainer's in-loop validation estimate, this evaluates a
configurable token budget (or the full split) and reports token-weighted
loss, perplexity, bits/token, and per-domain diagnostics.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.nn as nn
from mlx.nn.losses import cross_entropy
from tokenizers import Tokenizer

from kestrel.common.config import load_config
from kestrel.corpus.config import CorpusConfig
from kestrel.data.pretrain_dataset import PretrainDataset, PretrainDatasetConfig
from kestrel.model.config import ModelConfig
from kestrel.model.generate import generate
from kestrel.model.io import load as load_model
from kestrel.train.pretrain import PretrainConfig

DEFAULT_MAX_TOKENS = 100_000
DEFAULT_PROGRESS_EVERY_TOKENS = 100_000
GENERATION_PROMPTS = ("Hello", "The capital of France is", "def add(a, b):")
GENERATION_MAX_TOKENS = 32


@dataclass
class LossAccumulator:
    """Token-weighted cross-entropy accumulator."""

    loss_sum: float = 0.0
    tokens: int = 0
    batches: int = 0

    def add(self, loss_sum: float, tokens: int) -> None:
        if tokens < 0:
            msg = f"token count must be non-negative, got {tokens}"
            raise ValueError(msg)
        self.loss_sum += loss_sum
        self.tokens += tokens
        self.batches += 1

    @property
    def loss(self) -> float:
        if self.tokens == 0:
            msg = "no tokens have been accumulated"
            raise ValueError(msg)
        return self.loss_sum / self.tokens


@dataclass(frozen=True)
class EvalMetrics:
    """Token-weighted evaluation metrics for one data stream."""

    loss: float
    tokens: int
    batches: int

    @property
    def perplexity(self) -> float:
        return math.exp(self.loss)

    @property
    def bits_per_token(self) -> float:
        return self.loss / math.log(2.0)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
            "tokens": self.tokens,
            "batches": self.batches,
        }


@dataclass(frozen=True)
class PretrainEvalResult:
    """Checkpoint evaluation result: mixed split, per-domain splits, samples."""

    checkpoint: Path
    split: str
    mixed: EvalMetrics
    domains: dict[str, EvalMetrics]
    samples: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "split": self.split,
            "mixed": self.mixed.to_dict(),
            "domains": {name: metrics.to_dict() for name, metrics in self.domains.items()},
            "samples": list(self.samples) if self.samples is not None else None,
        }


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _load_manifest(split_dir: Path) -> dict[str, Any] | None:
    path = split_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _manifest_total_tokens(manifest: dict[str, Any] | None) -> int | None:
    if manifest is None:
        return None
    total = _as_int(manifest.get("total_token_count"))
    if total is None:
        total = _as_int(manifest.get("total_estimated_token_count"))
    if total is None:
        return None
    docs = _as_int(manifest.get("total_doc_count")) or 0
    return total + 2 * docs


def _manifest_domain_tokens(
    manifest: dict[str, Any] | None, domain: str, extension: str
) -> int | None:
    if manifest is None:
        return None
    files = manifest.get("files")
    if not isinstance(files, list):
        return None
    for raw in files:
        if not isinstance(raw, dict):
            continue
        if raw.get("domain") != domain and raw.get("path") != f"{domain}.{extension}":
            continue
        total = _as_int(raw.get("token_count"))
        if total is None:
            total = _as_int(raw.get("estimated_token_count"))
        if total is None:
            return None
        docs = _as_int(raw.get("doc_count")) or 0
        return total + 2 * docs
    return None


def _format_progress(acc: LossAccumulator, expected_tokens: int | None) -> str:
    if expected_tokens is None or expected_tokens <= 0:
        return f"{acc.tokens:,} tokens"
    percentage = min(100.0, 100.0 * acc.tokens / expected_tokens)
    return f"{acc.tokens:,}/~{expected_tokens:,} tokens ({percentage:.1f}%)"


def _evaluate_dataset(
    model: nn.Module,
    dataset_config: PretrainDatasetConfig,
    *,
    label: str | None = None,
    progress_every_tokens: int = 0,
    total_token_estimate: int | None = None,
) -> EvalMetrics:
    dataset = PretrainDataset(dataset_config)
    iterator = dataset.iterator()
    acc = LossAccumulator()
    next_progress = progress_every_tokens
    expected_tokens = total_token_estimate
    if dataset_config.total_tokens is not None and expected_tokens is not None:
        expected_tokens = min(expected_tokens, dataset_config.total_tokens)
    try:
        for x, target, doc_ids in iterator:
            logits = model(x, doc_ids)
            loss_sum = cross_entropy(logits[:, :-1], target[:, :-1], reduction="sum")
            tokens = x.shape[0] * max(x.shape[1] - 1, 0)
            acc.add(cast(float, loss_sum.item()), tokens)
            if label is not None and progress_every_tokens > 0 and acc.tokens >= next_progress:
                print(
                    f"[eval] {label}: {_format_progress(acc, expected_tokens)}, "
                    f"loss={acc.loss:.6f}, batches={acc.batches}",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress += progress_every_tokens
    finally:
        iterator.close()

    if acc.tokens == 0:
        msg = f"no tokens evaluated from input {dataset_config.input!r}"
        raise ValueError(msg)
    return EvalMetrics(loss=acc.loss, tokens=acc.tokens, batches=acc.batches)


def _dataset_config(
    pretrain_config: PretrainConfig,
    corpus_config: CorpusConfig,
    input_path: Path,
    max_tokens: int | None,
) -> PretrainDatasetConfig:
    return PretrainDatasetConfig(
        input=str(input_path),
        tokenizer_path=pretrain_config.tokenizer,
        context_length=pretrain_config.trainer.seq_len,
        batch_size=pretrain_config.trainer.batch_size,
        total_tokens=max_tokens,
        seed=corpus_config.seed,
    )


def _generate_samples(model: nn.Module, tokenizer_path: str) -> tuple[str, ...]:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tuple(
        generate(
            model,
            tokenizer,
            prompt,
            max_tokens=GENERATION_MAX_TOKENS,
            temp=0.0,
            skip_special_tokens=True,
        )
        for prompt in GENERATION_PROMPTS
    )


def evaluate_checkpoint(
    *,
    pretrain_config: PretrainConfig,
    checkpoint: str | Path,
    split: str = "val",
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    generate_samples: bool = False,
    progress_every_tokens: int = DEFAULT_PROGRESS_EVERY_TOKENS,
) -> PretrainEvalResult:
    """Evaluate a checkpoint on a corpus split without modifying any artifacts.

    ``max_tokens`` caps the number of dataset tokens evaluated for the mixed
    split and for each per-domain file. ``max_tokens=None`` evaluates each
    stream until exhaustion. Progress is written to stderr every
    ``progress_every_tokens`` tokens; ``0`` disables progress output. When a
    corpus manifest is available, progress includes an estimated percentage.
    """
    checkpoint_path = Path(checkpoint)
    weights_path = checkpoint_path / "weights.npz"
    if not weights_path.is_file():
        msg = f"checkpoint weights not found: {weights_path}"
        raise FileNotFoundError(msg)

    model_config = load_config(pretrain_config.model, ModelConfig)
    corpus_config = load_config(pretrain_config.corpus, CorpusConfig)
    model = load_model(model_config, checkpoint_path)

    split_dir = Path(corpus_config.output_dir) / split
    if not split_dir.is_dir():
        msg = f"corpus split directory not found: {split_dir}"
        raise FileNotFoundError(msg)

    manifest = _load_manifest(split_dir)
    mixed = _evaluate_dataset(
        model,
        _dataset_config(pretrain_config, corpus_config, split_dir, max_tokens),
        label="mixed",
        progress_every_tokens=progress_every_tokens,
        total_token_estimate=_manifest_total_tokens(manifest),
    )

    extension = "jsonl" if corpus_config.output_format == "jsonl" else "txt"
    domains: dict[str, EvalMetrics] = {}
    for component in corpus_config.components:
        domain_path = split_dir / f"{component.name}.{extension}"
        if domain_path.is_file():
            domains[component.name] = _evaluate_dataset(
                model,
                _dataset_config(pretrain_config, corpus_config, domain_path, max_tokens),
                label=f"domain:{component.name}",
                progress_every_tokens=progress_every_tokens,
                total_token_estimate=_manifest_domain_tokens(manifest, component.name, extension),
            )

    samples = _generate_samples(model, pretrain_config.tokenizer) if generate_samples else None
    return PretrainEvalResult(
        checkpoint=checkpoint_path,
        split=split,
        mixed=mixed,
        domains=domains,
        samples=samples,
    )
