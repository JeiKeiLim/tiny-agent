"""Example-based SFT dataset for M2 short-context fine-tuning.

Reads a unified SFT JSONL file, validates each row with the strict SFT schema,
renders it with the M2 chat template, and yields padded ``(input, target,
loss_mask)`` int32 batches of shape ``(batch_size, context_length)``.

Rows longer than ``context_length`` are filtered out rather than truncated so
that chat markers and assistant payloads are never cut in the middle. Padding
uses token id ``0`` and is always loss-masked. The final token of each real
example is also loss-masked because its shifted target is either padding or a
repeated boundary token.

``SFTDataset`` exposes a resumable ``SFTDatasetIterator`` whose state is the
current epoch and batch index. The selected example order is deterministic from
the dataset config, so the iterator can be reconstructed from a checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from pydantic import Field, ValidationError
from tokenizers import Tokenizer

from kestrel.common.config import BaseConfig
from kestrel.data.sft_chat import render_sft
from kestrel.data.sft_schema import SFTRow

_PAD_TOKEN_ID = 0


class SFTDatasetConfig(BaseConfig):
    """Strict settings for the M2 SFT dataset."""

    input: str
    tokenizer_path: str
    context_length: int = Field(default=1024, gt=1)
    batch_size: int = Field(default=8, gt=0)
    seed: int = 0
    max_examples: int | None = Field(default=None, ge=1)
    preserve_source_ratios: bool = True
    epochs: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class _Example:
    """One rendered and context-checked SFT example."""

    source: str
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]


def _load_examples(path: Path, tokenizer: Tokenizer, context_length: int) -> list[_Example]:
    examples: list[_Example] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"line {line_number}: invalid JSON: {exc}"
                raise ValueError(msg) from exc
            try:
                row = SFTRow.model_validate(payload)
            except ValidationError as exc:
                msg = f"line {line_number}: invalid SFT row: {exc}"
                raise ValueError(msg) from exc

            rendered = render_sft(row, tokenizer)
            if len(rendered.token_ids) < 2 or len(rendered.token_ids) > context_length:
                continue
            examples.append(
                _Example(
                    source=row.source,
                    token_ids=rendered.token_ids,
                    loss_mask=rendered.loss_mask,
                )
            )
    return examples


def _allocate_source_counts(groups: dict[str, list[_Example]], max_examples: int) -> dict[str, int]:
    """Allocate ``max_examples`` across sources using largest-remainder rounding."""
    total = sum(len(items) for items in groups.values())
    if max_examples >= total:
        return {source: len(items) for source, items in groups.items()}

    quotas = {source: len(items) * max_examples / total for source, items in groups.items()}
    allocated = {source: math.floor(quotas[source]) for source in groups}
    remaining = max_examples - sum(allocated.values())

    order = sorted(
        groups,
        key=lambda source: (-(quotas[source] - allocated[source]), source),
    )
    for source in order:
        if remaining <= 0:
            break
        if allocated[source] < len(groups[source]):
            allocated[source] += 1
            remaining -= 1

    return allocated


def _select_examples(
    examples: list[_Example],
    max_examples: int | None,
    preserve_source_ratios: bool,
    seed: int,
) -> list[_Example]:
    if max_examples is None or max_examples >= len(examples):
        selected = list(examples)
    elif not preserve_source_ratios:
        indices = list(range(len(examples)))
        random.Random(seed).shuffle(indices)
        selected = [examples[index] for index in indices[:max_examples]]
    else:
        groups: dict[str, list[_Example]] = {}
        for example in examples:
            groups.setdefault(example.source, []).append(example)
        allocated = _allocate_source_counts(groups, max_examples)
        rng = random.Random(seed)
        selected = []
        for source, items in groups.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            selected.extend(shuffled[: allocated[source]])
        rng.shuffle(selected)

    return selected


def _pad_example(example: _Example, context_length: int) -> tuple[list[int], list[int]]:
    token_ids = list(example.token_ids)
    loss_mask = list(example.loss_mask)
    real_length = len(token_ids)
    loss_mask[real_length - 1] = 0

    padding = context_length - real_length
    token_ids.extend([_PAD_TOKEN_ID] * padding)
    loss_mask.extend([0] * padding)
    return token_ids, loss_mask


class SFTDataset:
    """Iterable that yields ``(input, target, loss_mask)`` int32 batches."""

    def __init__(self, config: SFTDatasetConfig) -> None:
        self.config = config
        input_path = Path(config.input)
        if not input_path.is_file():
            msg = f"SFT input path not found: {config.input}"
            raise FileNotFoundError(msg)

        tokenizer = Tokenizer.from_file(config.tokenizer_path)
        examples = _load_examples(input_path, tokenizer, config.context_length)
        self._examples = _select_examples(
            examples,
            config.max_examples,
            config.preserve_source_ratios,
            config.seed,
        )
        self._source_counts: dict[str, int] = {}
        for example in self._examples:
            self._source_counts[example.source] = self._source_counts.get(example.source, 0) + 1

    @property
    def source_counts(self) -> dict[str, int]:
        """Number of selected examples per source tag."""
        return dict(self._source_counts)

    def estimated_steps(self) -> int:
        """Full training steps available from the selected examples."""
        return self.config.epochs * (len(self._examples) // self.config.batch_size)

    def iterator(self) -> SFTDatasetIterator:
        """Return a fresh resumable iterator over the dataset."""
        return SFTDatasetIterator(self)

    def load_iterator(self, state: dict[str, Any]) -> SFTDatasetIterator:
        """Return an iterator restored from a :meth:`SFTDatasetIterator.state_dict`."""
        iterator = self.iterator()
        iterator.load_state_dict(state)
        return iterator

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        return self.iterator()


class SFTDatasetIterator:
    """Resumable iterator that yields ``(input, target, loss_mask)`` int32 batches."""

    def __init__(self, dataset: SFTDataset) -> None:
        self._dataset = dataset
        self._config = dataset.config
        self._batches_per_epoch = len(dataset._examples) // self._config.batch_size
        self._epoch = 0
        self._batch_index = 0
        self._closed = False

    def __iter__(self) -> SFTDatasetIterator:
        return self

    def close(self) -> None:
        self._closed = True

    def __next__(self) -> tuple[mx.array, mx.array, mx.array]:
        if self._closed:
            raise StopIteration
        if self._batch_index >= self._batches_per_epoch:
            if self._epoch + 1 >= self._config.epochs:
                self.close()
                raise StopIteration
            self._epoch += 1
            self._batch_index = 0
            if self._batch_index >= self._batches_per_epoch:
                self.close()
                raise StopIteration

        batch = self._make_batch(self._batch_index)
        self._batch_index += 1
        return batch

    def _make_batch(self, batch_index: int) -> tuple[mx.array, mx.array, mx.array]:
        batch_size = self._config.batch_size
        context_length = self._config.context_length
        start = batch_index * batch_size
        examples = self._dataset._examples[start : start + batch_size]

        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        masks: list[list[int]] = []
        for example in examples:
            token_ids, loss_mask = _pad_example(example, context_length)
            inputs.append(token_ids)
            targets.append([*token_ids[1:], token_ids[-1]])
            masks.append(loss_mask)
        return (
            mx.array(inputs, dtype=mx.int32),
            mx.array(targets, dtype=mx.int32),
            mx.array(masks, dtype=mx.int32),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the iterator position."""
        return {
            "format_version": 1,
            "config": self._config.model_dump(),
            "epoch": self._epoch,
            "batch_index": self._batch_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator position from a :meth:`state_dict` snapshot."""
        if state.get("format_version") != 1:
            msg = f"unsupported dataset state format_version: {state.get('format_version')!r}"
            raise ValueError(msg)
        if state.get("config") != self._config.model_dump():
            msg = "dataset state was created with a different SFTDatasetConfig"
            raise ValueError(msg)

        epoch = int(state["epoch"])
        batch_index = int(state["batch_index"])
        if epoch < 0 or epoch >= self._config.epochs:
            msg = f"invalid dataset state epoch {epoch} for {self._config.epochs} epochs"
            raise ValueError(msg)
        if batch_index < 0 or batch_index > self._batches_per_epoch:
            msg = f"invalid dataset state batch_index {batch_index}"
            raise ValueError(msg)
        if self._batches_per_epoch == 0 and batch_index != 0:
            msg = "dataset state batch_index must be 0 when no full batches exist"
            raise ValueError(msg)

        self.close()
        self._closed = False
        self._epoch = epoch
        self._batch_index = batch_index
