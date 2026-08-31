"""Example-based SFT dataset for M2 short-context fine-tuning.

Reads a unified SFT JSONL file, validates each row with the strict SFT schema,
renders it with the M2 chat template, and yields padded ``(input, target,
loss_mask)`` int32 batches of shape ``(batch_size, context_length)``.

Rows longer than ``context_length`` are filtered out rather than truncated so
that chat markers and assistant payloads are never cut in the middle. Padding
uses token id ``0`` and is always loss-masked. The final token of each real
example is also loss-masked because its shifted target is either padding or a
repeated boundary token.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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
        return len(self._examples) // self.config.batch_size

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        batch_size = self.config.batch_size
        context_length = self.config.context_length
        full_examples = len(self._examples) - len(self._examples) % batch_size

        for start in range(0, full_examples, batch_size):
            inputs: list[list[int]] = []
            targets: list[list[int]] = []
            masks: list[list[int]] = []
            for example in self._examples[start : start + batch_size]:
                token_ids, loss_mask = _pad_example(example, context_length)
                inputs.append(token_ids)
                targets.append([*token_ids[1:], token_ids[-1]])
                masks.append(loss_mask)
            yield (
                mx.array(inputs, dtype=mx.int32),
                mx.array(targets, dtype=mx.int32),
                mx.array(masks, dtype=mx.int32),
            )
