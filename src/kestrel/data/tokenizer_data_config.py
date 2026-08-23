"""Tokenizer training-data configuration (Pydantic model loaded from YAML)."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from kestrel.common.config import BaseConfig


class SourceConfig(BaseConfig):
    """One data source contributing to the tokenizer training sample.

    ``text_field`` names the column to read as text. When ``None``, the whole
    row is serialized to a JSONL line (for structured sources).
    """

    dataset: str
    config: str | None = None
    text_field: str | None = None
    fraction: float = Field(gt=0.0, le=1.0)


class TokenizerTrainDataConfig(BaseConfig):
    """How to assemble the tokenizer training sample (plan §7).

    Each source is streamed up to ``total_bytes * fraction`` (or until the
    source is exhausted). Source fractions must sum to 1.0.
    """

    total_bytes: int = Field(gt=0)
    output_dir: str = "data/tokenizer_train"
    sources: dict[str, SourceConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        total = sum(s.fraction for s in self.sources.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"source fractions must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self
