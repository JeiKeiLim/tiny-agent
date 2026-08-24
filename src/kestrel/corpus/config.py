"""Pretraining corpus configuration (Pydantic models loaded from YAML).

The corpus is a weighted mix of text sources. Each component names a source
(``hf`` = stream from a HuggingFace dataset, or ``local`` = read an existing
file) and the fraction of the total byte budget it should contribute. Component
fractions must sum to 1.0.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from kestrel.common.config import BaseConfig


class HfSourceConfig(BaseConfig):
    """Stream text from a HuggingFace Hub dataset (via the ``datasets`` lib)."""

    type: Literal["hf"]
    dataset: str
    config: str | None = None
    text_field: str | None = None


class LocalSourceConfig(BaseConfig):
    """Read text from an existing file on disk."""

    type: Literal["local"]
    path: str


SourceConfig = Annotated[
    HfSourceConfig | LocalSourceConfig,
    Field(discriminator="type"),
]


class ComponentConfig(BaseConfig):
    """One named source contributing ``fraction`` of the corpus byte budget."""

    name: str
    source: SourceConfig
    fraction: float = Field(gt=0.0, le=1.0)


class CorpusConfig(BaseConfig):
    """How to assemble the pretraining corpus (plan §8).

    Each component is read/streamed up to ``total_bytes * fraction`` (or until
    the source is exhausted). Component fractions must sum to 1.0.

    The assembled corpus is split into train/val(/test) by a deterministic,
    order-independent per-line hash (seeded by ``seed``): each line is routed to
    ``test`` if its hash falls in ``[0, test_fraction)``, else ``val`` if in
    ``[test_fraction, test_fraction + val_fraction)``, else ``train``. This gives
    a reproducible held-out validation (and optional test) slice of the same
    distribution, with no document leaking across splits.
    """

    total_bytes: int = Field(gt=0)
    seed: int = 0
    output_dir: str = "data/corpus"
    val_fraction: float = Field(ge=0.0, le=1.0, default=0.1)
    test_fraction: float = Field(ge=0.0, le=1.0, default=0.0)
    components: list[ComponentConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        total = sum(c.fraction for c in self.components)
        if abs(total - 1.0) > 1e-6:
            msg = f"component fractions must sum to 1.0, got {total}"
            raise ValueError(msg)
        if self.val_fraction + self.test_fraction > 1.0 + 1e-6:
            msg = (
                f"val_fraction + test_fraction must be <= 1.0, "
                f"got {self.val_fraction + self.test_fraction}"
            )
            raise ValueError(msg)
        return self
