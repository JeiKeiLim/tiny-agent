"""Kestrel model configuration (Pydantic model loaded from YAML)."""

from __future__ import annotations

from kestrel.common.config import BaseConfig


class ModelConfig(BaseConfig):
    """Shape of a Kestrel decoder-only transformer (plan §9).

    Defaults match Kestrel-50M; the 150M shape is supplied via its YAML config.
    Strict mode rejects mistyped values (e.g. ``n_layers: "15"``).
    """

    name: str
    vocab_size: int = 16384
    context_length: int = 2048
    n_layers: int = 15
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_size: int = 512
    intermediate_size: int = 1408
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0
