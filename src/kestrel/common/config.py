"""YAML -> Pydantic model config loading.

Every tunable in Kestrel lives in a YAML file loaded into a typed, validated
Pydantic model. All config models subclass :class:`BaseConfig`, which enforces
strict typing (a mistyped value such as ``n_layers: "15"`` is rejected) and
forbids unknown keys (typos are caught).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base for all Kestrel config models: strict typing, no unknown keys."""

    model_config = ConfigDict(strict=True, extra="forbid")


def load_config[T: BaseModel](path: str | Path, config_type: type[T]) -> T:
    """Load the YAML file at ``path`` into Pydantic model ``config_type``.

    Raises:
        pydantic.ValidationError: if the document has wrong types or unknown keys.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_type.model_validate(data)
