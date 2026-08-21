"""Generic YAML -> typed-dataclass config loading.

Every tunable in Kestrel lives in a YAML file loaded into a typed dataclass.
This module maps a YAML mapping onto a (possibly nested) dataclass, recursing
into nested dataclasses and handling basic scalar / list types. Missing keys
fall back to the dataclass defaults; unknown keys are rejected.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a YAML document cannot be mapped onto the target dataclass."""


def _inner(hint: Any) -> Any:
    """Strip Optional[X] down to X; otherwise return the hint unchanged."""
    if get_origin(hint) is not None:
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _convert(value: Any, hint: Any) -> Any:
    """Convert a raw YAML value to the target type ``hint``."""
    if value is None:
        return None
    hint = _inner(hint)
    if is_dataclass(hint) and isinstance(value, dict):
        return build_config(value, hint)
    if get_origin(hint) is list:
        (item_hint,) = get_args(hint)
        return [_convert(v, item_hint) for v in value]
    return value


def build_config(data: dict, config_type: type) -> Any:
    """Map a dict onto dataclass ``config_type``, recursing into nested dataclasses."""
    if not is_dataclass(config_type):
        raise ConfigError(f"{config_type!r} is not a dataclass")
    if not isinstance(data, dict):
        raise ConfigError(
            f"expected a mapping for {config_type.__name__}, got {type(data).__name__}"
        )
    hints = get_type_hints(config_type)
    known = {f.name for f in fields(config_type)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} for {config_type.__name__}")
    kwargs = {key: _convert(value, hints[key]) for key, value in data.items()}
    return config_type(**kwargs)


def load_config(path: str | Path, config_type: type) -> Any:
    """Load the YAML file at ``path`` into dataclass ``config_type``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return build_config(data, config_type)
