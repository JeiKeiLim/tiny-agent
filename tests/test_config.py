"""Tests for the YAML -> dataclass config loader.

The loader is the code under test. Config *values* are tunable data, not
invariants, so these tests never pin real config values — the real model
configs are only checked for being well-formed and loadable.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

from kestrel.common.config import ConfigError, build_config, load_config
from kestrel.model.config import ModelConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --- Loader mechanism (synthetic fixtures, decoupled from real config values) ---


@dataclass
class Inner:
    a: int = 1
    b: str = "x"


@dataclass
class Outer:
    name: str
    count: int = 0
    rate: float = 0.5
    tags: List[str] = field(default_factory=list)
    inner: Optional[Inner] = None
    flag: bool = False


def test_loader_maps_scalars_list_and_nested(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: demo\ncount: 7\nrate: 1.5\ntags: [a, b, c]\n"
        "inner: {a: 2, b: hello}\nflag: true\n"
    )
    cfg = load_config(p, Outer)
    assert cfg.name == "demo"
    assert cfg.count == 7
    assert cfg.rate == 1.5
    assert cfg.tags == ["a", "b", "c"]
    assert cfg.inner == Inner(a=2, b="hello")
    assert cfg.flag is True


def test_loader_applies_defaults_for_missing_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: minimal\n")
    cfg = load_config(p, Outer)
    assert cfg.name == "minimal"
    assert cfg.count == 0
    assert cfg.rate == 0.5
    assert cfg.tags == []
    assert cfg.inner is None
    assert cfg.flag is False


def test_loader_rejects_unknown_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nnonsense: 1\n")
    with pytest.raises(ConfigError):
        load_config(p, Outer)


def test_build_config_requires_a_dataclass():
    with pytest.raises(ConfigError):
        build_config({"a": 1}, int)


# --- Real config files: well-formed + loadable (values intentionally NOT pinned) ---


def test_real_model_configs_load():
    for size in ("50m", "150m"):
        cfg = load_config(CONFIGS / "kestrel" / size / "model.yaml", ModelConfig)
        assert isinstance(cfg, ModelConfig)
