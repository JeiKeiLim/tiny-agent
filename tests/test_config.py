"""Tests for the YAML -> Pydantic config loader.

The loader is the code under test. Config *values* are tunable data, not
invariants, so these tests never pin real config values — the real model
configs are only checked for being well-formed and loadable.
"""

from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from kestrel.common.config import BaseConfig, load_config
from kestrel.model.config import ModelConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --- Loader mechanism (synthetic fixtures, decoupled from real config values) ---


class Inner(BaseConfig):
    a: int = 1
    b: str = "x"


class Outer(BaseConfig):
    name: str
    count: int = 0
    rate: float = 0.5
    tags: list[str] = Field(default_factory=list)
    inner: Inner | None = None
    flag: bool = False


def test_loader_maps_scalars_list_and_nested(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: demo\ncount: 7\nrate: 1.5\ntags: [a, b, c]\ninner: {a: 2, b: hello}\nflag: true\n"
    )
    cfg = load_config(p, Outer)
    assert cfg.name == "demo"
    assert cfg.count == 7
    assert cfg.rate == 1.5
    assert cfg.tags == ["a", "b", "c"]
    assert cfg.inner == Inner(a=2, b="hello")
    assert cfg.flag is True


def test_loader_rejects_mistyped_scalar(tmp_path):
    # The original gap: a string where an int is expected must be rejected.
    p = tmp_path / "c.yaml"
    p.write_text('name: demo\ncount: "7"\n')
    with pytest.raises(ValidationError):
        load_config(p, Outer)


def test_loader_rejects_unknown_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nnonsense: 1\n")
    with pytest.raises(ValidationError):
        load_config(p, Outer)


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


# --- Real config files: well-formed + loadable (values intentionally NOT pinned) ---


def test_real_model_configs_load():
    for size in ("50m", "150m"):
        cfg = load_config(CONFIGS / "kestrel" / size / "model.yaml", ModelConfig)
        assert isinstance(cfg, ModelConfig)
