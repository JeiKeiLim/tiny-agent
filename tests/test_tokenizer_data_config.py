"""Tests for the tokenizer training-data config model.

The model + its validator are the code under test. Real config *values* are
tunable data, so tests use synthetic fixtures and only check that the real
config is well-formed and loadable.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.data.tokenizer_data_config import TokenizerTrainDataConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "train_data.yaml"
    p.write_text(body)
    return p


def test_valid_config_loads(tmp_path):
    p = _write(
        tmp_path,
        "total_bytes: 1000\n"
        "sources:\n"
        "  web:\n"
        "    dataset: some/ds\n"
        "    text_field: text\n"
        "    fraction: 0.7\n"
        "  code:\n"
        "    dataset: some/code\n"
        "    fraction: 0.3\n",
    )
    cfg = load_config(p, TokenizerTrainDataConfig)
    assert cfg.total_bytes == 1000
    assert set(cfg.sources) == {"web", "code"}
    assert cfg.sources["web"].text_field == "text"
    assert cfg.sources["code"].text_field is None


def test_fractions_must_sum_to_one(tmp_path):
    p = _write(
        tmp_path,
        "total_bytes: 1000\n"
        "sources:\n"
        "  web:\n"
        "    dataset: some/ds\n"
        "    fraction: 0.5\n"
        "  code:\n"
        "    dataset: some/code\n"
        "    fraction: 0.3\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, TokenizerTrainDataConfig)


def test_rejects_unknown_key(tmp_path):
    p = _write(
        tmp_path,
        "total_bytes: 1000\n"
        "sources:\n"
        "  web:\n"
        "    dataset: some/ds\n"
        "    fraction: 1.0\n"
        "bogus: 1\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, TokenizerTrainDataConfig)


def test_rejects_mistyped_scalar(tmp_path):
    p = _write(
        tmp_path,
        'total_bytes: "1000"\n'
        "sources:\n"
        "  web:\n"
        "    dataset: some/ds\n"
        "    fraction: 1.0\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, TokenizerTrainDataConfig)


def test_real_tokenizer_data_config_loads():
    cfg = load_config(CONFIGS / "tokenizer" / "train_data.yaml", TokenizerTrainDataConfig)
    assert isinstance(cfg, TokenizerTrainDataConfig)
