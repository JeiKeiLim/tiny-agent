"""Tests for the tokenizer training config model.

The model + its validators are the code under test. Real config *values* are
tunable data, so tests use synthetic fixtures and only check that the real
config is well-formed and loadable.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.tokenizer.config import TokenizerConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

VALID = "vocab_size: 1000\nspecial_tokens:\n  - im_start\n  - im_end\neos_token: im_end\n"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text(body)
    return p


def test_valid_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, VALID), TokenizerConfig)
    assert cfg.vocab_size == 1000
    assert cfg.eos_token == "im_end"
    assert cfg.special_tokens == ["im_start", "im_end"]


def test_defaults_apply(tmp_path):
    cfg = load_config(_write(tmp_path, "special_tokens: [a, b]\neos_token: b\n"), TokenizerConfig)
    assert cfg.vocab_size == 16384
    assert cfg.train_dir == "data/tokenizer_train"
    assert cfg.output_dir == "checkpoints/tokenizer"
    assert cfg.min_frequency == 2


def test_rejects_unknown_key(tmp_path):
    p = _write(tmp_path, VALID + "bogus: 1\n")
    with pytest.raises(ValidationError):
        load_config(p, TokenizerConfig)


def test_rejects_mistyped_scalar(tmp_path):
    p = _write(tmp_path, VALID.replace("vocab_size: 1000", 'vocab_size: "1000"'))
    with pytest.raises(ValidationError):
        load_config(p, TokenizerConfig)


def test_rejects_vocab_too_small(tmp_path):
    # 256 byte tokens + 2 special tokens do not fit in 257.
    p = _write(tmp_path, VALID.replace("vocab_size: 1000", "vocab_size: 257"))
    with pytest.raises(ValidationError):
        load_config(p, TokenizerConfig)


def test_rejects_eos_not_in_special_tokens(tmp_path):
    p = _write(tmp_path, VALID.replace("eos_token: im_end", "eos_token: nope"))
    with pytest.raises(ValidationError):
        load_config(p, TokenizerConfig)


def test_rejects_duplicate_special_tokens(tmp_path):
    p = _write(tmp_path, VALID.replace("  - im_end\n", "  - im_end\n  - im_end\n"))
    with pytest.raises(ValidationError):
        load_config(p, TokenizerConfig)


def test_real_tokenizer_config_loads():
    cfg = load_config(CONFIGS / "tokenizer" / "train.yaml", TokenizerConfig)
    assert isinstance(cfg, TokenizerConfig)
