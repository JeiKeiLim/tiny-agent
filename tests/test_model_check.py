"""Tests for the model smoke-test CLI's pure logic (scripts/check_model.py).

The CLI is a thin I/O shell; the code under test is ``check_model`` /
``report_from_model``, exercised on a tiny model + tiny trained tokenizer so it
runs fast.
"""

import math
from pathlib import Path

from check_model import ModelReport, check_model, report_from_model
from tokenizers import Tokenizer

from kestrel.model.config import ModelConfig
from kestrel.model.io import load, save
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train

TEXT = "hello world the quick brown fox"


def _tiny_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(("hello world " * 500) + ("the quick brown fox jumps " * 500))
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp_path / "tok"),
        special_tokens=["im_start", "im_end"],
        eos_token="im_end",
    )
    return train(config)


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        name="tiny",
        vocab_size=400,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_size=64,
        intermediate_size=128,
    )


def test_check_model_random_init(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_model_config()
    report = check_model(config, tokenizer_path=tok, text=TEXT)
    assert isinstance(report, ModelReport)
    assert report.param_count > 0
    assert report.logits_shape[0] == 1  # batch
    assert report.logits_shape[1] >= 2  # at least 2 tokens
    assert report.logits_shape[2] == config.vocab_size
    assert math.isfinite(report.loss)
    assert len(report.top_token_ids) == 5
    assert len(report.top_tokens) == 5
    assert all(0 <= i < config.vocab_size for i in report.top_token_ids)


def test_checkpoint_round_trip(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_model_config()
    model = load(config)  # random init
    ckpt = tmp_path / "pretrain" / "tiny"
    save(model, ckpt)
    report_ckpt = check_model(config, checkpoint=ckpt, tokenizer_path=tok, text=TEXT)
    expected = report_from_model(model, Tokenizer.from_file(str(tok)), TEXT, 5)
    assert report_ckpt.logits_shape == expected.logits_shape
    assert report_ckpt.top_token_ids == expected.top_token_ids
    assert math.isclose(report_ckpt.loss, expected.loss)
