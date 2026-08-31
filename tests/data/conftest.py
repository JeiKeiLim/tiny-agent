"""Shared fixtures for M2 SFT data-preparation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS, TokenizerConfig
from kestrel.tokenizer.train import train

SENTENCE = "hello world the quick brown fox jumps over the lazy dog. "


@pytest.fixture(scope="session")
def tiny_sft_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("sft_prepare_tokenizer")
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "sft.txt").write_text(SENTENCE * 500 + "hello world assistant user tool " * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp / "tok"),
        special_tokens=list(DEFAULT_SPECIAL_TOKENS),
        eos_token=DEFAULT_SPECIAL_TOKENS[1],
    )
    return train(config)


@pytest.fixture(scope="session")
def tiny_sft_tokenizer_obj(tiny_sft_tokenizer: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tiny_sft_tokenizer))
