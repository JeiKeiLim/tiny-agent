"""Integration test: train a tiny BPE tokenizer end-to-end.

Uses a small synthetic corpus and a small vocab so it runs in a fraction of a
second. The real 16k-vocab training is a runtime step on the prepared sample.
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train


def _write_corpus(corpus_dir: Path) -> None:
    (corpus_dir / "web.txt").write_text(
        ("hello world " * 500) + ("the quick brown fox jumps " * 500)
    )
    (corpus_dir / "code.txt").write_text("def foo():\n    return 42\n" * 500)


def test_train_saves_artifact_and_round_trips(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus(corpus)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp_path / "out"),
        special_tokens=["im_start", "im_end"],
        eos_token="im_end",
    )
    out_path = train(config)
    assert out_path.exists()

    tok = Tokenizer.from_file(str(out_path))
    # A tiny corpus yields fewer merges than the requested vocab size, but the
    # artifact never exceeds it and always keeps the special tokens.
    assert len(tok.get_vocab()) <= 400
    assert tok.token_to_id("im_end") is not None

    text = "hello world def foo(): return 42"
    assert tok.decode(tok.encode(text).ids) == text


def test_train_fails_without_corpus(tmp_path):
    config = TokenizerConfig(train_dir=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        train(config)
