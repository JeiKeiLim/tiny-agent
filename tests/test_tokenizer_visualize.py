"""Tests for the tokenizer explorer's pure logic (classification + byte values).

The REPL itself is a thin I/O shell; the code under test is TokenizerView's
classification, byte extraction, and round-trip check, exercised on a tiny
trained tokenizer.
"""

from pathlib import Path

from tokenizers import Tokenizer
from visualize_tokenizer import KIND_BYTE, KIND_MERGED, KIND_SPECIAL, TokenizerView

from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train


def _tiny_view(tmp_path: Path) -> TokenizerView:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(("hello world " * 500) + ("the quick brown fox jumps " * 500))
    (corpus / "code.txt").write_text("def foo():\n    return 42\n" * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp_path / "out"),
        special_tokens=["im_start", "im_end"],
        eos_token="im_end",
    )
    return TokenizerView(Tokenizer.from_file(str(train(config))))


def test_view_classifies_kinds(tmp_path):
    view = _tiny_view(tmp_path)
    infos = view.view("im_start hello world")
    kinds = [i.kind for i in infos]
    assert kinds[0] == KIND_SPECIAL
    assert KIND_MERGED in kinds
    assert all(i.kind in (KIND_SPECIAL, KIND_MERGED, KIND_BYTE) for i in infos)


def test_view_byte_values(tmp_path):
    view = _tiny_view(tmp_path)
    info = view.view("o")[0]
    assert info.kind == KIND_BYTE
    assert info.byte_values == (111,)


def test_view_id_round_trip(tmp_path):
    view = _tiny_view(tmp_path)
    info = view.view_id(view.tokenizer.token_to_id("im_end"))
    assert info is not None
    assert info.kind == KIND_SPECIAL
    assert view.view_id(999_999) is None


def test_roundtrip_ok(tmp_path):
    view = _tiny_view(tmp_path)
    assert view.roundtrip_ok("hello world def foo(): return 42")
