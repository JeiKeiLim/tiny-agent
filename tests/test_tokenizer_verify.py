"""Tokenizer round-trip + byte-coverage verification.

The verification logic (``verify_bytes`` + ``byte_coverage``) lives here as
the code under test: it round-trips raw bytes through a trained tokenizer and
measures losslessness and raw-byte coverage. Bytes are bridged through latin-1
(a bijection between the 256 byte values and U+0000-U+00FF) so arbitrary bytes
— not just valid UTF-8 — are covered. Tests exercise the logic on tiny
tokenizers trained in-test (self-contained; no trained artifact required).
"""

import random
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train


@dataclass(frozen=True)
class VerifyStats:
    """Byte-level round-trip stats for one input."""

    raw_bytes: int
    token_count: int
    roundtrip_bytes: int
    lossless: bool

    @property
    def diff(self) -> int:
        """raw - round-trip; 0 iff lossless."""
        return self.raw_bytes - self.roundtrip_bytes

    @property
    def bytes_per_token(self) -> float:
        return self.raw_bytes / self.token_count if self.token_count else 0.0

    @property
    def token_id_bytes(self) -> int:
        """Size of the id sequence stored as uint16 (a 16k vocab fits in 14 bits)."""
        return 2 * self.token_count


def verify_bytes(tokenizer: Tokenizer, data: bytes) -> VerifyStats:
    """Round-trip ``data`` through the tokenizer and measure the result.

    The bytes are mapped to text with latin-1 so non-UTF-8 bytes are covered
    too; the ByteLevel pre-tokenizer maps them back to the same bytes.
    """
    ids = tokenizer.encode(data.decode("latin-1")).ids
    # skip_special_tokens=False: a special token matched as a substring
    # (e.g. "tool_call" in "tool_calling") must be restored for the round-trip
    # to be lossless. The default True is for generation, not verification.
    roundtrip = tokenizer.decode(ids, skip_special_tokens=False).encode("latin-1")
    return VerifyStats(
        raw_bytes=len(data),
        token_count=len(ids),
        roundtrip_bytes=len(roundtrip),
        lossless=roundtrip == data,
    )


def byte_coverage(tokenizer: Tokenizer) -> tuple[list[int], list[int]]:
    """Return ``(covered, missing)`` raw byte values in 0..255.

    A byte is covered if a lone byte round-trips. Bytes with no vocab token
    (unobserved during training) are silently dropped and thus missing.
    """
    covered: list[int] = []
    missing: list[int] = []
    for b in range(256):
        if verify_bytes(tokenizer, bytes([b])).lossless:
            covered.append(b)
        else:
            missing.append(b)
    return covered, missing


ASCII_CORPUS = (
    "hello world " * 300 + "the quick brown fox jumps " * 300 + "def foo():\n    return 42\n" * 300
)


def _train(tmp_path: Path, corpus: str, vocab_size: int) -> Tokenizer:
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    (cdir / "web.txt").write_text(corpus)
    config = TokenizerConfig(
        vocab_size=vocab_size,
        train_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
        special_tokens=["im_start", "im_end"],
        eos_token="im_end",
    )
    return Tokenizer.from_file(str(train(config)))


def _ascii_tokenizer(tmp_path: Path) -> Tokenizer:
    return _train(tmp_path, ASCII_CORPUS, 512)


def _restricted_tokenizer(tmp_path: Path) -> Tokenizer:
    """A byte-level BPE whose base alphabet is restricted (only 'a' forced).

    Our ``train()`` now guarantees all 256 byte-tokens, so to still exercise
    the verifier's *detection* of a non-lossless round-trip we build a
    tokenizer that genuinely drops bytes (unobserved, not forced).
    """
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    (cdir / "web.txt").write_text(ASCII_CORPUS)
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train(
        [str(cdir / "web.txt")],
        BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=512,
            min_frequency=2,
            special_tokens=["im_start", "im_end"],
            initial_alphabet=["a"],
            show_progress=False,
        ),
    )
    return tok


def _rich_corpus() -> str:
    # Enough unique repeated words to fill any small vocab with merges.
    random.seed(0)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    words = ["".join(random.choices(alpha, k=random.randint(3, 8))) for _ in range(3000)]
    return (" ".join(words) + "\n") * 20


def test_verify_bytes_lossless_on_text(tmp_path):
    tok = _ascii_tokenizer(tmp_path)
    data = b"hello world def foo(): return 42"
    stats = verify_bytes(tok, data)
    assert stats.lossless
    assert stats.diff == 0
    assert stats.raw_bytes == stats.roundtrip_bytes == len(data)


def test_verify_bytes_stats_consistent(tmp_path):
    tok = _ascii_tokenizer(tmp_path)
    stats = verify_bytes(tok, b"hello hello hello")
    assert stats.token_id_bytes == 2 * stats.token_count
    assert stats.diff == stats.raw_bytes - stats.roundtrip_bytes
    assert stats.bytes_per_token == stats.raw_bytes / stats.token_count


def test_verify_bytes_detects_missing_byte(tmp_path):
    tok = _restricted_tokenizer(tmp_path)
    # NUL (0x00) is unobserved and not forced, so it has no token and is dropped.
    stats = verify_bytes(tok, b"abc\x00def")
    assert not stats.lossless
    assert stats.diff > 0
    assert stats.roundtrip_bytes < stats.raw_bytes


def test_train_produces_full_byte_coverage(tmp_path):
    # train() now seeds all 256 byte-tokens, so every byte round-trips.
    tok = _ascii_tokenizer(tmp_path)
    covered, missing = byte_coverage(tok)
    assert len(covered) == 256
    assert missing == []


def test_byte_coverage_partitions_256(tmp_path):
    tok = _ascii_tokenizer(tmp_path)
    covered, missing = byte_coverage(tok)
    assert sorted(covered + missing) == list(range(256))
    assert not set(covered) & set(missing)


def test_natural_text_round_trip(tmp_path):
    tok = _ascii_tokenizer(tmp_path)
    text = "hello world def foo(): return 42"
    assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text


def test_vocab_size_reaches_configured(tmp_path):
    tok = _train(tmp_path, _rich_corpus(), 512)
    assert len(tok.get_vocab()) == 512


def _main() -> None:
    """Manual entry point: verify the trained artifact against file(s).

    Run directly (not via pytest):
        uv run python tests/test_tokenizer_verify.py FILE [FILE ...] [--coverage]
    """
    import argparse
    import sys

    from kestrel.common.config import load_config

    parser = argparse.ArgumentParser(
        description="Verify the trained tokenizer artifact against file(s)."
    )
    parser.add_argument("files", nargs="+", help="file path(s) to round-trip")
    parser.add_argument("--config", default="configs/tokenizer/train.yaml")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="also report which of the 256 raw byte values round-trip",
    )
    args = parser.parse_args()

    config = load_config(args.config, TokenizerConfig)
    artifact = Path(config.output_dir) / "tokenizer.json"
    if not artifact.exists():
        print(f"artifact not found: {artifact}")
        print("train it first: uv run python -m kestrel.tokenizer.train")
        raise SystemExit(1)

    tokenizer = Tokenizer.from_file(str(artifact))
    all_lossless = True
    for file in args.files:
        path = Path(file)
        if not path.exists():
            print(f"{file}: not found", file=sys.stderr)
            all_lossless = False
            continue
        stats = verify_bytes(tokenizer, path.read_bytes())
        status = "LOSSLESS" if stats.lossless else f"DIFF {stats.diff:+d} BYTES"
        print(
            f"{file}: raw={stats.raw_bytes} tokens={stats.token_count} "
            f"round-trip={stats.roundtrip_bytes} {status}"
        )
        all_lossless = all_lossless and stats.lossless

    if args.coverage:
        covered, missing = byte_coverage(tokenizer)
        line = f"raw-byte coverage: {len(covered)}/256"
        if missing:
            line += f"  (missing {len(missing)}: {', '.join(hex(b) for b in missing)})"
        print(line)

    raise SystemExit(0 if all_lossless else 1)


if __name__ == "__main__":
    _main()
