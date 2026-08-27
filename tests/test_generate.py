"""Tests for kestrel.model.generate (minimal autoregressive sampling).

Uses a TINY in-test tokenizer (WordLevel, 5 tokens) + a TINY Kestrel model so the
tests run fast and depend on no gitignored artifacts (no real 16384-vocab
tokenizer, no large model). A scripted stub model is used where a controlled
token sequence is needed (exact max_tokens count, EOS early-stop).
"""

from collections.abc import Callable

import mlx.core as mx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from kestrel.model.config import ModelConfig
from kestrel.model.generate import generate
from kestrel.model.kestrel import Kestrel

VOCAB = {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "im_end": 4}
V = len(VOCAB)  # 5


def _tiny_tokenizer() -> Tokenizer:
    tok = Tokenizer(WordLevel(vocab=dict(VOCAB), unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok


def _tiny_model() -> Kestrel:
    mx.random.seed(0)
    return Kestrel(
        ModelConfig(
            name="gen-test",
            vocab_size=V,
            context_length=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            hidden_size=16,
            intermediate_size=32,
        )
    )


def _scripted_model(
    prompt_len: int, script: list[int], calls: dict[str, int]
) -> Callable[[mx.array], mx.array]:
    """Stub model that emits the tokens in ``script`` (via last-position argmax)."""

    def scripted(x: mx.array) -> mx.array:
        calls["n"] += 1
        n_gen = x.shape[1] - prompt_len
        logits = mx.zeros((1, x.shape[1], V))
        logits[0, -1, script[n_gen] if n_gen < len(script) else 0] = 10.0
        return logits

    return scripted


def test_generate_produces_max_tokens_and_str() -> None:
    tok = _tiny_tokenizer()
    calls: dict[str, int] = {"n": 0}
    model = _scripted_model(2, [1, 1, 1, 1, 1], calls)  # prompt "a b" (2 tokens)
    out = generate(model, tok, "a b", max_tokens=5, temp=0.0)
    assert isinstance(out, str)
    assert out == "a a a a a"
    assert len(tok.encode(out).ids) == 5  # exactly max_tokens, no early stop


def test_generate_deterministic_at_temp_zero() -> None:
    tok = _tiny_tokenizer()
    model = _tiny_model()
    r1 = generate(model, tok, "a b", max_tokens=5, temp=0.0)
    r2 = generate(model, tok, "a b", max_tokens=5, temp=0.0)
    assert r1 == r2


def test_generate_stops_on_eos_before_max_tokens() -> None:
    tok = _tiny_tokenizer()
    calls: dict[str, int] = {"n": 0}
    model = _scripted_model(1, [1, 2, 4], calls)  # prompt "a"; emit a, b, then im_end
    out = generate(model, tok, "a", max_tokens=10, temp=0.0)
    assert out == "a b"
    assert calls["n"] == 3  # stopped right after im_end, not at max_tokens=10


def test_generate_sampling_runs() -> None:
    tok = _tiny_tokenizer()
    out = generate(_tiny_model(), tok, "a b", max_tokens=3, temp=1.0)
    assert isinstance(out, str)


def test_generate_sampling_passes_scaled_logits_to_categorical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = _tiny_tokenizer()
    base = _tiny_model()
    captured_logits: list[mx.array] = []
    categorical_inputs: list[mx.array] = []

    def model(x: mx.array) -> mx.array:
        logits = base(x)
        captured_logits.append(logits[0, -1, :])
        return logits

    def fake_categorical(logits: mx.array) -> mx.array:
        categorical_inputs.append(logits)
        return mx.array(1)

    monkeypatch.setattr("mlx.core.random.categorical", fake_categorical)
    generate(model, tok, "a b", max_tokens=1, temp=0.5)

    assert len(captured_logits) == 1
    assert len(categorical_inputs) == 1
    assert bool(mx.allclose(categorical_inputs[0], captured_logits[0] / 0.5).item())
