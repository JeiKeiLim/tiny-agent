"""Tests for kestrel.model.generate (minimal autoregressive sampling).

Uses a TINY in-test tokenizer (WordLevel, 5 tokens) + a TINY Kestrel model so the
tests run fast and depend on no gitignored artifacts (no real 16384-vocab
tokenizer, no large model). A scripted stub model is used where a controlled
token sequence is needed (exact max_tokens count, EOS early-stop).
"""

import inspect
from collections.abc import Callable

import mlx.core as mx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from kestrel.model.config import ModelConfig
from kestrel.model.generate import _generate_no_cache, generate
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


def _fixed_logits_model(logits_by_id: dict[int, float]) -> Callable[[mx.array], mx.array]:
    """Stub model whose final-position logits are fixed regardless of input."""

    def fixed(x: mx.array) -> mx.array:
        logits = mx.zeros((1, x.shape[1], V))
        for token_id, value in logits_by_id.items():
            logits[0, -1, token_id] = value
        return logits

    return fixed


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


def test_repetition_penalty_default_is_noop() -> None:
    tok = _tiny_tokenizer()
    model = _fixed_logits_model({1: 10.0, 2: 9.0})
    out_default = generate(model, tok, "a b", max_tokens=3, temp=0.0)
    out_one = generate(model, tok, "a b", max_tokens=3, temp=0.0, repetition_penalty=1.0)
    assert out_default == "a a a"
    assert out_one == out_default


def test_repetition_penalty_suppresses_repeated_token_at_temp_zero() -> None:
    tok = _tiny_tokenizer()
    model = _fixed_logits_model({1: 10.0, 2: 9.0})
    out = generate(model, tok, "a b", max_tokens=2, temp=0.0, repetition_penalty=2.0)
    assert out == "a b"


def test_repetition_penalty_does_not_penalize_prompt_tokens() -> None:
    tok = _tiny_tokenizer()
    model = _fixed_logits_model({1: 10.0, 2: 9.0})
    out = generate(model, tok, "a", max_tokens=1, temp=0.0, repetition_penalty=2.0)
    assert out == "a"


def test_repetition_penalty_applies_before_sampling_at_temp_greater_than_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = _tiny_tokenizer()
    model = _fixed_logits_model({1: 10.0, 2: 9.0})
    categorical_inputs: list[mx.array] = []

    def fake_categorical(logits: mx.array) -> mx.array:
        categorical_inputs.append(logits)
        return mx.array(1)

    monkeypatch.setattr("mlx.core.random.categorical", fake_categorical)
    generate(model, tok, "a b", max_tokens=2, temp=0.5, repetition_penalty=2.0)

    assert len(categorical_inputs) == 2
    expected_first = mx.array([0.0, 20.0, 18.0, 0.0, 0.0])
    expected_second = mx.array([0.0, 10.0, 18.0, 0.0, 0.0])
    assert bool(mx.allclose(categorical_inputs[0], expected_first).item())
    assert bool(mx.allclose(categorical_inputs[1], expected_second).item())


def test_repetition_penalty_below_one_raises() -> None:
    tok = _tiny_tokenizer()
    model = _fixed_logits_model({1: 1.0})
    with pytest.raises(ValueError, match="repetition_penalty"):
        generate(model, tok, "a", max_tokens=1, temp=0.0, repetition_penalty=0.5)


def test_generate_clear_cache_default_is_sixty_four() -> None:
    parameter = inspect.signature(generate).parameters["clear_cache_every"]
    assert parameter.default == 64


def test_generate_clears_cache_on_configured_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = _tiny_tokenizer()
    calls: dict[str, int] = {"n": 0}
    model = _scripted_model(1, [1] * 8, calls)
    clear_calls: list[int] = []

    def fake_clear_cache() -> None:
        clear_calls.append(calls["n"])

    monkeypatch.setattr(mx, "clear_cache", fake_clear_cache)

    out = generate(model, tok, "a", max_tokens=8, temp=0.0, clear_cache_every=2)

    assert out == "a a a a a a a a"
    assert len(clear_calls) == 4


def test_generate_clear_cache_disabled_when_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = _tiny_tokenizer()
    calls: dict[str, int] = {"n": 0}
    model = _scripted_model(1, [1] * 4, calls)
    clear_calls: list[int] = []

    def fake_clear_cache() -> None:
        clear_calls.append(calls["n"])

    monkeypatch.setattr(mx, "clear_cache", fake_clear_cache)

    out = generate(model, tok, "a", max_tokens=4, temp=0.0, clear_cache_every=0)

    assert out == "a a a a"
    assert clear_calls == []


def test_generate_negative_clear_cache_every_raises() -> None:
    tok = _tiny_tokenizer()
    model = _fixed_logits_model({1: 1.0})
    with pytest.raises(ValueError, match="clear_cache_every"):
        generate(model, tok, "a", max_tokens=1, temp=0.0, clear_cache_every=-1)


def test_generate_kestrel_matches_no_cache_fallback() -> None:
    mx.random.seed(0)
    model = _tiny_model()
    tok = _tiny_tokenizer()

    cached = generate(model, tok, "a b", max_tokens=8, temp=0.0)
    no_cache = _generate_no_cache(
        model,
        tok,
        "a b",
        8,
        0.0,
        tok.token_to_id("im_end"),
        1.0,
        0,
    )

    assert cached == no_cache


class _ScriptedCachedModel:
    """Minimal prefill/decode model for exercising the cached generate path."""

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.prefill_calls = 0
        self.decode_calls = 0
        self._decode_index = 0

    def _logits(self) -> mx.array:
        logits = mx.zeros((1, 1, V))
        if self._decode_index < len(self.script):
            logits[0, 0, self.script[self._decode_index]] = 10.0
        return logits

    def prefill(self, x: mx.array, reserve: int = 0) -> tuple[mx.array, list[object]]:
        self.prefill_calls += 1
        return self._logits(), []

    def decode(self, x: mx.array, caches: list[object]) -> tuple[mx.array, list[object]]:
        self.decode_calls += 1
        self._decode_index += 1
        return self._logits(), caches


def test_generate_cached_path_produces_max_tokens() -> None:
    tok = _tiny_tokenizer()
    model = _ScriptedCachedModel([1] * 5)

    out = generate(model, tok, "a", max_tokens=5, temp=0.0)

    assert out == "a a a a a"
    assert model.prefill_calls == 1
    assert model.decode_calls == 5


def test_generate_cached_path_stops_on_eos() -> None:
    tok = _tiny_tokenizer()
    model = _ScriptedCachedModel([1, 2, 4])

    out = generate(model, tok, "a", max_tokens=10, temp=0.0)

    assert out == "a b"
    assert model.prefill_calls == 1
    assert model.decode_calls == 2


def test_generate_cached_path_applies_repetition_penalty() -> None:
    tok = _tiny_tokenizer()

    class FixedCachedModel:
        def prefill(self, x: mx.array, reserve: int = 0) -> tuple[mx.array, list[object]]:
            logits = mx.zeros((1, 1, V))
            logits[0, 0, 1] = 10.0
            logits[0, 0, 2] = 9.0
            return logits, []

        def decode(self, x: mx.array, caches: list[object]) -> tuple[mx.array, list[object]]:
            logits = mx.zeros((1, 1, V))
            logits[0, 0, 1] = 10.0
            logits[0, 0, 2] = 9.0
            return logits, caches

    out = generate(FixedCachedModel(), tok, "a", max_tokens=2, temp=0.0, repetition_penalty=2.0)

    assert out == "a b"


def test_generate_cached_path_clears_cache_on_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = _tiny_tokenizer()
    model = _ScriptedCachedModel([1] * 4)
    clear_calls: list[int] = []

    def fake_clear_cache() -> None:
        clear_calls.append(model.decode_calls)

    monkeypatch.setattr(mx, "clear_cache", fake_clear_cache)

    out = generate(model, tok, "a", max_tokens=4, temp=0.0, clear_cache_every=2)

    assert out == "a a a a"
    assert clear_calls == [2, 4]


def test_generate_zero_max_tokens_returns_empty_without_model_call() -> None:
    tok = _tiny_tokenizer()
    model = _ScriptedCachedModel([1])

    out = generate(model, tok, "a", max_tokens=0, temp=0.0)

    assert out == ""
    assert model.prefill_calls == 0
    assert model.decode_calls == 0
