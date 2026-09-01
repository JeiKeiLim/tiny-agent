"""Autoregressive text generation.

Given a model + tokenizer + prompt, repeatedly predict the next token from the
last position's logits until ``max_tokens`` new tokens are produced or the stop
(EOS) token is emitted. This is the same core ``generate()`` the plan assigns to
``serve/`` (doc-001 §14); it lives in ``model/`` now (model inference, no server)
so ``serve/`` can wrap it later.

Models exposing ``prefill`` and ``decode`` use a KV-cache path: the prompt is
processed once, then each new token runs a single-token decode step. Plain
callable models fall back to the no-cache path, which re-runs the full sequence
every step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
from tokenizers import Tokenizer

# im_end doubles as the EOS/stop token (see kestrel.tokenizer.config).
DEFAULT_STOP_TOKEN = "im_end"


def _apply_repetition_penalty(
    logits: mx.array, generated_ids: list[int], penalty: float
) -> mx.array:
    """Penalize logits for previously generated token IDs.

    Uses the common Hugging Face-style rule: positive logits are divided by
    ``penalty`` and negative logits are multiplied by ``penalty``.
    """
    ids = mx.array(sorted(set(generated_ids)), dtype=mx.int32)
    selected = mx.take(logits, ids)
    adjusted = mx.where(selected > 0.0, selected / penalty, selected * penalty)
    return mx.put_along_axis(logits, ids, adjusted, axis=0)


def _next_token_id(
    logits: mx.array,
    generated: list[int],
    temp: float,
    repetition_penalty: float,
) -> int:
    last = logits
    if repetition_penalty != 1.0 and generated:
        last = _apply_repetition_penalty(last, generated, repetition_penalty)
    if temp <= 0.0:
        return cast(int, mx.argmax(last).item())
    return cast(int, mx.random.categorical(last / temp).item())


def _generate_no_cache(
    model: Callable[[mx.array], mx.array],
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temp: float,
    stop_token_id: int,
    repetition_penalty: float,
    clear_cache_every: int,
) -> str:
    x = mx.array([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=mx.int32)
    generated: list[int] = []

    for _ in range(max_tokens):
        last = model(x)[0, -1, :]  # (V,) next-token logits
        next_id = _next_token_id(last, generated, temp, repetition_penalty)
        if next_id == stop_token_id:
            break
        generated.append(next_id)
        x = mx.concatenate([x, mx.array([[next_id]], dtype=mx.int32)], axis=1)
        if clear_cache_every > 0 and len(generated) % clear_cache_every == 0:
            mx.clear_cache()

    return tokenizer.decode(generated)


def _generate_with_cache(
    model: Any,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temp: float,
    stop_token_id: int,
    repetition_penalty: float,
    clear_cache_every: int,
) -> str:
    x = mx.array([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=mx.int32)
    logits, caches = model.prefill(x, reserve=max_tokens)
    generated: list[int] = []

    for _ in range(max_tokens):
        last = logits[0, -1, :]  # (V,) next-token logits
        next_id = _next_token_id(last, generated, temp, repetition_penalty)
        if next_id == stop_token_id:
            break
        generated.append(next_id)
        logits, caches = model.decode(mx.array([[next_id]], dtype=mx.int32), caches)
        if clear_cache_every > 0 and len(generated) % clear_cache_every == 0:
            mx.clear_cache()

    return tokenizer.decode(generated)


def generate(
    model: Callable[[mx.array], mx.array],
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
    stop_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    clear_cache_every: int = 64,
) -> str:
    """Generate text by repeatedly sampling the next token.

    ``temp == 0`` is greedy (argmax, deterministic); ``temp > 0`` samples from
    ``softmax(logits / temp)``. Stops after ``max_tokens`` new tokens or as soon
    as ``stop_token_id`` (default: the tokenizer's ``im_end``/EOS id) is
    produced. Returns the decoded generated text (the prompt is excluded).

    ``repetition_penalty`` is a decoding-side penalty for previously generated
    token IDs. ``1.0`` disables it; values greater than ``1.0`` make repeated
    tokens less likely.

    ``clear_cache_every`` releases unused MLX allocator cache every N generated
    tokens. A positive cadence bounds retained temporary cache memory. ``0``
    disables clearing.

    Models exposing ``prefill`` and ``decode`` use the KV-cache path; other
    callable models use the no-cache fallback.
    """
    if repetition_penalty < 1.0:
        msg = f"repetition_penalty must be >= 1.0, got {repetition_penalty}"
        raise ValueError(msg)
    if clear_cache_every < 0:
        msg = f"clear_cache_every must be >= 0, got {clear_cache_every}"
        raise ValueError(msg)
    if max_tokens <= 0:
        return tokenizer.decode([])
    if stop_token_id is None:
        default_stop = tokenizer.token_to_id(DEFAULT_STOP_TOKEN)
        if default_stop is None:
            msg = f"default stop token {DEFAULT_STOP_TOKEN!r} not found in tokenizer"
            raise ValueError(msg)
        stop_token_id = default_stop

    if hasattr(model, "prefill") and hasattr(model, "decode"):
        return _generate_with_cache(
            model,
            tokenizer,
            prompt,
            max_tokens,
            temp,
            stop_token_id,
            repetition_penalty,
            clear_cache_every,
        )
    return _generate_no_cache(
        model,
        tokenizer,
        prompt,
        max_tokens,
        temp,
        stop_token_id,
        repetition_penalty,
        clear_cache_every,
    )
