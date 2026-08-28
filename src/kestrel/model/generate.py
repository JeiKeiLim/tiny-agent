"""Minimal autoregressive text generation.

Given a model + tokenizer + prompt, repeatedly predict the next token from the
last position's logits until ``max_tokens`` new tokens are produced or the stop
(EOS) token is emitted. This is the same core ``generate()`` the plan assigns to
``serve/`` (doc-001 §14); it lives in ``model/`` now (model inference, no server)
so ``serve/`` can wrap it later.

No KV cache: each step re-runs the full sequence through the model. That is the
minimal implementation and is fine for the 50M validation run (short
generations); a cached fast path can be added in ``serve/`` if generation
throughput ever matters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

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


def generate(
    model: Callable[[mx.array], mx.array],
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
    stop_token_id: int | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate text by repeatedly sampling the next token.

    ``temp == 0`` is greedy (argmax, deterministic); ``temp > 0`` samples from
    ``softmax(logits / temp)``. Stops after ``max_tokens`` new tokens or as soon
    as ``stop_token_id`` (default: the tokenizer's ``im_end``/EOS id) is
    produced. Returns the decoded generated text (the prompt is excluded).

    ``repetition_penalty`` is a decoding-side penalty for previously generated
    token IDs. ``1.0`` disables it; values greater than ``1.0`` make repeated
    tokens less likely.
    """
    if repetition_penalty < 1.0:
        msg = f"repetition_penalty must be >= 1.0, got {repetition_penalty}"
        raise ValueError(msg)
    if stop_token_id is None:
        stop_token_id = tokenizer.token_to_id(DEFAULT_STOP_TOKEN)

    x = mx.array([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=mx.int32)
    generated: list[int] = []

    for _ in range(max_tokens):
        last = model(x)[0, -1, :]  # (V,) next-token logits
        if repetition_penalty != 1.0 and generated:
            last = _apply_repetition_penalty(last, generated, repetition_penalty)
        if temp <= 0.0:
            next_id = cast(int, mx.argmax(last).item())
        else:
            next_id = cast(int, mx.random.categorical(last / temp).item())

        if stop_token_id is not None and next_id == stop_token_id:
            break
        generated.append(next_id)
        x = mx.concatenate([x, mx.array([[next_id]], dtype=mx.int32)], axis=1)

    return tokenizer.decode(generated)
