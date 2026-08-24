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


def generate(
    model: Callable[[mx.array], mx.array],
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
    stop_token_id: int | None = None,
) -> str:
    """Generate text by repeatedly sampling the next token.

    ``temp == 0`` is greedy (argmax, deterministic); ``temp > 0`` samples from
    ``softmax(logits / temp)``. Stops after ``max_tokens`` new tokens or as soon
    as ``stop_token_id`` (default: the tokenizer's ``im_end``/EOS id) is
    produced. Returns the decoded generated text (the prompt is excluded).
    """
    if stop_token_id is None:
        stop_token_id = tokenizer.token_to_id(DEFAULT_STOP_TOKEN)

    x = mx.array([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=mx.int32)
    generated: list[int] = []

    for _ in range(max_tokens):
        last = model(x)[0, -1, :]  # (V,) next-token logits
        if temp <= 0.0:
            next_id = cast(int, mx.argmax(last).item())
        else:
            next_id = cast(int, mx.random.categorical(mx.softmax(last / temp)).item())

        if stop_token_id is not None and next_id == stop_token_id:
            break
        generated.append(next_id)
        x = mx.concatenate([x, mx.array([[next_id]], dtype=mx.int32)], axis=1)

    return tokenizer.decode(generated)
