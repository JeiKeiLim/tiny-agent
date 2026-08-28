"""Kestrel model smoke-test CLI (dev tool).

Loads a Kestrel model via ``model/io.py`` (random-init when no checkpoint, a
trained checkpoint when ``--checkpoint`` is given), tokenizes a sample sentence
with the BPE tokenizer, runs a forward pass, and prints a report: param count,
logits shape, cross-entropy loss, and the top-k argmax token ids (+ decoded
strings) at the final position.

Built on the ``load(config, checkpoint)`` factory, so pointing ``--checkpoint``
at a pretraining output later requires no code change.

Usage:
    uv run python scripts/check_model.py --config configs/kestrel/50m/model.yaml
    uv run python scripts/check_model.py --checkpoint checkpoints/pretrain/kestrel-50m
    uv run python scripts/check_model.py --checkpoint ... --generate --max-tokens 256 --temp 0.8
    uv run python scripts/check_model.py --checkpoint ... --generate --repetition-penalty 1.2

An untrained (random-init) model gives a loss near ln(vocab) (~9.7 for 16k) and
gibberish top tokens — expected, not a bug.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx.nn.losses import cross_entropy
from tokenizers import Tokenizer

from kestrel.common.config import load_config
from kestrel.model.config import ModelConfig
from kestrel.model.generate import generate
from kestrel.model.io import load
from kestrel.model.kestrel import Kestrel, count_params

DEFAULT_CONFIG = "configs/kestrel/50m/model.yaml"
DEFAULT_TOKENIZER = "checkpoints/tokenizer/tokenizer.json"
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


@dataclass(frozen=True)
class ModelReport:
    """Result of a Kestrel forward-pass smoke test."""

    param_count: int
    logits_shape: tuple[int, ...]
    loss: float
    top_token_ids: list[int]
    top_tokens: list[str]


def report_from_model(model: Kestrel, tokenizer: Tokenizer, text: str, top_k: int) -> ModelReport:
    """Run ``model`` on ``text`` and summarize the forward pass."""
    input_ids = mx.array(tokenizer.encode(text).ids, dtype=mx.int32).reshape(1, -1)
    logits = model(input_ids)
    loss = cast(float, cross_entropy(logits[0, :-1], input_ids[0, 1:], reduction="mean").item())
    top_ids = [int(i) for i in cast(list[int], mx.argsort(-logits[0, -1])[:top_k].tolist())]
    return ModelReport(
        param_count=count_params(model),
        logits_shape=tuple(logits.shape),
        loss=loss,
        top_token_ids=top_ids,
        top_tokens=[tokenizer.decode([i]) for i in top_ids],
    )


def check_model(
    config: ModelConfig,
    checkpoint: str | Path | None = None,
    text: str = DEFAULT_TEXT,
    top_k: int = 5,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
) -> ModelReport:
    """Load a Kestrel model and run a forward-pass smoke test on ``text``.

    ``checkpoint=None`` gives a random-init model; otherwise weights are loaded
    (strictly) from ``checkpoint/weights.npz``.
    """
    model = load(config, checkpoint)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return report_from_model(model, tokenizer, text, top_k)


def _print_report(report: ModelReport, text: str) -> None:
    print(f"text:        {text!r}")
    print(f"param count: {report.param_count:,}")
    print(f"logits:      {report.logits_shape}")
    print(f"CE loss:     {report.loss:.4f}")
    top = [f"{t!r} (id {i})" for t, i in zip(report.top_tokens, report.top_token_ids, strict=True)]
    print(f"top tokens:  {', '.join(top)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test a Kestrel model (load, tokenize, forward, report)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="model config YAML")
    parser.add_argument(
        "--checkpoint", default=None, help="checkpoint dir (weights.npz); random-init if omitted"
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="sample sentence / generation prompt")
    parser.add_argument("--top-k", type=int, default=5, help="number of top argmax tokens to show")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="path to tokenizer.json")
    parser.add_argument("--generate", action="store_true", help="generate text after the report")
    parser.add_argument("--max-tokens", type=int, default=128, help="maximum tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="sampling temperature (0 = greedy)")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="HF-style repetition penalty for generated tokens (1.0 = disabled)",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"tokenizer artifact not found: {tokenizer_path}")
        print("train it first: uv run python -m kestrel.tokenizer.train")
        raise SystemExit(1)

    config = load_config(args.config, ModelConfig)
    model = load(config, args.checkpoint)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report = report_from_model(model, tokenizer, args.text, args.top_k)
    _print_report(report, args.text)

    if args.generate:
        generated = generate(
            model,
            tokenizer,
            args.text,
            args.max_tokens,
            temp=args.temp,
            repetition_penalty=args.repetition_penalty,
        )
        print(
            f"\ngenerated (max_tokens={args.max_tokens}, temp={args.temp}, "
            f"repetition_penalty={args.repetition_penalty}):"
        )
        print(generated)


if __name__ == "__main__":
    main()
