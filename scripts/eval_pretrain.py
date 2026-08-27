"""Kestrel pretrain checkpoint evaluation CLI (TASK-005.13).

Loads a saved pretrain checkpoint and reports held-out next-token metrics over
a configurable token budget. This is a read-only post-training evaluation tool;
it does not build the corpus, modify checkpoints, or resume training.

Usage:
    uv run python scripts/eval_pretrain.py \\
        --pretrain-config configs/kestrel/50m/pretrain.yaml \\
        --checkpoint checkpoints/pretrain/50m/best \\
        --max-tokens 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kestrel.common.config import load_config
from kestrel.eval.pretrain import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROGRESS_EVERY_TOKENS,
    evaluate_checkpoint,
)
from kestrel.train.pretrain import PretrainConfig

DEFAULT_CONFIG = "configs/kestrel/50m/pretrain.yaml"


def _format_metrics(name: str, metrics: object) -> list[str]:
    data = metrics.to_dict()  # type: ignore[attr-defined]
    return [
        f"{name}: ",
        f"  loss:         {data['loss']:.6f}",
        f"  perplexity:   {data['perplexity']:.4f}",
        f"  bits/token:   {data['bits_per_token']:.6f}",
        f"  tokens:       {data['tokens']}",
        f"  batches:      {data['batches']}",
    ]


def _format_text(result: object) -> str:
    lines = [
        f"checkpoint: {result.checkpoint}",  # type: ignore[attr-defined]
        f"split:      {result.split}",  # type: ignore[attr-defined]
        "",
    ]
    lines.extend(_format_metrics("mixed", result.mixed))  # type: ignore[attr-defined]
    if result.domains:  # type: ignore[attr-defined]
        lines.append("")
        for name, metrics in result.domains.items():  # type: ignore[attr-defined]
            lines.extend(_format_metrics(name, metrics))
    if result.samples is not None:  # type: ignore[attr-defined]
        lines.append("")
        lines.append("samples:")
        for sample in result.samples:  # type: ignore[attr-defined]
            lines.append(f"  {sample}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Kestrel pretrain checkpoint.")
    parser.add_argument(
        "--pretrain-config",
        default=DEFAULT_CONFIG,
        help="pretrain config YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint directory containing weights.npz (step_NNNNNN, best, or final)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="corpus split to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "maximum tokens to evaluate for the mixed split and each domain "
            "(0 = full split; default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--progress-every-tokens",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_TOKENS,
        help=(
            "print progress to stderr every N evaluated tokens (0 = disable; default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="also generate fixed greedy samples",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of text",
    )
    args = parser.parse_args()

    if args.max_tokens < 0:
        parser.error("--max-tokens must be >= 0")
    if args.progress_every_tokens < 0:
        parser.error("--progress-every-tokens must be >= 0")
    if args.split == "train":
        print("warning: evaluating the train split", file=sys.stderr)

    config_path = Path(args.pretrain_config)
    if not config_path.is_file():
        raise SystemExit(f"error: pretrain config not found: {config_path}")

    try:
        config = load_config(config_path, PretrainConfig)
        result = evaluate_checkpoint(
            pretrain_config=config,
            checkpoint=args.checkpoint,
            split=args.split,
            max_tokens=args.max_tokens or None,
            generate_samples=args.generate,
            progress_every_tokens=args.progress_every_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_text(result))


if __name__ == "__main__":
    main()
