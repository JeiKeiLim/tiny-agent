"""Kestrel pretrain external benchmark evaluation CLI (TASK-011).

Loads a saved pretrain checkpoint and evaluates locally downloaded external
benchmark datasets. This command is read-only with respect to checkpoints and
does not download or modify benchmark data.

Usage:
    uv run python scripts/run_eval_pretrain_benchmarks.py \\
        --pretrain-config configs/kestrel/50m/pretrain.yaml \\
        --checkpoint checkpoints/pretrain/50m/best \\
        --data-dir /path/to/datasets \\
        --max-tokens 100000 \\
        --max-examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kestrel.common.config import load_config
from kestrel.eval.pretrain_benchmarks import (
    benchmark_names,
    evaluate_selected_benchmarks,
    parse_only,
    selected_specs,
    write_scorecard,
)
from kestrel.train.pretrain import PretrainConfig

DEFAULT_CONFIG = "configs/kestrel/50m/pretrain.yaml"
DEFAULT_OUTPUT = "data/pretrain_eval/scorecard.json"


def _format_result_line(result: object) -> str:
    data = result.to_dict()  # type: ignore[attr-defined]
    metrics = data["metrics"]
    if data["status"] != "ok":
        detail = data.get("error") or "n/a"
        return f"{data['name']}\t{data['kind']}\t{data['status']}\t{detail}"

    if data["kind"] == "language_modeling":
        detail = (
            f"bpb={metrics['bpb']:.6f} loss={metrics['loss']:.6f} "
            f"ppl={metrics['perplexity']:.4f} tokens={metrics['tokens']} "
            f"bytes={metrics['bytes']} examples={metrics['examples']}"
        )
    else:
        detail = (
            f"acc={metrics['acc']:.2f} acc_norm={metrics['acc_norm']:.2f} "
            f"examples={metrics['examples']} tokens={metrics['tokens']}"
        )
    return f"{data['name']}\t{data['kind']}\tok\t{detail}"


def _format_scorecard(scorecard: object) -> str:
    lines = ["name\tkind\tstatus\tdetail"]
    for result in scorecard.results:  # type: ignore[attr-defined]
        lines.append(_format_result_line(result))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Kestrel pretrain checkpoint on external benchmarks."
    )
    parser.add_argument(
        "--pretrain-config",
        default=DEFAULT_CONFIG,
        help="pretrain config YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint directory containing weights.npz",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="directory containing downloaded benchmark dataset directories",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=f"comma-separated benchmark names to evaluate (known: {', '.join(benchmark_names())})",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="skip large language-modeling sets (c4_en_validation, pile_test)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100_000,
        help="maximum tokens per language-modeling benchmark (0 = full; default: %(default)s)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="maximum examples per benchmark (default: full split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for deterministic benchmark-specific shuffling (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="record missing or failing benchmarks instead of exiting",
    )
    parser.add_argument(
        "--progress-every-tokens",
        type=int,
        default=100_000,
        help="progress interval for LM evaluation (0 disables; default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="scorecard JSON output path (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print scorecard JSON to stdout instead of a table",
    )
    args = parser.parse_args()

    if args.max_tokens < 0:
        parser.error("--max-tokens must be >= 0")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be > 0")
    if args.progress_every_tokens < 0:
        parser.error("--progress-every-tokens must be >= 0")

    config_path = Path(args.pretrain_config)
    if not config_path.is_file():
        raise SystemExit(f"error: pretrain config not found: {config_path}")

    try:
        pretrain_config = load_config(config_path, PretrainConfig)
        only = parse_only(args.only)
        specs = selected_specs(args.skip_large, only)
        if not specs:
            raise SystemExit("error: no benchmarks selected")

        scorecard = evaluate_selected_benchmarks(
            pretrain_config=pretrain_config,
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            specs=specs,
            max_tokens=args.max_tokens or None,
            max_examples=args.max_examples,
            seed=args.seed,
            allow_missing=args.allow_missing,
            progress_every_tokens=args.progress_every_tokens,
        )
        output_path = write_scorecard(scorecard, args.output)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.json:
        print(json.dumps(scorecard.to_dict(), indent=2))
    else:
        print(f"scorecard: {output_path}")
        print(_format_scorecard(scorecard))


if __name__ == "__main__":
    main()
