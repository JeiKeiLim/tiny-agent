"""Kestrel SFT eval CLI (TASK-007.03.09).

Loads a saved checkpoint, runs inference-only evaluation on the held-out SFT
eval bundle, and writes a JSON scorecard. This command does not modify
checkpoints.

Usage:
    uv run python scripts/run_eval_sft.py --config configs/kestrel/50m/eval_sft.yaml
    uv run python scripts/run_eval_sft.py --config ... --max-rows 20 --skip-perplexity
"""

from __future__ import annotations

import argparse
import math

from kestrel.common.config import load_config
from kestrel.eval.sft import SFTEvalConfig, SFTEvalScorecard, evaluate_sft, write_scorecard

DEFAULT_CONFIG = "configs/kestrel/50m/eval_sft.yaml"


def _format_value(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def _format_scorecard(scorecard: SFTEvalScorecard) -> str:
    lines = [
        "checkpoint\tstatus\tppl\tgsm8k_exact\ttool_seen_valid_json\t"
        "tool_unseen_valid_json\tassistant_non_empty"
    ]
    for result in scorecard.checkpoints:
        if result.status != "ok":
            lines.append(f"{result.name}\t{result.status}\tn/a\tn/a\tn/a\tn/a\tn/a")
            continue
        perplexity = result.perplexity.perplexity if result.perplexity is not None else None
        lines.append(
            "\t".join(
                [
                    result.name,
                    result.status,
                    _format_value(perplexity),
                    _format_value(result.math.exact_match_rate if result.math else None),
                    _format_value(result.tool.seen.valid_json_rate if result.tool else None),
                    _format_value(result.tool.unseen.valid_json_rate if result.tool else None),
                    _format_value(result.assistant.non_empty_rate if result.assistant else None),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Kestrel SFT checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="SFT eval config YAML")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="evaluate at most this many rows from each held-out eval set",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="scorecard JSON output path (default: config output path)",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="skip held-out pretrain perplexity evaluation",
    )
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be > 0")

    config = load_config(args.config, SFTEvalConfig)
    if args.max_rows is not None:
        config.data.max_rows_per_set = args.max_rows
    if args.output is not None:
        config.output = args.output
    if args.skip_perplexity:
        config.perplexity.enabled = False

    scorecard = evaluate_sft(config)
    output_path = write_scorecard(scorecard, config.output)

    print(f"scorecard: {output_path}")
    print(_format_scorecard(scorecard))


if __name__ == "__main__":
    main()
