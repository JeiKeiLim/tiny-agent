"""Kestrel pretrain CLI (TASK-005.05).

Loads a :class:`~kestrel.train.pretrain.PretrainConfig` from YAML and runs the
pretrain phase (corpus -> dataset -> trainer). Mirrors ``scripts/check_model.py``.

Usage:
    uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain.yaml
"""

from __future__ import annotations

import argparse

from kestrel.common.config import load_config
from kestrel.train.pretrain import PretrainConfig, pretrain

DEFAULT_CONFIG = "configs/kestrel/50m/pretrain.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kestrel pretraining.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="pretrain config YAML")
    parser.add_argument(
        "--resume",
        default=None,
        help="full checkpoint directory to resume from (step_NNNNNN, best, or final)",
    )
    args = parser.parse_args()

    config = load_config(args.config, PretrainConfig)
    if args.resume is not None:
        config.resume = args.resume
    result = pretrain(config, config_path=args.config)
    print(f"final loss:  {result.final_loss:.4f}")
    print(f"steps:       {result.num_steps}")
    if result.best_val_loss is not None:
        print(f"best val:    {result.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
