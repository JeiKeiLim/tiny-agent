"""Kestrel SFT CLI (TASK-007.03.08).

Loads a :class:`~kestrel.train.sft.SFTConfig` from YAML and runs the SFT phase
(pretrain checkpoint -> masked SFT dataset -> trainer). Mirrors
``scripts/run_pretrain.py``.

Usage:
    uv run python scripts/run_sft.py --config configs/kestrel/50m/sft.yaml
"""

from __future__ import annotations

import argparse

from kestrel.common.config import load_config
from kestrel.train.sft import SFTConfig, sft

DEFAULT_CONFIG = "configs/kestrel/50m/sft.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kestrel SFT.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="SFT config YAML")
    parser.add_argument(
        "--resume",
        default=None,
        help="full SFT checkpoint directory to resume from (step_NNNNNN, best, or final)",
    )
    args = parser.parse_args()

    config = load_config(args.config, SFTConfig)
    if args.resume is not None:
        config.resume = args.resume
    result = sft(config, config_path=args.config)
    print(f"final loss:  {result.final_loss:.4f}")
    print(f"steps:       {result.num_steps}")
    if result.best_val_loss is not None:
        print(f"best val:    {result.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
