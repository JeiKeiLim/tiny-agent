"""Build the M2 SFT training mixture from prepared raw source files.

Usage:
    uv run python scripts/run_build_sft_mixture.py --config configs/kestrel/sft_data.yaml
"""

from __future__ import annotations

import argparse

from kestrel.common.config import load_config
from kestrel.data.sft_mixture import build_mixture
from kestrel.data.sft_prepare import SFTDataConfig

DEFAULT_CONFIG = "configs/kestrel/sft_data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the M2 SFT training mixture.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="SFT data config YAML")
    args = parser.parse_args()

    config = load_config(args.config, SFTDataConfig)
    if config.mixture is None:
        msg = "mixture config is missing from the SFT data config"
        raise SystemExit(msg)

    manifest = build_mixture(config.mixture)
    print(f"recipe:      {manifest.recipe_used}")
    print(f"total rows:  {manifest.total_rows}/{manifest.requested_total}")
    print(f"mixture:     {manifest.output_path}")
    print(f"manifest:    {manifest.manifest_path}")


if __name__ == "__main__":
    main()
