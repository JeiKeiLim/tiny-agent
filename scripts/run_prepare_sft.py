"""Prepare M2 SFT source datasets.

Usage:
    uv run python scripts/run_prepare_sft.py --source all
    uv run python scripts/run_prepare_sft.py --source assistant
    uv run python scripts/run_prepare_sft.py --source gsm8k
    uv run python scripts/run_prepare_sft.py --source tool
    uv run python scripts/run_prepare_sft.py --source public_tool
    uv run python scripts/run_prepare_sft.py --source internal_llm
    uv run python scripts/run_prepare_sft.py --source eval
"""

from __future__ import annotations

import argparse

import truststore

truststore.inject_into_ssl()

from kestrel.common.config import load_config  # noqa: E402
from kestrel.data.sft_prepare import (  # noqa: E402
    SFTDataConfig,
    SourceManifest,
    prepare_all,
    prepare_assistant,
    prepare_gsm8k,
    prepare_internal_llm,
    prepare_public_tool,
    prepare_tool,
)
from kestrel.data.sft_prepare_eval import prepare_eval  # noqa: E402

DEFAULT_CONFIG = "configs/kestrel/sft_data.yaml"


def _print_manifest(manifest: SourceManifest) -> None:
    print(
        f"{manifest.source}: wrote {manifest.written_rows}/{manifest.requested_rows} rows"
        f" -> {manifest.output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare M2 SFT source datasets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="SFT data config YAML")
    parser.add_argument(
        "--source",
        choices=[
            "assistant",
            "gsm8k",
            "tool",
            "public_tool",
            "internal_llm",
            "eval",
            "all",
        ],
        default="all",
        help="Which SFT source to prepare",
    )
    args = parser.parse_args()

    config = load_config(args.config, SFTDataConfig)
    if args.source == "assistant":
        manifests = [prepare_assistant(config)]
    elif args.source == "gsm8k":
        manifests = [prepare_gsm8k(config)]
    elif args.source == "tool":
        manifests = list(prepare_tool(config).values())
    elif args.source == "public_tool":
        manifests = [prepare_public_tool(config)]
    elif args.source == "internal_llm":
        manifest = prepare_internal_llm(config)
        if manifest is None:
            msg = "internal_llm source is disabled in the config"
            raise SystemExit(msg)
        manifests = [manifest]
    elif args.source == "eval":
        manifests = list(prepare_eval(config).values())
    else:
        manifests = list(prepare_all(config).values())

    for manifest in manifests:
        _print_manifest(manifest)


if __name__ == "__main__":
    main()
