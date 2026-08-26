"""Build a pretraining corpus from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from kestrel.common.config import load_config
from kestrel.corpus.builder import build
from kestrel.corpus.config import CorpusConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pretraining corpus.")
    parser.add_argument("--config", required=True, type=Path, help="Corpus YAML config path")
    parser.add_argument("--force", action="store_true", help="Rebuild even if complete")
    args = parser.parse_args()

    config = load_config(args.config, CorpusConfig)
    results = build(config, force=args.force)
    for name, byte_count in results.items():
        print(f"{name}: {byte_count} bytes")


if __name__ == "__main__":
    main()
