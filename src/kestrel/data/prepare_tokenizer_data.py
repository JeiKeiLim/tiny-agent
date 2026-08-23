"""Prepare a representative text sample for BPE tokenizer training (plan §7).

Streams samples from configured HF sources (web / code / jsonl) up to each
source's byte target and writes one text file per source under ``output_dir``.
The raw sample is a runtime artifact (gitignored); this script + its config are
the reproducible source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import truststore

truststore.inject_into_ssl()

from datasets import load_dataset  # noqa: E402

from kestrel.common.config import load_config  # noqa: E402
from kestrel.data.tokenizer_data_config import (  # noqa: E402
    SourceConfig,
    TokenizerTrainDataConfig,
)

_MB = 1024 * 1024


def _extract_text(row: dict[str, object], text_field: str | None) -> str | None:
    """Return the text for a row, or None if the row has no usable text."""
    if text_field is not None:
        value = row.get(text_field)
        return value if isinstance(value, str) and value.strip() else None
    try:
        return json.dumps(row, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _stream_source(
    name: str, source: SourceConfig, target_bytes: int, out_path: Path
) -> int:
    """Stream ``source`` up to ``target_bytes`` into ``out_path``.

    Returns the number of bytes written (may be less than the target if the
    source is exhausted first).
    """
    ds = load_dataset(source.dataset, name=source.config, streaming=True)
    split = list(ds.keys())[0]
    written = 0
    next_log_mb = 50
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds[split]:
            text = _extract_text(row, source.text_field)
            if text is None:
                continue
            line = text + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            if written // _MB >= next_log_mb:
                print(f"  [{name}] {written // _MB} MB", flush=True)
                next_log_mb += 50
            if written >= target_bytes:
                break
    return written


def prepare(config: TokenizerTrainDataConfig) -> dict[str, int]:
    """Build the sample; return per-source bytes written."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for name, source in config.sources.items():
        target = int(config.total_bytes * source.fraction)
        out_path = out_dir / f"{name}.txt"
        print(f"[{name}] target {target / _MB:.0f} MB from {source.dataset}", flush=True)
        results[name] = _stream_source(name, source, target, out_path)
        print(f"[{name}] wrote {results[name] / _MB:.1f} MB -> {out_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenizer training data.")
    parser.add_argument("--config", default="configs/tokenizer/train_data.yaml")
    args = parser.parse_args()
    config = load_config(args.config, TokenizerTrainDataConfig)
    results = prepare(config)
    total = sum(results.values())
    print(f"Total: {total / _MB:.1f} MB across {len(results)} sources")


if __name__ == "__main__":
    main()
