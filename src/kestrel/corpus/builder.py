"""Build the pretraining corpus: a weighted mix of text sources (plan §8).

Generalizes the tokenizer-data prep into a reusable builder supporting two
source types: ``hf`` (stream from a HuggingFace dataset) and ``local`` (read an
existing file). ``build`` writes one text file per component under
``output_dir`` and returns the per-component bytes written.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from kestrel.corpus.config import (
    CorpusConfig,
    HfSourceConfig,
    LocalSourceConfig,
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


def _build_local(source: LocalSourceConfig, target: int, out_path: Path) -> int:
    """Copy up to ``target`` bytes from ``source.path`` into ``out_path``.

    Lines are normalized to end with a newline. Returns the bytes written (may
    be less than ``target`` if the source file is exhausted first).
    """
    src = Path(source.path)
    written = 0
    with src.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as f:
        for line in fin:
            if not line.endswith("\n"):
                line += "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            if written >= target:
                break
    return written


def _build_hf(name: str, source: HfSourceConfig, target: int, out_path: Path) -> int:
    """Stream ``source`` (a HuggingFace dataset) up to ``target`` bytes.

    Returns the bytes written (may be less than ``target`` if the dataset is
    exhausted first).
    """
    import truststore

    truststore.inject_into_ssl()

    ds = load_dataset(source.dataset, name=source.config, streaming=True)
    split = next(iter(ds.keys()))
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
            if written >= target:
                break
    return written


def build(config: CorpusConfig) -> dict[str, int]:
    """Assemble the corpus; return per-component bytes written."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for comp in config.components:
        target = int(config.total_bytes * comp.fraction)
        out_path = out_dir / f"{comp.name}.txt"
        origin = (
            comp.source.path if isinstance(comp.source, LocalSourceConfig) else comp.source.dataset
        )
        print(f"[{comp.name}] target {target / _MB:.1f} MB from {origin}", flush=True)
        if isinstance(comp.source, LocalSourceConfig):
            results[comp.name] = _build_local(comp.source, target, out_path)
        else:
            results[comp.name] = _build_hf(comp.name, comp.source, target, out_path)
        print(f"[{comp.name}] wrote {results[comp.name] / _MB:.1f} MB -> {out_path}", flush=True)
    return results
