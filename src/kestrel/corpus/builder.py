"""Build the pretraining corpus: a weighted mix of text sources (plan §8).

Generalizes the tokenizer-data prep into a reusable builder supporting two
source types: ``hf`` (stream from a HuggingFace dataset) and ``local`` (read an
existing file). ``build`` writes one text file per component under
``output_dir/{train,val[,test]}/`` and returns the per-component bytes written.

The corpus is split into train/val(/test) by a deterministic, order-independent
per-line hash (seeded by ``config.seed``), so the held-out validation/test slices
are reproducible and share the same distribution as the training data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

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


def _split_for(line: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    """Deterministically route a line to ``train``/``val``/``test``.

    Uses a stable SHA-256 of ``f"{seed}:{line}"`` mapped to a uniform value in
    [0, 1). Order-independent and reproducible: the same line + seed always maps
    to the same split, so no document leaks across splits.
    """
    h = int.from_bytes(hashlib.sha256(f"{seed}:{line}".encode()).digest()[:4], "big")
    r = (h % 10000) / 10000.0
    if test_fraction > 0 and r < test_fraction:
        return "test"
    if val_fraction > 0 and r < test_fraction + val_fraction:
        return "val"
    return "train"


@contextmanager
def _split_writers(
    out_dir: Path, name: str, val_fraction: float, test_fraction: float
) -> Iterator[dict[str, IO[str]]]:
    """Open ``out_dir/{split}/<name>.txt`` for each active split; close on exit."""
    splits = ["train"]
    if test_fraction > 0:
        splits.append("test")
    if val_fraction > 0:
        splits.append("val")
    handles: dict[str, IO[str]] = {}
    try:
        for split in splits:
            d = out_dir / split
            d.mkdir(parents=True, exist_ok=True)
            handles[split] = (d / f"{name}.txt").open("w", encoding="utf-8")
        yield handles
    finally:
        for handle in handles.values():
            handle.close()


def _build_local(
    source: LocalSourceConfig,
    target: int,
    out_dir: Path,
    name: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> int:
    """Copy up to ``target`` bytes from ``source.path``, split into train/val(/test).

    Lines are normalized to end with a newline and routed by :func:`_split_for`.
    Returns the total bytes written (may be less than ``target`` if the source
    file is exhausted first).
    """
    src = Path(source.path)
    written = 0
    with (
        _split_writers(out_dir, name, val_fraction, test_fraction) as handles,
        src.open("r", encoding="utf-8") as fin,
    ):
        for line in fin:
            if not line.endswith("\n"):
                line += "\n"
            handles[_split_for(line, seed, val_fraction, test_fraction)].write(line)
            written += len(line.encode())
            if written >= target:
                break
    return written


def _build_hf(
    name: str,
    source: HfSourceConfig,
    target: int,
    out_dir: Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> int:
    """Stream ``source`` (a HuggingFace dataset) up to ``target`` bytes, split.

    Returns the total bytes written (may be less than ``target`` if the dataset
    is exhausted first).
    """
    import truststore

    truststore.inject_into_ssl()

    ds = load_dataset(source.dataset, name=source.config, streaming=True)
    split = next(iter(ds.keys()))
    written = 0
    next_log_mb = 50
    with _split_writers(out_dir, name, val_fraction, test_fraction) as handles:
        for row in ds[split]:
            text = _extract_text(row, source.text_field)
            if text is None:
                continue
            line = text + "\n"
            handles[_split_for(line, seed, val_fraction, test_fraction)].write(line)
            written += len(line.encode())
            if written // _MB >= next_log_mb:
                print(f"  [{name}] {written // _MB} MB", flush=True)
                next_log_mb += 50
            if written >= target:
                break
    return written


def build(config: CorpusConfig) -> dict[str, int]:
    """Assemble the corpus; return per-component bytes written (across splits)."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for comp in config.components:
        target = int(config.total_bytes * comp.fraction)
        origin = (
            comp.source.path if isinstance(comp.source, LocalSourceConfig) else comp.source.dataset
        )
        print(f"[{comp.name}] target {target / _MB:.1f} MB from {origin}", flush=True)
        if isinstance(comp.source, LocalSourceConfig):
            results[comp.name] = _build_local(
                comp.source,
                target,
                out_dir,
                comp.name,
                config.seed,
                config.val_fraction,
                config.test_fraction,
            )
        else:
            results[comp.name] = _build_hf(
                comp.name,
                comp.source,
                target,
                out_dir,
                config.seed,
                config.val_fraction,
                config.test_fraction,
            )
        print(f"[{comp.name}] wrote {results[comp.name] / _MB:.1f} MB -> {out_dir}", flush=True)
    return results
