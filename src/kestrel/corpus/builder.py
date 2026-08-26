"""Build the pretraining corpus: a weighted mix of text sources (plan §8).

Generalizes the tokenizer-data prep into a reusable builder supporting two
source types: ``hf`` (stream from a HuggingFace dataset) and ``local`` (read an
existing file). ``build`` writes one file per component under
``output_dir/{train,val[,test]}/`` and returns the per-component bytes written.

The default output format is document-level JSONL. One physical line is one
document, and internal newlines are preserved inside the JSON ``text`` field.
The legacy ``txt`` output format treats one physical line as one document.

The corpus is split into train/val(/test) by a deterministic, order-independent
per-document hash (seeded by ``config.seed``), so the held-out validation/test
slices are reproducible and share the same distribution as the training data.

``build`` is idempotent: if the output directory already contains a complete
corpus whose per-split manifests match the current config fingerprint, it
returns the existing byte counts without rebuilding. Pass ``force=True`` to
rebuild unconditionally.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from datasets import load_dataset
from tokenizers import Tokenizer

from kestrel.corpus.config import (
    CorpusConfig,
    HfSourceConfig,
    LocalSourceConfig,
)

_MB = 1024 * 1024


@dataclass
class _FileStats:
    """Per-split/per-component corpus statistics for manifest generation."""

    doc_count: int = 0
    byte_count: int = 0
    token_count: int = 0


def _extract_text(row: dict[str, object], text_field: str | None) -> str | None:
    """Return the text for a row, or None if the row has no usable text."""
    if text_field is not None:
        value = row.get(text_field)
        return value if isinstance(value, str) and value.strip() else None
    try:
        return json.dumps(row, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _local_row_text(row: dict[str, object]) -> str | None:
    """Return document text from a local JSONL row.

    Rows with a non-empty string ``text`` field use that field directly. Other
    rows are serialized as JSON so structured rows remain one document.
    """
    value = row.get("text")
    if isinstance(value, str) and value.strip():
        return value
    try:
        return json.dumps(row, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _split_for(document: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    """Deterministically route a document to ``train``/``val``/``test``.

    Uses a stable SHA-256 of ``f"{seed}:{document}"`` mapped to a uniform value in
    [0, 1). Order-independent and reproducible: the same document + seed always
    maps to the same split, so no document leaks across splits.
    """
    h = int.from_bytes(hashlib.sha256(f"{seed}:{document}".encode()).digest()[:4], "big")
    r = (h % 10000) / 10000.0
    if test_fraction > 0 and r < test_fraction:
        return "test"
    if val_fraction > 0 and r < test_fraction + val_fraction:
        return "val"
    return "train"


def _encode_document(domain: str, text: str, output_format: str) -> str:
    """Encode one document as a physical output line."""
    if output_format == "jsonl":
        return json.dumps({"domain": domain, "text": text}, ensure_ascii=False) + "\n"
    return text + "\n"


@contextmanager
def _split_writers(
    out_dir: Path,
    name: str,
    val_fraction: float,
    test_fraction: float,
    output_format: str,
) -> Iterator[dict[str, IO[str]]]:
    """Open ``out_dir/{split}/<name>.<ext>`` for each active split; close on exit."""
    splits = ["train"]
    if test_fraction > 0:
        splits.append("test")
    if val_fraction > 0:
        splits.append("val")
    ext = "jsonl" if output_format == "jsonl" else "txt"
    handles: dict[str, IO[str]] = {}
    try:
        for split in splits:
            d = out_dir / split
            d.mkdir(parents=True, exist_ok=True)
            handles[split] = (d / f"{name}.{ext}").open("w", encoding="utf-8")
        yield handles
    finally:
        for handle in handles.values():
            handle.close()


def _stats_for(
    stats: dict[str, dict[str, _FileStats]],
    split: str,
    name: str,
) -> _FileStats:
    return stats.setdefault(split, {}).setdefault(name, _FileStats())


def _write_document(
    handle: IO[str],
    domain: str,
    text: str,
    output_format: str,
    file_stats: _FileStats,
    tokenizer: Tokenizer | None,
) -> int:
    line = _encode_document(domain, text, output_format)
    handle.write(line)
    size = len(line.encode())
    file_stats.doc_count += 1
    file_stats.byte_count += size
    if tokenizer is not None:
        file_stats.token_count += len(tokenizer.encode(text, add_special_tokens=False).ids)
    return size


def _iter_local_documents(source: LocalSourceConfig) -> Iterator[str]:
    """Yield documents from a local ``.jsonl`` or legacy ``.txt`` file."""
    src = Path(source.path)
    if src.suffix == ".jsonl":
        with src.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    msg = f"local JSONL rows must be JSON objects: {src}"
                    raise ValueError(msg)
                text = _local_row_text(row)
                if text is not None:
                    yield text
        return

    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            text = line.rstrip("\r\n")
            if text.strip():
                yield text


def _build_local(
    source: LocalSourceConfig,
    target: int,
    out_dir: Path,
    name: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    output_format: str,
    stats: dict[str, dict[str, _FileStats]],
    tokenizer: Tokenizer | None,
) -> int:
    """Copy up to ``target`` bytes from ``source.path``, split into train/val(/test).

    Returns the total bytes written (may be less than ``target`` if the source
    file is exhausted first).
    """
    written = 0
    with _split_writers(out_dir, name, val_fraction, test_fraction, output_format) as handles:
        for text in _iter_local_documents(source):
            split = _split_for(text, seed, val_fraction, test_fraction)
            written += _write_document(
                handles[split],
                name,
                text,
                output_format,
                _stats_for(stats, split, name),
                tokenizer,
            )
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
    output_format: str,
    stats: dict[str, dict[str, _FileStats]],
    tokenizer: Tokenizer | None,
) -> int:
    """Stream ``source`` (a HuggingFace dataset) up to ``target`` bytes, split.

    Returns the total bytes written (may be less than ``target`` if the dataset
    is exhausted first).
    """
    import truststore

    truststore.inject_into_ssl()

    ds = load_dataset(source.dataset, name=source.config, streaming=True)
    dataset_split = next(iter(ds.keys()))
    written = 0
    next_log_mb = 50
    with _split_writers(out_dir, name, val_fraction, test_fraction, output_format) as handles:
        for row in ds[dataset_split]:
            text = _extract_text(row, source.text_field)
            if text is None:
                continue
            split = _split_for(text, seed, val_fraction, test_fraction)
            written += _write_document(
                handles[split],
                name,
                text,
                output_format,
                _stats_for(stats, split, name),
                tokenizer,
            )
            if written // _MB >= next_log_mb:
                print(f"  [{name}] {written // _MB} MB", flush=True)
                next_log_mb += 50
            if written >= target:
                break
    return written


def _active_splits(config: CorpusConfig) -> list[str]:
    splits = ["train"]
    if config.test_fraction > 0:
        splits.append("test")
    if config.val_fraction > 0:
        splits.append("val")
    return splits


def _config_fingerprint(config: CorpusConfig) -> str:
    """Stable hash of the corpus settings that determine the built artifacts."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _existing_results(out_dir: Path, config: CorpusConfig) -> dict[str, int] | None:
    """Return existing per-component byte counts if the corpus is complete.

    A corpus is complete when every active split has a manifest that matches the
    current config fingerprint, lists every component, and points at existing
    files whose sizes match the manifest byte counts.
    """
    fingerprint = _config_fingerprint(config)
    ext = "jsonl" if config.output_format == "jsonl" else "txt"
    results = {comp.name: 0 for comp in config.components}

    for split in _active_splits(config):
        manifest_path = out_dir / split / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(manifest, dict):
            return None
        if (
            manifest.get("split") != split
            or manifest.get("seed") != config.seed
            or manifest.get("output_format") != config.output_format
            or manifest.get("config_fingerprint") != fingerprint
        ):
            return None

        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != len(config.components):
            return None

        total_docs = 0
        total_bytes = 0
        for comp in config.components:
            expected_path = f"{comp.name}.{ext}"
            entry = next(
                (
                    item
                    for item in files
                    if isinstance(item, dict) and item.get("path") == expected_path
                ),
                None,
            )
            if entry is None:
                return None
            path = out_dir / split / expected_path
            if not path.exists():
                return None
            byte_count = entry.get("byte_count")
            doc_count = entry.get("doc_count")
            if not isinstance(byte_count, int) or not isinstance(doc_count, int):
                return None
            if path.stat().st_size != byte_count:
                return None
            results[comp.name] += byte_count
            total_docs += doc_count
            total_bytes += byte_count

        if (
            manifest.get("total_doc_count") != total_docs
            or manifest.get("total_byte_count") != total_bytes
        ):
            return None

    return results


def _write_manifests(
    out_dir: Path,
    config: CorpusConfig,
    stats: dict[str, dict[str, _FileStats]],
) -> None:
    ext = "jsonl" if config.output_format == "jsonl" else "txt"
    for split in _active_splits(config):
        by_name = stats.get(split, {})
        files: list[dict[str, object]] = []
        total_docs = 0
        total_bytes = 0
        total_tokens = 0
        for comp in config.components:
            file_stats = by_name.get(comp.name, _FileStats())
            entry: dict[str, object] = {
                "path": f"{comp.name}.{ext}",
                "domain": comp.name,
                "doc_count": file_stats.doc_count,
                "byte_count": file_stats.byte_count,
                "target_fraction": comp.fraction,
            }
            if config.tokenizer_path is None:
                entry["token_count"] = None
                entry["estimated_token_count"] = file_stats.byte_count // 4
            else:
                entry["token_count"] = file_stats.token_count
                entry["estimated_token_count"] = None
            files.append(entry)
            total_docs += file_stats.doc_count
            total_bytes += file_stats.byte_count
            total_tokens += file_stats.token_count

        manifest: dict[str, object] = {
            "split": split,
            "seed": config.seed,
            "output_format": config.output_format,
            "config_fingerprint": _config_fingerprint(config),
            "files": files,
            "total_doc_count": total_docs,
            "total_byte_count": total_bytes,
        }
        if config.tokenizer_path is None:
            manifest["total_token_count"] = None
            manifest["total_estimated_token_count"] = total_bytes // 4
        else:
            manifest["total_token_count"] = total_tokens
            manifest["total_estimated_token_count"] = None

        (out_dir / split / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def build(config: CorpusConfig, force: bool = False) -> dict[str, int]:
    """Assemble the corpus; return per-component bytes written (across splits).

    If the corpus under ``config.output_dir`` is already complete and matches
    ``config``, return the existing byte counts without rebuilding.
    """
    out_dir = Path(config.output_dir)
    if not force:
        existing = _existing_results(out_dir, config)
        if existing is not None:
            print(f"corpus already complete in {out_dir}; skipping build")
            return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = (
        Tokenizer.from_file(config.tokenizer_path) if config.tokenizer_path is not None else None
    )
    stats: dict[str, dict[str, _FileStats]] = {}
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
                config.output_format,
                stats,
                tokenizer,
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
                config.output_format,
                stats,
                tokenizer,
            )
        print(f"[{comp.name}] wrote {results[comp.name] / _MB:.1f} MB -> {out_dir}", flush=True)
        minimum = target * config.min_component_fill
        if results[comp.name] < minimum:
            msg = (
                f"component '{comp.name}' (source: {origin}) wrote "
                f"{results[comp.name]} bytes, below the minimum fill of "
                f"{minimum:.0f} bytes ({config.min_component_fill:.0%} of "
                f"target {target} bytes)"
            )
            raise ValueError(msg)
    _write_manifests(out_dir, config, stats)
    return results
