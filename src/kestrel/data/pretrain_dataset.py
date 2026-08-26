"""Pretraining dataset: stream document-level JSONL, tokenize, pack into batches.

Reads the corpus builder's document-level output (one JSON document per physical
line), tokenizes each document with the trained BPE tokenizer, wraps each document
in ``im_start`` / ``im_end`` boundary tokens, and assigns every token a ``doc_id``.
Tokens are accumulated in a buffer and cut into fixed-length sequences of
``context_length``. Each sequence yields a next-token pair: ``input = seq`` and
``target = seq`` shifted left by one; the final position repeats the last token.
Batches of shape ``(batch_size, context_length)`` (int32) are yielded as
``(input, target, doc_ids)`` until the ``total_tokens`` cap is reached or the input
is exhausted.

For directory inputs, documents are drawn from the component files with a
token-deficit scheduler, so the emitted token stream approximates the manifest's
target domain mix instead of consuming one file completely before moving to the
next. The scheduler tracks emitted tokens per source, not documents or bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from kestrel.common.config import BaseConfig


class PretrainDatasetConfig(BaseConfig):
    """Settings for the pretraining token stream (strict, no unknown keys).

    ``input`` is a single ``.jsonl`` file or a directory of ``.jsonl`` files (the
    corpus builder's output). ``total_tokens=None`` runs until the input is
    exhausted.
    """

    input: str
    tokenizer_path: str
    context_length: int = 2048
    batch_size: int = 8
    total_tokens: int | None = None
    seed: int = 0


@dataclass
class _Source:
    """One corpus component file and its target share of the token stream."""

    path: Path
    domain: str
    target_fraction: float
    estimated_tokens: int
    total_docs: int


@dataclass
class _ManifestInfo:
    """Corpus totals used by :meth:`PretrainDataset.estimated_steps`."""

    total_docs: int = 0
    total_text_tokens: int = 0


def _document_text(row: object) -> str:
    """Return document text from a JSONL row."""
    if isinstance(row, dict):
        text = row.get("text")
        if isinstance(text, str):
            return text
    return json.dumps(row, ensure_ascii=False)


def _line_offsets(path: Path) -> np.ndarray:
    """Collect byte offsets for non-blank physical lines in a corpus file."""
    offsets = array("Q")
    offset = 0
    with path.open("rb") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            if line not in (b"\n", b"\r\n"):
                offsets.append(offset)
            offset += len(line)
    return np.frombuffer(offsets, dtype=np.uint64).copy()


def _source_shuffle_seed(config_seed: int, domain: str) -> int:
    """Derive a stable per-source shuffle seed from the dataset seed."""
    digest = hashlib.sha256(f"{config_seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class _ShuffledDocumentSource:
    """Deterministically shuffled document stream for one corpus file.

    Only physical line offsets are stored in memory; document text is read by
    seeking to each shuffled offset. The shuffled offset order is recomputed
    from the file and seed, so checkpoint state only needs the integer position.
    """

    def __init__(self, path: Path, seed: int) -> None:
        self.path = path
        self.seed = seed
        self.is_jsonl = path.suffix == ".jsonl"
        self.offsets = _line_offsets(path)
        if self.offsets.size:
            np.random.default_rng(seed).shuffle(self.offsets)
        self.position = 0
        self._fin: BinaryIO | None = None

    def _ensure_open(self) -> BinaryIO:
        if self._fin is None:
            self._fin = self.path.open("rb")
        return self._fin

    def next_text(self) -> str:
        if self.position >= self.offsets.size:
            raise StopIteration
        fin = self._ensure_open()
        while self.position < self.offsets.size:
            offset = int(self.offsets[self.position])
            self.position += 1
            fin.seek(offset)
            line = fin.readline().decode("utf-8")
            if self.is_jsonl:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                text = _document_text(json.loads(line))
            else:
                text = line.rstrip("\r\n")
            if text.strip():
                return text
        self.close()
        raise StopIteration

    def close(self) -> None:
        if self._fin is not None:
            self._fin.close()
            self._fin = None

    def state(self) -> dict[str, int]:
        return {"position": self.position, "total": int(self.offsets.size)}

    def load_state(self, state: dict[str, int]) -> None:
        expected = int(state["total"])
        if expected != self.offsets.size:
            msg = (
                f"corpus file changed for {self.path}: "
                f"expected {expected} documents, found {self.offsets.size}"
            )
            raise ValueError(msg)
        position = int(state["position"])
        if position < 0 or position > self.offsets.size:
            msg = f"invalid shuffled document position {position} for {self.path}"
            raise ValueError(msg)
        self.position = position


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _normalize_fractions(sources: list[_Source]) -> None:
    """Make ``target_fraction`` sum to 1, falling back to estimated token counts."""
    total = sum(source.target_fraction for source in sources)
    if total > 0:
        for source in sources:
            source.target_fraction /= total
        return

    estimated_total = sum(source.estimated_tokens for source in sources)
    if estimated_total > 0:
        for source in sources:
            source.target_fraction = source.estimated_tokens / estimated_total
        return

    for source in sources:
        source.target_fraction = 1.0 / len(sources)


def choose_deficit_source(
    active: Sequence[int],
    fractions: Sequence[float],
    emitted: Sequence[float],
    rng: random.Random,
) -> int:
    """Choose the active source whose emitted tokens are furthest below target.

    Fractions are renormalized over the active set, so exhausted sources do not
    distort the remaining mix. A tiny seeded random term breaks ties
    deterministically.
    """
    if not active:
        msg = "no active sources"
        raise ValueError(msg)
    if len(active) == 1:
        return active[0]

    total = sum(emitted[index] for index in active)
    fraction_total = sum(fractions[index] for index in active)
    best_index = active[0]
    best_score = -math.inf
    for index in active:
        target = fractions[index] / fraction_total if fraction_total > 0 else 1.0 / len(active)
        score = target * total - emitted[index] + rng.random() * 1e-9
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _resolve_sources(input_path: str) -> tuple[list[_Source], _ManifestInfo]:
    """Resolve the input path into weighted document sources and manifest totals."""
    p = Path(input_path)
    if p.is_dir():
        manifest_path = p / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sources: list[_Source] = []
            raw_files = manifest.get("files", [])
            if not isinstance(raw_files, list):
                raw_files = []
            for raw in raw_files:
                if not isinstance(raw, dict):
                    continue
                path_value = raw.get("path")
                if not isinstance(path_value, str):
                    continue
                path = p / path_value
                if not path.exists():
                    continue
                token_count = _as_int(raw.get("token_count"))
                estimated = _as_int(raw.get("estimated_token_count"))
                if token_count is None and estimated is None:
                    estimated = max(path.stat().st_size, 1) // 4
                sources.append(
                    _Source(
                        path=path,
                        domain=str(raw.get("domain", path.stem)),
                        target_fraction=float(raw.get("target_fraction") or 0.0),
                        estimated_tokens=token_count
                        if token_count is not None
                        else (estimated if estimated is not None else 0),
                        total_docs=_as_int(raw.get("doc_count")) or 0,
                    )
                )
            if not sources:
                msg = f"no corpus files found in {input_path}"
                raise FileNotFoundError(msg)
            total_token_count = _as_int(manifest.get("total_token_count"))
            total_estimated = _as_int(manifest.get("total_estimated_token_count"))
            if total_token_count is None and total_estimated is None:
                total_estimated = sum(source.estimated_tokens for source in sources)
            info = _ManifestInfo(
                total_docs=_as_int(manifest.get("total_doc_count"))
                or sum(source.total_docs for source in sources),
                total_text_tokens=total_token_count
                if total_token_count is not None
                else (total_estimated if total_estimated is not None else 0),
            )
            _normalize_fractions(sources)
            return sources, info

        files = sorted(p.glob("*.jsonl"))
        if not files:
            files = sorted(p.glob("*.txt"))
        if not files:
            msg = f"no .jsonl files found in {input_path}"
            raise FileNotFoundError(msg)
        sources = [
            _Source(
                path=path,
                domain=path.stem,
                target_fraction=0.0,
                estimated_tokens=max(path.stat().st_size, 1) // 4,
                total_docs=0,
            )
            for path in files
        ]
        info = _ManifestInfo(
            total_docs=0, total_text_tokens=sum(s.estimated_tokens for s in sources)
        )
        _normalize_fractions(sources)
        return sources, info

    if p.is_file():
        source = _Source(
            path=p,
            domain=p.stem,
            target_fraction=1.0,
            estimated_tokens=max(p.stat().st_size, 1) // 4,
            total_docs=0,
        )
        return [source], _ManifestInfo(total_docs=0, total_text_tokens=source.estimated_tokens)

    msg = f"pretrain input path not found: {input_path}"
    raise FileNotFoundError(msg)


class PretrainDataset:
    """Iterable that yields ``(input, target, doc_ids)`` int32 batches of shape ``(B, T)``."""

    def __init__(self, config: PretrainDatasetConfig) -> None:
        self.config = config
        self._tokenizer = Tokenizer.from_file(config.tokenizer_path)
        self._sources, self._manifest = _resolve_sources(config.input)
        self._im_start = self._require_token("im_start")
        self._im_end = self._require_token("im_end")

    def _require_token(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None or token_id < 0:
            msg = f"tokenizer is missing required special token: {token}"
            raise ValueError(msg)
        return token_id

    def estimated_steps(self) -> int:
        """Estimate full training steps from manifest token/doc totals.

        The estimate adds two boundary tokens per document and divides by the
        tokens consumed per step (``batch_size * context_length``). If
        ``total_tokens`` is set, the cap is used as an upper bound.
        """
        tokens = self._manifest.total_text_tokens + 2 * self._manifest.total_docs
        if self.config.total_tokens is not None:
            tokens = min(tokens, self.config.total_tokens)
        return tokens // (self.config.batch_size * self.config.context_length)

    def iterator(self) -> PretrainDatasetIterator:
        """Return a fresh resumable iterator over the dataset."""
        return PretrainDatasetIterator(self)

    def load_iterator(self, state: dict[str, Any]) -> PretrainDatasetIterator:
        """Return an iterator restored from a :meth:`PretrainDatasetIterator.state_dict`."""
        iterator = self.iterator()
        iterator.load_state_dict(state)
        return iterator

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        return self.iterator()


class PretrainDatasetIterator:
    """Resumable iterator that yields ``(input, target, doc_ids)`` int32 batches."""

    def __init__(self, dataset: PretrainDataset) -> None:
        self._dataset = dataset
        config = dataset.config
        self._t = config.context_length
        self._b = config.batch_size
        self._cap = config.total_tokens
        self._sources = dataset._sources
        self._fractions = [source.target_fraction for source in self._sources]
        self._active = list(range(len(self._sources)))
        self._emitted = [0.0] * len(self._sources)
        self._rng = random.Random(config.seed)
        self._buf: list[int] = []
        self._doc_buf: list[int] = []
        self._batch_in: list[list[int]] = []
        self._batch_tgt: list[list[int]] = []
        self._batch_doc: list[list[int]] = []
        self._emitted_total = 0
        self._next_doc_id = 0
        self._source_iters: list[_ShuffledDocumentSource | None] = [None] * len(self._sources)
        self._closed = False

    def __iter__(self) -> PretrainDatasetIterator:
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for source in self._source_iters:
            if source is not None:
                source.close()

    def __next__(self) -> tuple[mx.array, mx.array, mx.array]:
        if self._closed:
            raise StopIteration
        if self._cap is not None and self._emitted_total >= self._cap:
            self.close()
            raise StopIteration

        while True:
            while len(self._buf) >= self._t and len(self._batch_in) < self._b:
                window = self._buf[: self._t]
                del self._buf[: self._t]
                doc_window = self._doc_buf[: self._t]
                del self._doc_buf[: self._t]
                self._batch_in.append(window)
                self._batch_tgt.append([*window[1:], window[-1]])
                self._batch_doc.append(doc_window)
                self._emitted_total += self._t

            if len(self._batch_in) == self._b:
                batch = (
                    mx.array(self._batch_in, dtype=mx.int32),
                    mx.array(self._batch_tgt, dtype=mx.int32),
                    mx.array(self._batch_doc, dtype=mx.int32),
                )
                self._batch_in = []
                self._batch_tgt = []
                self._batch_doc = []
                if self._cap is not None and self._emitted_total >= self._cap:
                    self.close()
                return batch

            if not self._active:
                self.close()
                raise StopIteration

            self._append_next_document()

    def _append_next_document(self) -> None:
        """Append one non-empty document from the token-deficit scheduler."""
        while self._active:
            index = choose_deficit_source(self._active, self._fractions, self._emitted, self._rng)
            source = self._sources[index]
            source_iter = self._source_iters[index]
            if source_iter is None:
                seed = _source_shuffle_seed(self._dataset.config.seed, source.domain)
                source_iter = _ShuffledDocumentSource(source.path, seed)
                self._source_iters[index] = source_iter
            try:
                text = source_iter.next_text()
            except StopIteration:
                self._active.remove(index)
                continue

            token_ids = self._dataset._tokenizer.encode(text, add_special_tokens=False).ids
            if not token_ids:
                continue
            seq = [self._dataset._im_start, *token_ids, self._dataset._im_end]
            self._buf.extend(seq)
            self._doc_buf.extend([self._next_doc_id] * len(seq))
            self._next_doc_id += 1
            self._emitted[index] += len(seq)
            return

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the iterator position."""
        return {
            "format_version": 1,
            "config": self._dataset.config.model_dump(),
            "active": list(self._active),
            "emitted": list(self._emitted),
            "emitted_total": self._emitted_total,
            "next_doc_id": self._next_doc_id,
            "buf": list(self._buf),
            "doc_buf": list(self._doc_buf),
            "batch_in": [list(row) for row in self._batch_in],
            "batch_tgt": [list(row) for row in self._batch_tgt],
            "batch_doc": [list(row) for row in self._batch_doc],
            "rng_state": list(self._rng.getstate()),
            "source_states": [
                source.state() if source is not None else None for source in self._source_iters
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator position from a :meth:`state_dict` snapshot."""
        if state.get("format_version") != 1:
            msg = f"unsupported dataset state format_version: {state.get('format_version')!r}"
            raise ValueError(msg)
        current_config = self._dataset.config.model_dump()
        if state.get("config") != current_config:
            msg = "dataset state was created with a different PretrainDatasetConfig"
            raise ValueError(msg)

        self.close()
        self._closed = False
        self._active = [int(index) for index in state["active"]]
        self._emitted = [float(value) for value in state["emitted"]]
        self._emitted_total = int(state["emitted_total"])
        self._next_doc_id = int(state["next_doc_id"])
        self._buf = [int(token) for token in state["buf"]]
        self._doc_buf = [int(doc_id) for doc_id in state["doc_buf"]]
        self._batch_in = [[int(token) for token in row] for row in state["batch_in"]]
        self._batch_tgt = [[int(token) for token in row] for row in state["batch_tgt"]]
        self._batch_doc = [[int(doc_id) for doc_id in row] for row in state["batch_doc"]]

        rng_state = state["rng_state"]
        self._rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))

        source_states = state["source_states"]
        if len(source_states) != len(self._sources):
            msg = "dataset state source_states length does not match the current corpus"
            raise ValueError(msg)
        self._source_iters = [None] * len(self._sources)
        for index, raw_state in enumerate(source_states):
            if raw_state is None:
                continue
            source = self._sources[index]
            source_iter = _ShuffledDocumentSource(
                source.path, _source_shuffle_seed(self._dataset.config.seed, source.domain)
            )
            source_iter.load_state(raw_state)
            self._source_iters[index] = source_iter
