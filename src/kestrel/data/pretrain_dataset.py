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

import json
import math
import random
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
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
    iterator: Generator[str] | None = field(default=None, repr=False)


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


def _iter_documents(path: Path) -> Generator[str]:
    """Yield document texts from a ``.jsonl`` file or legacy ``.txt`` file."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                text = _document_text(json.loads(line))
                if text.strip():
                    yield text
        return

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            text = line.rstrip("\r\n")
            if text.strip():
                yield text


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

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        t = self.config.context_length
        b = self.config.batch_size
        cap = self.config.total_tokens
        sources = self._sources
        active = list(range(len(sources)))
        emitted = [0.0] * len(sources)
        fractions = [source.target_fraction for source in sources]
        rng = random.Random(self.config.seed)

        buf: list[int] = []
        doc_buf: list[int] = []
        batch_in: list[list[int]] = []
        batch_tgt: list[list[int]] = []
        batch_doc: list[list[int]] = []
        emitted_total = 0
        next_doc_id = 0

        try:
            while active:
                index = choose_deficit_source(active, fractions, emitted, rng)
                source = sources[index]
                if source.iterator is None:
                    source.iterator = _iter_documents(source.path)
                try:
                    text = next(source.iterator)
                except StopIteration:
                    active.remove(index)
                    continue

                token_ids = self._tokenizer.encode(text, add_special_tokens=False).ids
                if not token_ids:
                    continue
                seq = [self._im_start, *token_ids, self._im_end]
                buf.extend(seq)
                doc_buf.extend([next_doc_id] * len(seq))
                next_doc_id += 1
                emitted[index] += len(seq)

                while len(buf) >= t:
                    window = buf[:t]
                    del buf[:t]
                    doc_window = doc_buf[:t]
                    del doc_buf[:t]
                    batch_in.append(window)
                    batch_tgt.append([*window[1:], window[-1]])
                    batch_doc.append(doc_window)
                    emitted_total += t
                    if len(batch_in) == b:
                        yield (
                            mx.array(batch_in, dtype=mx.int32),
                            mx.array(batch_tgt, dtype=mx.int32),
                            mx.array(batch_doc, dtype=mx.int32),
                        )
                        batch_in, batch_tgt, batch_doc = [], [], []
                    if cap is not None and emitted_total >= cap:
                        return
        finally:
            for source in sources:
                if source.iterator is not None:
                    source.iterator.close()
