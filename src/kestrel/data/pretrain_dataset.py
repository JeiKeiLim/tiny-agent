"""Pretraining dataset: stream corpus text, tokenize, pack into (input, target) batches.

Reads raw text (the corpus builder's output) line by line, tokenizes each line with
the trained BPE tokenizer, accumulates token ids into a buffer, and cuts them into
fixed-length sequences of ``context_length``. Each sequence yields a next-token pair:
``input = seq`` and ``target = seq`` shifted left by one (``target[t] == input[t+1]``
for ``t < T-1``); the final position has no valid target and is dropped. Batches of
shape ``(batch_size, context_length)`` (int32) are yielded until the ``total_tokens``
cap is reached or the input is exhausted.

For directory inputs, lines are drawn from the ``.txt`` files with a seeded weighted
scheduler, so batches approximate the file-size domain mix instead of consuming one
file completely before moving to the next.

The shift convention matches the loss already used in ``scripts/check_model.py``
(``cross_entropy(logits[:, :-1], input_ids[:, 1:])``), so the trainer consumes the
batches with no extra bookkeeping.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO

import mlx.core as mx
from tokenizers import Tokenizer

from kestrel.common.config import BaseConfig


class PretrainDatasetConfig(BaseConfig):
    """Settings for the pretraining token stream (strict, no unknown keys).

    ``input`` is a single ``.txt`` file or a directory of ``.txt`` files (the corpus
    builder's output). ``total_tokens=None`` runs until the input is exhausted.
    """

    input: str
    tokenizer_path: str
    context_length: int = 2048
    batch_size: int = 8
    total_tokens: int | None = None
    seed: int = 0


class WeightedLineScheduler:
    """Yield ``(path, line)`` pairs from weighted text files.

    The scheduler samples one active file at a time with probability proportional to
    its weight, then yields one line from that file. Exhausted files are removed and
    the remaining weights are renormalized implicitly by sampling over the active set.
    """

    def __init__(self, files: Sequence[tuple[Path, float]], seed: int) -> None:
        self._files = list(files)
        self._seed = seed

    def iter_lines(self) -> Iterator[tuple[Path, str]]:
        if not self._files:
            return
        if len(self._files) == 1:
            path, _ = self._files[0]
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    yield path, line
            return

        rng = random.Random(self._seed)
        active: list[tuple[Path, float]] = list(self._files)
        open_files: dict[Path, IO[str]] = {}
        try:
            while active:
                weights = [weight for _, weight in active]
                if sum(weights) <= 0:
                    return
                index = rng.choices(range(len(active)), weights=weights, k=1)[0]
                path, _ = active[index]
                file_iter = open_files.get(path)
                if file_iter is None:
                    file_iter = path.open("r", encoding="utf-8")
                    open_files[path] = file_iter
                try:
                    line = next(file_iter)
                except StopIteration:
                    file_iter.close()
                    del open_files[path]
                    active.pop(index)
                    continue
                yield path, line
        finally:
            for file_iter in open_files.values():
                file_iter.close()


class PretrainDataset:
    """Iterable that yields ``(input, target)`` int32 batches of shape ``(B, T)``."""

    def __init__(self, config: PretrainDatasetConfig) -> None:
        self.config = config
        self._tokenizer = Tokenizer.from_file(config.tokenizer_path)
        self._files = self._resolve_files(config.input)

    @staticmethod
    def _resolve_files(input_path: str) -> list[tuple[Path, float]]:
        p = Path(input_path)
        if p.is_dir():
            files = sorted(p.glob("*.txt"))
            if not files:
                raise FileNotFoundError(f"no .txt files found in {input_path}")
            return [(path, float(max(path.stat().st_size, 1))) for path in files]
        if p.is_file():
            return [(p, 1.0)]
        raise FileNotFoundError(f"pretrain input path not found: {input_path}")

    def _iter_lines(self) -> Iterator[str]:
        scheduler = WeightedLineScheduler(self._files, self.config.seed)
        for _, line in scheduler.iter_lines():
            yield line

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        t = self.config.context_length
        b = self.config.batch_size
        cap = self.config.total_tokens
        buf: list[int] = []
        batch_in: list[list[int]] = []
        batch_tgt: list[list[int]] = []
        emitted = 0
        for line in self._iter_lines():
            buf.extend(self._tokenizer.encode(line, add_special_tokens=False).ids)
            while len(buf) >= t:
                seq = buf[:t]
                del buf[:t]
                batch_in.append(seq)
                batch_tgt.append([*seq[1:], seq[-1]])
                emitted += t
                if len(batch_in) == b:
                    inp = mx.array(batch_in, dtype=mx.int32)
                    tgt = mx.array(batch_tgt, dtype=mx.int32)
                    yield inp, tgt
                    batch_in, batch_tgt = [], []
                if cap is not None and emitted >= cap:
                    return
