"""Tests for the pretraining dataset (src/kestrel/data/pretrain_dataset.py).

A tiny BPE tokenizer is trained in-test (same pattern as test_model_check.py) so the
tests do not depend on the gitignored checkpoints/tokenizer/tokenizer.json.
"""

from collections import Counter
from itertools import islice
from pathlib import Path

import mlx.core as mx
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer

from kestrel.data.pretrain_dataset import (
    PretrainDataset,
    PretrainDatasetConfig,
    WeightedLineScheduler,
)
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train

SENTENCE = "hello world the quick brown fox jumps over the lazy dog. "


def _tiny_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(SENTENCE * 500 + "the quick brown fox jumps " * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp_path / "tok"),
        special_tokens=["im_start", "im_end"],
        eos_token="im_end",
    )
    return train(config)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _config(input_path: Path, tok: Path, **overrides: object) -> PretrainDatasetConfig:
    data: dict[str, object] = {
        "input": str(input_path),
        "tokenizer_path": str(tok),
        "context_length": 8,
        "batch_size": 2,
        "total_tokens": None,
        "seed": 0,
    }
    data.update(overrides)
    return PretrainDatasetConfig.model_validate(data)


def test_batch_shape(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    data = _write(tmp_path, "data.txt", "\n".join([SENTENCE * 2] * 5) + "\n")
    batches = list(PretrainDataset(_config(data, tok)))
    assert len(batches) >= 2
    for inp, tgt in batches:
        assert inp.shape == (2, 8)
        assert tgt.shape == (2, 8)
        assert inp.dtype == mx.int32
        assert tgt.dtype == mx.int32


def test_next_token_shift(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    text = SENTENCE * 5
    data = _write(tmp_path, "data.txt", text)
    inp, tgt = next(iter(PretrainDataset(_config(data, tok))))
    assert bool((tgt[:, :-1] == inp[:, 1:]).all())
    expected = Tokenizer.from_file(str(tok)).encode(text, add_special_tokens=False).ids[:8]
    assert inp[0].tolist() == expected


def test_total_tokens_cap(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    data = _write(tmp_path, "data.txt", SENTENCE * 20)
    capped = list(PretrainDataset(_config(data, tok, total_tokens=16)))
    assert len(capped) == 1
    uncapped = list(PretrainDataset(_config(data, tok)))
    assert len(uncapped) > 1


def test_config_rejects_bad_values(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    with pytest.raises(ValidationError):
        PretrainDatasetConfig(input="x", tokenizer_path=str(tok), context_length="8")
    with pytest.raises(ValidationError):
        PretrainDatasetConfig(input="x", tokenizer_path=str(tok), unknown_key=1)


def test_dir_input(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text(SENTENCE * 10, encoding="utf-8")
    (d / "b.txt").write_text(SENTENCE * 10, encoding="utf-8")
    batches = list(PretrainDataset(_config(d, tok)))
    assert len(batches) >= 1
    for inp, _ in batches:
        assert inp.shape == (2, 8)


def test_full_batches_only(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    data = _write(tmp_path, "data.txt", SENTENCE * 3)
    total = sum(inp.shape[0] * inp.shape[1] for inp, _ in PretrainDataset(_config(data, tok)))
    assert total % (2 * 8) == 0


def test_multi_file_batches_mix_domains(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("alpha " * 200, encoding="utf-8")
    (d / "b.txt").write_text("beta " * 200, encoding="utf-8")
    dataset = PretrainDataset(_config(d, tok, batch_size=1))
    decoder = Tokenizer.from_file(str(tok))
    decoded = " ".join(decoder.decode(inp[0].tolist()) for inp, _ in dataset)
    assert "alpha" in decoded
    assert "beta" in decoded


def test_single_file_preserves_line_order(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    words = [f"marker{index:02d}" for index in range(10)]
    text = (" ".join(words) + " ") * 10
    data = _write(tmp_path, "data.txt", text)
    inp, _ = next(iter(PretrainDataset(_config(data, tok, batch_size=1, context_length=256))))
    decoded = Tokenizer.from_file(str(tok)).decode(inp[0].tolist())
    positions = [decoded.index(word) for word in words]
    assert positions == sorted(positions)


def _scheduler_dir(tmp_path: Path, counts: dict[str, int]) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    for name, count in counts.items():
        (d / f"{name}.txt").write_text(f"{name}\n" * count, encoding="utf-8")
    return d


def test_weighted_scheduler_share_within_tolerance(tmp_path: Path) -> None:
    d = _scheduler_dir(tmp_path, {"a": 100_000, "b": 5_000, "c": 5_000})
    files = PretrainDataset._resolve_files(str(d))
    scheduler = WeightedLineScheduler(files, seed=0)
    counts: Counter[Path] = Counter()
    gen = scheduler.iter_lines()
    for path, _ in islice(gen, 100_000):
        counts[path] += 1
    gen.close()

    total = sum(counts.values())
    assert total == 100_000
    total_weight = sum(weight for _, weight in files)
    for path, weight in files:
        expected = weight / total_weight
        actual = counts[path] / total
        assert abs(actual - expected) <= 0.05


def _scheduled_names(files: list[tuple[Path, float]], seed: int, count: int) -> list[str]:
    scheduler = WeightedLineScheduler(files, seed=seed)
    return [path.name for path, _ in islice(scheduler.iter_lines(), count)]


def test_weighted_scheduler_deterministic(tmp_path: Path) -> None:
    d = _scheduler_dir(tmp_path, {"a": 5_000, "b": 5_000, "c": 5_000})
    files = PretrainDataset._resolve_files(str(d))
    first = _scheduled_names(files, seed=7, count=2_000)
    second = _scheduled_names(files, seed=7, count=2_000)
    assert first == second


def test_weighted_scheduler_exhausts_all_files(tmp_path: Path) -> None:
    d = _scheduler_dir(tmp_path, {"a": 3, "b": 4, "c": 5})
    files = PretrainDataset._resolve_files(str(d))
    lines = list(WeightedLineScheduler(files, seed=0).iter_lines())
    assert len(lines) == 12
    counts = Counter(path.name for path, _ in lines)
    assert counts == Counter({"a.txt": 3, "b.txt": 4, "c.txt": 5})
