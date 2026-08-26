"""Tests for the pretraining dataset (src/kestrel/data/pretrain_dataset.py).

A tiny BPE tokenizer is trained once per test module (same pattern as
test_model_check.py) so the tests do not depend on the gitignored
checkpoints/tokenizer/tokenizer.json.
"""

import json
import random
from pathlib import Path

import mlx.core as mx
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer

from kestrel.data.pretrain_dataset import (
    PretrainDataset,
    PretrainDatasetConfig,
    _resolve_sources,
    choose_deficit_source,
)
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train

SENTENCE = "hello world the quick brown fox jumps over the lazy dog. "


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tokenizer")
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(SENTENCE * 500 + "the quick brown fox jumps " * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp / "tok"),
        special_tokens=["im_start", "im_end"],
        eos_token="im_end",
    )
    return train(config)


def _write_jsonl(path: Path, docs: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"domain": path.stem, "text": doc}, ensure_ascii=False) + "\n")


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


def test_batch_shape(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [SENTENCE * 2] * 5)
    batches = list(PretrainDataset(_config(data, tok)))
    assert len(batches) >= 2
    for inp, tgt, docs in batches:
        assert inp.shape == (2, 8)
        assert tgt.shape == (2, 8)
        assert docs.shape == (2, 8)
        assert inp.dtype == mx.int32
        assert tgt.dtype == mx.int32
        assert docs.dtype == mx.int32


def test_next_token_shift(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    text = SENTENCE * 5
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [text])
    inp, tgt, docs = next(iter(PretrainDataset(_config(data, tok))))
    assert bool((tgt[:, :-1] == inp[:, 1:]).all())

    decoder = Tokenizer.from_file(str(tok))
    im_start = decoder.token_to_id("im_start")
    assert im_start is not None
    expected = [im_start, *decoder.encode(text, add_special_tokens=False).ids[:7]]
    assert inp[0].tolist() == expected
    assert docs[0].tolist() == [0] * 8


def test_total_tokens_cap(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [SENTENCE * 20])
    capped = list(PretrainDataset(_config(data, tok, total_tokens=16)))
    assert len(capped) == 1
    uncapped = list(PretrainDataset(_config(data, tok)))
    assert len(uncapped) > 1


def test_config_rejects_bad_values(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    with pytest.raises(ValidationError):
        PretrainDatasetConfig(input="x", tokenizer_path=str(tok), context_length="8")
    with pytest.raises(ValidationError):
        PretrainDatasetConfig(input="x", tokenizer_path=str(tok), unknown_key=1)


def test_dir_input(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    d = tmp_path / "data"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", [SENTENCE * 10])
    _write_jsonl(d / "b.jsonl", [SENTENCE * 10])
    batches = list(PretrainDataset(_config(d, tok)))
    assert len(batches) >= 1
    for inp, _, _ in batches:
        assert inp.shape == (2, 8)


def test_full_batches_only(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [SENTENCE * 3])
    total = sum(inp.shape[0] * inp.shape[1] for inp, _, _ in PretrainDataset(_config(data, tok)))
    assert total % (2 * 8) == 0


def test_multi_file_batches_mix_domains(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    d = tmp_path / "data"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", ["alpha " * 200])
    _write_jsonl(d / "b.jsonl", ["beta " * 200])
    dataset = PretrainDataset(_config(d, tok, batch_size=1))
    decoder = Tokenizer.from_file(str(tok))
    decoded = " ".join(decoder.decode(inp[0].tolist()) for inp, _, _ in dataset)
    assert "alpha" in decoded
    assert "beta" in decoded


def test_multi_domain_directory_has_multiple_doc_ids(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    d = tmp_path / "data"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", ["alpha " * 20])
    _write_jsonl(d / "b.jsonl", ["beta " * 20])
    dataset = PretrainDataset(_config(d, tok, batch_size=1, context_length=8))
    doc_ids: set[int] = set()
    for _, _, docs in dataset:
        for row in docs.tolist():
            doc_ids.update(row)
    assert len(doc_ids) >= 2


def test_manifest_token_counts_set_target_fractions(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", ["alpha"])
    _write_jsonl(d / "b.jsonl", ["beta"])
    manifest = {
        "split": "train",
        "files": [
            {
                "path": "a.jsonl",
                "domain": "a",
                "doc_count": 1,
                "byte_count": 10,
                "target_fraction": 0.0,
                "token_count": 900,
                "estimated_token_count": None,
            },
            {
                "path": "b.jsonl",
                "domain": "b",
                "doc_count": 1,
                "byte_count": 10,
                "target_fraction": 0.0,
                "token_count": 100,
                "estimated_token_count": None,
            },
        ],
        "total_doc_count": 2,
        "total_byte_count": 20,
        "total_token_count": 1000,
        "total_estimated_token_count": None,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sources, info = _resolve_sources(str(d))
    assert [source.target_fraction for source in sources] == pytest.approx([0.9, 0.1])
    assert info.total_text_tokens == 1000
    assert info.total_docs == 2


def test_single_file_preserves_document_order(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    words = [f"marker{index:02d}" for index in range(10)]
    text = (" ".join(words) + " ") * 10
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [text])
    inp, _, _ = next(iter(PretrainDataset(_config(data, tok, batch_size=1, context_length=256))))
    decoded = Tokenizer.from_file(str(tok)).decode(inp[0].tolist())
    positions = [decoded.index(word) for word in words]
    assert positions == sorted(positions)


def test_multiline_document_keeps_one_doc_id(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [("alpha\nbeta ") * 20])
    dataset = PretrainDataset(_config(data, tok, batch_size=1, context_length=8))
    decoder = Tokenizer.from_file(str(tok))
    decoded_parts: list[str] = []
    doc_ids: set[int] = set()
    for inp, _, docs in dataset:
        decoded_parts.append(decoder.decode(inp[0].tolist()))
        for row in docs.tolist():
            doc_ids.update(row)
    decoded = " ".join(decoded_parts)
    assert "alpha" in decoded
    assert "beta" in decoded
    assert doc_ids == {0}


def test_multiple_documents_get_distinct_doc_ids(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, ["alpha " * 20, "beta " * 20])
    dataset = PretrainDataset(_config(data, tok, batch_size=1, context_length=8))
    decoder = Tokenizer.from_file(str(tok))
    decoded_parts: list[str] = []
    doc_ids: set[int] = set()
    for inp, _, docs in dataset:
        decoded_parts.append(decoder.decode(inp[0].tolist()))
        for row in docs.tolist():
            doc_ids.update(row)
    decoded = " ".join(decoded_parts)
    assert "alpha" in decoded
    assert "beta" in decoded
    assert len(doc_ids) >= 2


def _scheduler_order(
    seed: int,
    fractions: list[float],
    lengths: list[int],
    count: int,
    remaining: list[int] | None = None,
) -> list[int]:
    active = list(range(len(fractions)))
    emitted = [0.0] * len(fractions)
    remaining = list(remaining) if remaining is not None else [count] * len(fractions)
    rng = random.Random(seed)
    order: list[int] = []
    while active and len(order) < count:
        index = choose_deficit_source(active, fractions, emitted, rng)
        order.append(index)
        emitted[index] += lengths[index]
        remaining[index] -= 1
        if remaining[index] == 0:
            active.remove(index)
    return order


def test_token_deficit_scheduler_share_within_tolerance() -> None:
    fractions = [0.5, 0.5]
    lengths = [100, 1]
    order = _scheduler_order(0, fractions, lengths, count=10_000)
    counts = [order.count(0), order.count(1)]
    tokens = [counts[0] * lengths[0], counts[1] * lengths[1]]
    total = sum(tokens)
    for actual, target in zip(tokens, fractions, strict=True):
        assert abs(actual / total - target) <= 0.05


def test_token_deficit_scheduler_deterministic() -> None:
    fractions = [0.5, 0.5]
    lengths = [7, 3]
    first = _scheduler_order(7, fractions, lengths, count=1_000)
    second = _scheduler_order(7, fractions, lengths, count=1_000)
    assert first == second


def test_token_deficit_scheduler_exhausts_all_sources() -> None:
    fractions = [1.0, 1.0, 1.0]
    lengths = [1, 1, 1]
    order = _scheduler_order(0, fractions, lengths, count=12, remaining=[3, 4, 5])
    counts = [order.count(0), order.count(1), order.count(2)]
    assert counts == [3, 4, 5]


def test_estimated_steps_from_manifest(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    d = tmp_path / "data"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", ["alpha"] * 10)
    _write_jsonl(d / "b.jsonl", ["beta"] * 5)
    manifest = {
        "split": "train",
        "files": [
            {
                "path": "a.jsonl",
                "domain": "a",
                "doc_count": 10,
                "byte_count": 100,
                "target_fraction": 0.5,
                "token_count": 600,
                "estimated_token_count": None,
            },
            {
                "path": "b.jsonl",
                "domain": "b",
                "doc_count": 5,
                "byte_count": 50,
                "target_fraction": 0.5,
                "token_count": 400,
                "estimated_token_count": None,
            },
        ],
        "total_doc_count": 15,
        "total_byte_count": 150,
        "total_token_count": 1000,
        "total_estimated_token_count": None,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = PretrainDataset(_config(d, tok, batch_size=2, context_length=8))
    assert dataset.estimated_steps() == (1000 + 2 * 15) // (2 * 8)


def test_estimated_steps_uses_total_tokens_cap(tmp_path: Path, tiny_tokenizer: Path) -> None:
    tok = tiny_tokenizer
    d = tmp_path / "data"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", ["alpha"] * 10)
    manifest = {
        "split": "train",
        "files": [
            {
                "path": "a.jsonl",
                "domain": "a",
                "doc_count": 10,
                "byte_count": 100,
                "target_fraction": 1.0,
                "token_count": 1000,
                "estimated_token_count": None,
            }
        ],
        "total_doc_count": 10,
        "total_byte_count": 100,
        "total_token_count": 1000,
        "total_estimated_token_count": None,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = PretrainDataset(_config(d, tok, batch_size=2, context_length=8, total_tokens=100))
    assert dataset.estimated_steps() == 100 // (2 * 8)
