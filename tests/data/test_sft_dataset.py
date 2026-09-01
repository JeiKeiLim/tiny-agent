"""Tests for the example-based M2 SFT dataset."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer

from kestrel.data.sft_chat import render_sft
from kestrel.data.sft_dataset import SFTDataset, SFTDatasetConfig
from kestrel.data.sft_schema import SFTRow
from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS, TokenizerConfig
from kestrel.tokenizer.train import train

SENTENCE = "hello world the quick brown fox jumps over the lazy dog. "


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("sft_tokenizer")
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "sft.txt").write_text(SENTENCE * 500 + "hello world assistant user tool " * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp / "tok"),
        special_tokens=list(DEFAULT_SPECIAL_TOKENS),
        eos_token=DEFAULT_SPECIAL_TOKENS[1],
    )
    return train(config)


def _row(source: str, content: str = "hello world") -> dict[str, object]:
    return {
        "source": source,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": content},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fin:
        for row in rows:
            fin.write(json.dumps(row, ensure_ascii=False) + "\n")


def _config(input_path: Path, tokenizer_path: Path, **overrides: object) -> SFTDatasetConfig:
    data: dict[str, object] = {
        "input": str(input_path),
        "tokenizer_path": str(tokenizer_path),
        "context_length": 64,
        "batch_size": 2,
        "seed": 0,
        "max_examples": None,
        "preserve_source_ratios": True,
    }
    data.update(overrides)
    return SFTDatasetConfig.model_validate(data)


def _batch_key(batch: tuple[mx.array, mx.array, mx.array]) -> list[list[int]]:
    return [array.tolist() for array in batch]


def test_config_is_strict_and_supports_max_examples(tmp_path: Path, tiny_tokenizer: Path) -> None:
    with pytest.raises(ValidationError):
        SFTDatasetConfig(input="x", tokenizer_path=str(tiny_tokenizer), context_length="64")
    with pytest.raises(ValidationError):
        SFTDatasetConfig(input="x", tokenizer_path=str(tiny_tokenizer), unknown_key=1)
    with pytest.raises(ValidationError):
        SFTDatasetConfig(input="x", tokenizer_path=str(tiny_tokenizer), max_examples=0)

    config = SFTDatasetConfig(input="x", tokenizer_path=str(tiny_tokenizer), max_examples=7)
    assert config.max_examples == 7
    assert config.context_length == 1024
    assert config.preserve_source_ratios is True


def test_batches_have_expected_shape_and_target_shift(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", f"hello world {index}") for index in range(4)])
    dataset = SFTDataset(_config(data, tiny_tokenizer))

    batches = list(dataset)
    assert len(batches) == 2
    for inp, target, mask in batches:
        assert inp.shape == (2, 64)
        assert target.shape == (2, 64)
        assert mask.shape == (2, 64)
        assert inp.dtype == mx.int32
        assert target.dtype == mx.int32
        assert mask.dtype == mx.int32
        assert bool((target[:, :-1] == inp[:, 1:]).all())


def test_loss_mask_matches_renderer_and_masks_padding(tmp_path: Path, tiny_tokenizer: Path) -> None:
    row = _row("a", "hello world")
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [row])
    dataset = SFTDataset(_config(data, tiny_tokenizer, batch_size=1))

    _, _, mask = next(iter(dataset))
    rendered = render_sft(SFTRow.model_validate(row), Tokenizer.from_file(str(tiny_tokenizer)))
    expected = [*rendered.loss_mask[:-1], 0] + [0] * (64 - len(rendered.token_ids))

    assert mask[0].tolist() == expected
    assert int(mask.sum().item()) > 0


def test_tool_result_tokens_are_not_trained(tmp_path: Path, tiny_tokenizer: Path) -> None:
    row = {
        "source": "tool_local",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Seoul"}},
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "sunny"},
            {"role": "assistant", "content": "hello world"},
        ],
    }
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [row])
    dataset = SFTDataset(_config(data, tiny_tokenizer, batch_size=1, context_length=512))

    _, _, mask = next(iter(dataset))
    rendered = render_sft(SFTRow.model_validate(row), Tokenizer.from_file(str(tiny_tokenizer)))
    expected = [*rendered.loss_mask[:-1], 0] + [0] * (512 - len(rendered.token_ids))
    assert mask[0].tolist() == expected


def test_rows_longer_than_context_are_filtered(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", "hello " * 100), _row("b", "hello world")])
    dataset = SFTDataset(_config(data, tiny_tokenizer, context_length=32, batch_size=1))

    assert len(list(dataset)) == 1
    assert dataset.source_counts == {"b": 1}


def test_all_rows_filtered_yields_no_batches(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", "hello " * 100)])
    dataset = SFTDataset(_config(data, tiny_tokenizer, context_length=16, batch_size=1))

    assert list(dataset) == []
    assert dataset.source_counts == {}


def test_partial_final_batch_is_dropped(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", f"hello world {index}") for index in range(3)])
    dataset = SFTDataset(_config(data, tiny_tokenizer, batch_size=2))

    assert len(list(dataset)) == 1
    assert sum(dataset.source_counts.values()) == 3


def test_max_examples_preserves_source_ratios(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    rows = [_row("a", f"hello world {index}") for index in range(4)]
    rows.extend(_row("b", f"hello world {index}") for index in range(2))
    _write_jsonl(data, rows)

    dataset = SFTDataset(_config(data, tiny_tokenizer, max_examples=3, batch_size=1))
    assert dataset.source_counts == {"a": 2, "b": 1}


def test_max_examples_one_selects_largest_source(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    rows = [_row("a", f"hello world {index}") for index in range(2)]
    rows.extend(_row("b", f"hello world {index}") for index in range(1))
    _write_jsonl(data, rows)

    dataset = SFTDataset(_config(data, tiny_tokenizer, max_examples=1, batch_size=1))
    assert dataset.source_counts == {"a": 1}


def test_max_examples_larger_than_input_keeps_all(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    rows = [_row("a", f"hello world {index}") for index in range(2)]
    rows.extend(_row("b", f"hello world {index}") for index in range(1))
    _write_jsonl(data, rows)

    dataset = SFTDataset(_config(data, tiny_tokenizer, max_examples=100, batch_size=1))
    assert dataset.source_counts == {"a": 2, "b": 1}


def test_max_examples_without_ratio_preservation_selects_global_subset(
    tmp_path: Path, tiny_tokenizer: Path
) -> None:
    data = tmp_path / "sft.jsonl"
    rows = [_row("a", f"hello world {index}") for index in range(3)]
    rows.extend(_row("b", f"hello world {index}") for index in range(3))
    _write_jsonl(data, rows)

    dataset = SFTDataset(
        _config(data, tiny_tokenizer, max_examples=2, preserve_source_ratios=False, batch_size=1)
    )
    assert sum(dataset.source_counts.values()) == 2


def test_iteration_is_deterministic_for_same_seed(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    rows = [_row("a", f"hello world {index}") for index in range(4)]
    rows.extend(_row("b", f"hello world {index}") for index in range(2))
    _write_jsonl(data, rows)

    first = SFTDataset(_config(data, tiny_tokenizer, seed=123, batch_size=1))
    second = SFTDataset(_config(data, tiny_tokenizer, seed=123, batch_size=1))
    assert [_batch_key(batch) for batch in first] == [_batch_key(batch) for batch in second]


def test_invalid_jsonl_raises_with_line_number(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    data.write_text(json.dumps(_row("a")) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        SFTDataset(_config(data, tiny_tokenizer))


def test_invalid_schema_raises_with_line_number(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    bad_row = {"source": "a", "messages": [{"role": "assistant", "content": "hello"}]}
    data.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        SFTDataset(_config(data, tiny_tokenizer))


def test_iterator_state_resumes_at_batch_boundary(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", f"hello world {index}") for index in range(4)])
    dataset = SFTDataset(_config(data, tiny_tokenizer))

    iterator = dataset.iterator()
    first_batch = next(iterator)
    state = iterator.state_dict()

    resumed = dataset.load_iterator(state)
    remaining = list(resumed)

    fresh = list(dataset.iterator())
    assert len(remaining) == 1
    assert _batch_key(remaining[0]) == _batch_key(fresh[1])
    assert _batch_key((first_batch[0], first_batch[1], first_batch[2])) == _batch_key(fresh[0])


def test_iterator_repeats_for_epochs(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", f"hello world {index}") for index in range(4)])
    dataset = SFTDataset(_config(data, tiny_tokenizer, epochs=2))

    assert dataset.estimated_steps() == 4
    batches = list(dataset)

    assert len(batches) == 4
    assert _batch_key(batches[0]) == _batch_key(batches[2])
    assert _batch_key(batches[1]) == _batch_key(batches[3])


def test_iterator_rejects_config_mismatch(tmp_path: Path, tiny_tokenizer: Path) -> None:
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_row("a", f"hello world {index}") for index in range(4)])
    dataset = SFTDataset(_config(data, tiny_tokenizer))

    iterator = dataset.iterator()
    next(iterator)
    state = iterator.state_dict()
    state["config"]["batch_size"] = 4

    with pytest.raises(ValueError, match="different SFTDatasetConfig"):
        dataset.load_iterator(state)
