"""Tests for GSM8K SFT preparation."""

from __future__ import annotations

import json
from pathlib import Path

from kestrel.data.sft_prepare import prepare_rows
from kestrel.data.sft_prepare_gsm8k import convert_gsm8k_row
from kestrel.data.sft_schema import AssistantMessage


def _gsm8k_row() -> dict[str, object]:
    return {
        "question": "Natalia sold clips to 48 friends in April.",
        "answer": (
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
            "#### 72"
        ),
    }


def test_convert_gsm8k_row_strips_annotations_and_extracts_final_answer() -> None:
    row = convert_gsm8k_row(_gsm8k_row(), "gsm8k_math")

    assert row is not None
    assert row.source == "gsm8k_math"
    assert row.messages[0].role == "user"

    assistant = row.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.content is not None
    assert "<<" not in assistant.content
    assert ">>" not in assistant.content
    assert "Natalia sold 48/2 = 24 clips in May." in assistant.content
    assert "Final answer: 72" in assistant.content


def test_convert_gsm8k_row_rejects_invalid_rows() -> None:
    assert convert_gsm8k_row({}, "gsm8k_math") is None
    assert convert_gsm8k_row({"question": "  ", "answer": "#### 1"}, "gsm8k_math") is None
    assert convert_gsm8k_row({"question": "q", "answer": "no final marker"}, "gsm8k_math") is None
    assert (
        convert_gsm8k_row({"question": "q", "answer": "reasoning\n####   "}, "gsm8k_math") is None
    )


def test_convert_gsm8k_row_handles_missing_reasoning() -> None:
    row = convert_gsm8k_row({"question": "What is 1+1?", "answer": "#### 2"}, "gsm8k_math")

    assert row is not None
    assistant = row.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.content == "Final answer: 2"


def test_prepare_gsm8k_rows_writes_jsonl_and_manifest(
    tmp_path: Path, tiny_sft_tokenizer: Path
) -> None:
    rows = [_gsm8k_row() for _ in range(3)]

    manifest = prepare_rows(
        source="gsm8k_math",
        dataset_id="fixture/gsm8k",
        split="train",
        seed=11,
        target_rows=10,
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=1024,
        output_dir=str(tmp_path),
        load_rows=lambda: iter(rows),
        convert_row=lambda raw: convert_gsm8k_row(raw, "gsm8k_math"),
        dataset_config="main",
    )

    assert manifest.written_rows == 3
    assert manifest.candidate_rows == 3
    assert manifest.requested_rows == 10
    assert manifest.dataset_config == "main"

    lines = (tmp_path / "gsm8k_math.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert all(item["source"] == "gsm8k_math" for item in parsed)

    manifest_data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest_data["gsm8k_math"]
    assert entry["sha256"] == manifest.sha256
    assert entry["dataset_config"] == "main"


def test_prepare_gsm8k_rows_filters_long_rows(tmp_path: Path, tiny_sft_tokenizer: Path) -> None:
    rows = [
        {
            "question": "hello " * 300,
            "answer": "hello\n#### 1",
        },
        _gsm8k_row(),
    ]

    manifest = prepare_rows(
        source="gsm8k_math",
        dataset_id="fixture/gsm8k",
        split="train",
        seed=11,
        target_rows=2,
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=256,
        output_dir=str(tmp_path),
        load_rows=lambda: iter(rows),
        convert_row=lambda raw: convert_gsm8k_row(raw, "gsm8k_math"),
        dataset_config="main",
    )

    assert manifest.written_rows == 1
    assert manifest.filtered_rows == 1
