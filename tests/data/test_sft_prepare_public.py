"""Tests for public assistant SFT preparation (Smol-SmolTalk)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.data.sft_prepare import (
    SFTDataConfig,
    SourceManifest,
    prepare_rows,
    reservoir_sample,
)
from kestrel.data.sft_prepare_public import convert_smol_row
from kestrel.data.sft_schema import SFTRow


def _smol_row(index: int, assistant_content: str = "hello world") -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def test_convert_smol_row_valid() -> None:
    raw = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello world"},
        ]
    }

    row = convert_smol_row(raw, "assistant_public")

    assert row is not None
    assert row.source == "assistant_public"
    assert [message.role for message in row.messages] == ["system", "user", "assistant"]


def test_convert_smol_row_rejects_invalid_rows() -> None:
    assert convert_smol_row({}, "assistant_public") is None
    assert convert_smol_row({"messages": []}, "assistant_public") is None
    assert (
        convert_smol_row(
            {"messages": [{"role": "tool", "content": "hello"}]},
            "assistant_public",
        )
        is None
    )
    assert (
        convert_smol_row(
            {"messages": [{"role": "user", "content": "   "}]},
            "assistant_public",
        )
        is None
    )
    assert (
        convert_smol_row(
            {"messages": [{"role": "user", "content": "hello"}]},
            "assistant_public",
        )
        is None
    )
    assert (
        convert_smol_row(
            {"messages": [{"role": "assistant", "content": "hello"}]},
            "assistant_public",
        )
        is None
    )


def test_reservoir_sample_is_deterministic_and_bounded() -> None:
    rows = [{"index": i} for i in range(10)]

    first = reservoir_sample(iter(rows), 3, seed=123)
    second = reservoir_sample(iter(rows), 3, seed=123)

    assert first == second
    assert len(first) == 3
    assert all(0 <= row["index"] < 10 for row in first)


def test_reservoir_sample_target_larger_than_stream_keeps_all() -> None:
    rows = [{"index": i} for i in range(4)]
    assert len(reservoir_sample(iter(rows), 100, seed=0)) == 4


def test_prepare_rows_writes_jsonl_and_manifest(tmp_path: Path, tiny_sft_tokenizer: Path) -> None:
    rows = [_smol_row(index) for index in range(4)]

    manifest = prepare_rows(
        source="assistant_public",
        dataset_id="fixture/smol",
        split="train",
        seed=7,
        target_rows=4,
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=64,
        output_dir=str(tmp_path),
        load_rows=lambda: iter(rows),
        convert_row=lambda raw: convert_smol_row(raw, "assistant_public"),
    )

    assert manifest.written_rows == 4
    assert manifest.candidate_rows == 4
    assert manifest.filtered_rows == 0

    output_path = tmp_path / "assistant_public.jsonl"
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert all(item["source"] == "assistant_public" for item in parsed)

    manifest_data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest_data["assistant_public"]
    assert entry["sha256"] == manifest.sha256
    assert entry["seed"] == 7
    assert entry["dataset_id"] == "fixture/smol"


def test_prepare_rows_filters_rows_longer_than_context(
    tmp_path: Path, tiny_sft_tokenizer: Path
) -> None:
    rows = [
        _smol_row(0, assistant_content="hello " * 100),
        _smol_row(1, assistant_content="hello world"),
    ]

    manifest = prepare_rows(
        source="assistant_public",
        dataset_id="fixture/smol",
        split="train",
        seed=7,
        target_rows=2,
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=32,
        output_dir=str(tmp_path),
        load_rows=lambda: iter(rows),
        convert_row=lambda raw: convert_smol_row(raw, "assistant_public"),
    )

    assert manifest.written_rows == 1
    assert manifest.filtered_rows == 1


def test_sft_data_config_is_strict_and_yaml_loads() -> None:
    with pytest.raises(ValidationError):
        SFTDataConfig(context_length="1024")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SFTDataConfig(bogus=1)  # type: ignore[call-arg]

    config = load_config("configs/kestrel/sft_data.yaml", SFTDataConfig)
    assert config.output_dir == "data/sft/raw"
    assert config.context_length == 1024
    assert config.assistant.target_rows == 22_500
    assert config.gsm8k.target_rows == 7_500
    assert config.gsm8k.dataset_config == "main"


def test_sftrow_round_trips_prepared_public_row() -> None:
    row = convert_smol_row(_smol_row(0), "assistant_public")
    assert row is not None

    round_tripped = SFTRow.model_validate(row.model_dump(mode="json"))
    assert round_tripped.source == "assistant_public"
    assert len(round_tripped.messages) == 2


def test_run_prepare_sft_cli_selects_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "kestrel_run_prepare_sft", "scripts/run_prepare_sft.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake_manifest = SourceManifest(
        source="assistant_public",
        dataset_id="fixture/smol",
        split="train",
        seed=0,
        requested_rows=1,
        candidate_rows=1,
        written_rows=1,
        filtered_rows=0,
        output_path="assistant_public.jsonl",
        sha256="abc123",
    )
    calls: list[str] = []

    monkeypatch.setattr(module, "load_config", lambda path, config_type: SFTDataConfig())
    monkeypatch.setattr(
        module,
        "prepare_assistant",
        lambda config: calls.append("assistant") or fake_manifest,
    )
    monkeypatch.setattr(
        module,
        "prepare_gsm8k",
        lambda config: calls.append("gsm8k") or fake_manifest,
    )
    monkeypatch.setattr(module, "prepare_all", lambda config: {"assistant_public": fake_manifest})
    monkeypatch.setattr(sys, "argv", ["run_prepare_sft.py", "--source", "assistant"])

    module.main()

    assert calls == ["assistant"]
    assert "assistant_public: wrote 1/1 rows" in capsys.readouterr().out
