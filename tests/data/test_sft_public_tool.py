from __future__ import annotations

import json
from pathlib import Path

import pytest

from kestrel.data.sft_prepare import PublicToolSourceConfig, SFTDataConfig, prepare_public_tool
from kestrel.data.sft_public_tool import PublicToolNormalizer

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sft_public_tool_fixture.jsonl"


def _load_fixture() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]


def _row_by_id(rows: list[dict[str, object]], row_id: int) -> dict[str, object]:
    return next(row for row in rows if row.get("id") == row_id)


def _normalizer(**kwargs: object) -> PublicToolNormalizer:
    return PublicToolNormalizer(source="tool_public", excluded_tool_names=frozenset(), **kwargs)


def test_fixture_accepts_openai_and_xlam_tool_formats() -> None:
    normalizer = _normalizer()
    valid_rows = [row for row in map(normalizer.convert, _load_fixture()) if row is not None]

    called_names = [
        row.messages[-1].tool_calls[0].function.name  # type: ignore[index,union-attr]
        for row in valid_rows
    ]
    assert "roll_array" in called_names
    assert "web_chain_details" in called_names
    assert "split_list" in called_names
    assert len(valid_rows) == 5

    for row in valid_rows:
        assert row.source == "tool_public"
        assert [message.role for message in row.messages] == ["system", "user", "assistant"]
        assistant = row.messages[-1]
        assert assistant.role == "assistant"
        assert assistant.content is None
        assert assistant.tool_calls is not None
        assert len(assistant.tool_calls) == 1
        assert assistant.tool_calls[0].function.name in [tool.function.name for tool in row.tools]


def test_fixture_drops_invalid_and_duplicate_rows() -> None:
    normalizer = _normalizer()
    valid_rows = [row for row in map(normalizer.convert, _load_fixture()) if row is not None]
    called_names = {
        row.messages[-1].tool_calls[0].function.name  # type: ignore[index,union-attr]
        for row in valid_rows
    }
    assert "duplicate_id_tool" not in called_names
    assert "duplicate_hash_tool" not in called_names
    assert "duplicate_query_tool" not in called_names


@pytest.mark.parametrize(
    ("row_id", "kwargs"),
    [
        (3, {}),
        (4, {}),
        (5, {}),
        (6, {}),
        (7, {}),
        (8, {}),
        (9, {}),
        (14, {}),
        (15, {}),
        (17, {"max_list_items": 2}),
    ],
)
def test_individual_invalid_rows_are_dropped(row_id: int, kwargs: dict[str, object]) -> None:
    normalizer = _normalizer(**kwargs)
    assert normalizer.convert(_row_by_id(_load_fixture(), row_id)) is None


def test_long_query_is_accepted_when_limit_is_increased() -> None:
    normalizer = _normalizer(max_query_chars=10_000)
    row = normalizer.convert(_row_by_id(_load_fixture(), 15))
    assert row is not None


def test_excluded_tool_names_are_dropped() -> None:
    normalizer = PublicToolNormalizer(
        source="tool_public",
        excluded_tool_names=frozenset({"get_weather_forecast"}),
    )
    assert normalizer.convert(_row_by_id(_load_fixture(), 13)) is None


def test_prepare_public_tool_writes_manifest(
    tmp_path: Path, tiny_sft_tokenizer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kestrel.data.sft_prepare as module

    monkeypatch.setattr(
        module,
        "load_public_tool_rows",
        lambda dataset_id, split: iter(_load_fixture()),
    )
    config = SFTDataConfig(
        output_dir=str(tmp_path),
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=8192,
        seed=7,
        public_tool=PublicToolSourceConfig(target_rows=10),
    )

    manifest = prepare_public_tool(config)

    assert manifest.source == "tool_public"
    assert manifest.requested_rows == 10
    assert manifest.written_rows == 4
    assert manifest.dropped_rows > 0
    assert manifest.output_path == str(tmp_path / "tool_public.jsonl")

    rows = [
        json.loads(line)
        for line in (tmp_path / "tool_public.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == manifest.written_rows
    assert all(row["source"] == "tool_public" for row in rows)

    manifest_data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["tool_public"]["dropped_rows"] == manifest.dropped_rows
