from __future__ import annotations

import json
from pathlib import Path

from kestrel.data.sft_prepare import SFTDataConfig, ToolSourceConfig, prepare_tool
from kestrel.data.sft_tool_generator import ToolEvalBreakdown, ToolTrainBreakdown


def _config(output_dir: Path, tokenizer_path: Path) -> SFTDataConfig:
    return SFTDataConfig(
        output_dir=str(output_dir),
        tokenizer_path=str(tokenizer_path),
        context_length=2048,
        seed=3,
        tool=ToolSourceConfig(
            train=ToolTrainBreakdown(
                direct=1,
                no_tool=1,
                distractor_heavy=1,
                missing_info=1,
                hard_variation=1,
            ),
            eval=ToolEvalBreakdown(seen=1, unseen=1, no_call=1, missing_info=1),
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_tool_writes_train_and_eval_splits(
    tmp_path: Path, tiny_sft_tokenizer: Path
) -> None:
    config = _config(tmp_path, tiny_sft_tokenizer)
    manifests = prepare_tool(config)

    assert set(manifests) == {
        "tool_local",
        "tool_eval_seen",
        "tool_eval_unseen",
        "tool_eval_no_call",
        "tool_eval_missing_info",
    }

    train_manifest = manifests["tool_local"]
    assert train_manifest.requested_rows == 5
    assert train_manifest.written_rows == 5
    assert train_manifest.filtered_rows == 0
    assert train_manifest.output_path == str(tmp_path / "tool_local.jsonl")

    train_rows = _read_jsonl(tmp_path / "tool_local.jsonl")
    assert len(train_rows) == 5
    assert all(row["source"] == "tool_local" for row in train_rows)

    for source in (
        "tool_eval_seen",
        "tool_eval_unseen",
        "tool_eval_no_call",
        "tool_eval_missing_info",
    ):
        manifest = manifests[source]
        assert manifest.written_rows == 1
        rows = _read_jsonl(tmp_path / f"{source}.jsonl")
        assert len(rows) == 1
        assert rows[0]["source"] == source


def test_prepare_tool_is_deterministic(tmp_path: Path, tiny_sft_tokenizer: Path) -> None:
    first = prepare_tool(_config(tmp_path / "first", tiny_sft_tokenizer))
    second = prepare_tool(_config(tmp_path / "second", tiny_sft_tokenizer))
    assert first["tool_local"].sha256 == second["tool_local"].sha256
    assert first["tool_eval_unseen"].sha256 == second["tool_eval_unseen"].sha256


def test_prepare_tool_updates_manifest(tmp_path: Path, tiny_sft_tokenizer: Path) -> None:
    prepare_tool(_config(tmp_path, tiny_sft_tokenizer))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "tool_local" in manifest
    assert "tool_eval_unseen" in manifest
