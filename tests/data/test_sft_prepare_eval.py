"""Tests for held-out SFT eval bundle preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kestrel.common.config import load_config
from kestrel.data.sft_prepare import (
    SFTDataConfig,
    SFTDataEvalConfig,
    SourceManifest,
    ToolSourceConfig,
)
from kestrel.data.sft_prepare_eval import prepare_eval
from kestrel.data.sft_tool_generator import ToolEvalBreakdown
from kestrel.tools.schema_sampler import TRAIN_TOOL_FAMILIES
from kestrel.train.checkpoint import sha256_file


def _smol_row(index: int, assistant_content: str = "hello world") -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"hello {index}"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _gsm8k_row(question: str = "What is 1+1?") -> dict[str, object]:
    return {"question": question, "answer": "1+1 = 2\n#### 2"}


def _eval_config(tmp_path: Path, tiny_sft_tokenizer: Path) -> SFTDataConfig:
    return SFTDataConfig(
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=1024,
        tool=ToolSourceConfig(eval=ToolEvalBreakdown(seen=2, unseen=2, no_call=1, missing_info=1)),
        eval=SFTDataEvalConfig(
            output_dir=str(tmp_path / "eval"),
            assistant_target_rows=2,
            assistant_max_candidate_rows=10,
            gsm8k_target_rows=2,
        ),
    )


def test_sft_data_eval_config_yaml_loads() -> None:
    config = load_config("configs/kestrel/sft_data.yaml", SFTDataConfig)
    eval_config = config.eval
    assert eval_config is not None
    assert eval_config.output_dir == "data/sft/eval"
    assert eval_config.assistant_split == "test"
    assert eval_config.assistant_target_rows == 200
    assert eval_config.gsm8k_split == "test"
    assert eval_config.gsm8k_dataset_config == "main"
    assert eval_config.gsm8k_target_rows == 500
    assert eval_config.tool_eval is True


def test_prepare_eval_writes_eval_bundle(
    tmp_path: Path,
    tiny_sft_tokenizer: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.data.sft_prepare_eval as eval_module

    monkeypatch.setattr(
        eval_module,
        "load_smol_rows",
        lambda dataset_id, split, seed: iter([_smol_row(index) for index in range(4)]),
    )
    monkeypatch.setattr(
        eval_module,
        "load_gsm8k_rows",
        lambda dataset_id, dataset_config, split, seed: iter([_gsm8k_row() for _ in range(4)]),
    )
    config = _eval_config(tmp_path, tiny_sft_tokenizer)

    manifests = prepare_eval(config)

    assert set(manifests) == {
        "assistant_eval",
        "gsm8k_eval",
        "tool_eval_seen",
        "tool_eval_unseen",
        "tool_eval_no_call",
        "tool_eval_missing_info",
    }
    assert manifests["assistant_eval"].written_rows == 2
    assert manifests["gsm8k_eval"].written_rows == 2
    assert manifests["tool_eval_seen"].written_rows == 2
    assert manifests["tool_eval_unseen"].written_rows == 2
    assert manifests["tool_eval_no_call"].written_rows == 1
    assert manifests["tool_eval_missing_info"].written_rows == 1

    eval_dir = tmp_path / "eval"
    for manifest in manifests.values():
        output_path = Path(manifest.output_path)
        assert output_path.is_file()
        assert sha256_file(output_path) == manifest.sha256

    manifest_data = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest_data) == set(manifests)
    for source, manifest in manifests.items():
        assert manifest_data[source]["sha256"] == manifest.sha256
        assert manifest_data[source]["written_rows"] == manifest.written_rows


def test_prepare_eval_continues_past_filtered_rows_until_target(
    tmp_path: Path,
    tiny_sft_tokenizer: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.data.sft_prepare_eval as eval_module

    monkeypatch.setattr(
        eval_module,
        "load_smol_rows",
        lambda dataset_id, split, seed: iter(
            [
                _smol_row(0, assistant_content="hello " * 500),
                _smol_row(1),
                _smol_row(2, assistant_content="hello " * 500),
                _smol_row(3),
            ]
        ),
    )
    monkeypatch.setattr(
        eval_module,
        "load_gsm8k_rows",
        lambda dataset_id, dataset_config, split, seed: iter(
            [
                _gsm8k_row(question="hello " * 300),
                _gsm8k_row(),
            ]
        ),
    )
    config = _eval_config(tmp_path, tiny_sft_tokenizer)
    config.context_length = 256

    manifests = prepare_eval(config)

    assert manifests["assistant_eval"].written_rows == 2
    assert manifests["assistant_eval"].candidate_rows == 4
    assert manifests["assistant_eval"].filtered_rows == 2
    assert manifests["gsm8k_eval"].written_rows == 1
    assert manifests["gsm8k_eval"].filtered_rows == 1


def test_prepare_eval_tool_unseen_names_are_disjoint_from_train(
    tmp_path: Path,
    tiny_sft_tokenizer: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.data.sft_prepare_eval as eval_module

    monkeypatch.setattr(
        eval_module,
        "load_smol_rows",
        lambda dataset_id, split, seed: iter([_smol_row(index) for index in range(2)]),
    )
    monkeypatch.setattr(
        eval_module,
        "load_gsm8k_rows",
        lambda dataset_id, dataset_config, split, seed: iter([_gsm8k_row() for _ in range(2)]),
    )
    config = _eval_config(tmp_path, tiny_sft_tokenizer)

    manifests = prepare_eval(config)

    unseen_path = Path(manifests["tool_eval_unseen"].output_path)
    unseen_names: set[str] = set()
    for line in unseen_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        unseen_names.update(tool["function"]["name"] for tool in row["tools"])

    train_names = {name for family in TRAIN_TOOL_FAMILIES for name in family.tool_names}
    assert unseen_names
    assert unseen_names.isdisjoint(train_names)


def test_run_prepare_sft_cli_selects_eval_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "kestrel_run_prepare_sft_eval", "scripts/run_prepare_sft.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake_manifest = SourceManifest(
        source="assistant_eval",
        dataset_id="fixture/smol",
        split="test",
        seed=0,
        requested_rows=1,
        candidate_rows=1,
        written_rows=1,
        filtered_rows=0,
        output_path="assistant_eval.jsonl",
        sha256="abc123",
    )
    calls: list[str] = []

    monkeypatch.setattr(module, "load_config", lambda path, config_type: SFTDataConfig())
    monkeypatch.setattr(
        module,
        "prepare_eval",
        lambda config: calls.append("eval") or {"assistant_eval": fake_manifest},
    )
    monkeypatch.setattr(sys, "argv", ["run_prepare_sft.py", "--source", "eval"])

    module.main()

    assert calls == ["eval"]
    assert "assistant_eval: wrote 1/1 rows" in capsys.readouterr().out
