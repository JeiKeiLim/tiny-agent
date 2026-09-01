"""Tests for the SFT eval harness and scorecard (src/kestrel/eval/sft.py)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from kestrel.common.config import load_config
from kestrel.corpus.builder import build as build_corpus
from kestrel.corpus.config import ComponentConfig, CorpusConfig, LocalSourceConfig
from kestrel.data.sft_schema import ToolDefinition
from kestrel.eval.sft import (
    SFTCheckpointEntry,
    SFTEvalConfig,
    _extract_final_number,
    _has_obvious_repetition,
    _perplexity_metrics,
    evaluate_sft,
    write_scorecard,
)
from kestrel.eval.tool_calling import parse_generated_tool_call
from kestrel.model.config import ModelConfig
from kestrel.model.io import save as save_model
from kestrel.model.kestrel import Kestrel
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train as train_tokenizer
from kestrel.train.pretrain import PretrainConfig
from kestrel.train.trainer import TrainerConfig

BASE = "the quick brown fox jumps over the lazy dog. "
DOMAINS = ("web", "code", "synthetic")


def _tiny_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "tok_corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(BASE * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp_path / "tok"),
        special_tokens=[
            "im_start",
            "im_end",
            "im_system",
            "im_user",
            "im_assistant",
            "tool_call",
            "tool_call_end",
            "tool_response",
            "tool_response_end",
        ],
        eos_token="im_end",
    )
    return train_tokenizer(config)


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        name="tiny",
        vocab_size=400,
        context_length=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_size=64,
        intermediate_size=128,
    )


def _write_yaml(path: Path, obj: object) -> None:
    path.write_text(yaml.safe_dump(obj), encoding="utf-8")


def _tiny_corpus(tmp_path: Path) -> CorpusConfig:
    src_dir = tmp_path / "corpus_src"
    src_dir.mkdir()
    fractions = {"web": 0.4, "code": 0.3, "synthetic": 0.3}
    for domain in DOMAINS:
        lines = [f"{domain} line {i}: {BASE}" for i in range(200)]
        (src_dir / f"{domain}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = CorpusConfig(
        total_bytes=100_000,
        seed=0,
        output_dir=str(tmp_path / "corpus_out"),
        val_fraction=0.2,
        test_fraction=0.0,
        min_component_fill=0.0,
        components=[
            ComponentConfig(
                name=domain,
                source=LocalSourceConfig(type="local", path=str(src_dir / f"{domain}.txt")),
                fraction=fractions[domain],
            )
            for domain in DOMAINS
        ],
    )
    build_corpus(config)
    return config


def _tiny_pretrain_config(
    tmp_path: Path, tokenizer_path: Path, corpus_config: CorpusConfig
) -> tuple[PretrainConfig, Path]:
    model_yaml = tmp_path / "model.yaml"
    _write_yaml(model_yaml, _tiny_model_config().model_dump())
    corpus_yaml = tmp_path / "corpus.yaml"
    _write_yaml(corpus_yaml, corpus_config.model_dump())

    config = PretrainConfig(
        model=str(model_yaml),
        tokenizer=str(tokenizer_path),
        corpus=str(corpus_yaml),
        total_tokens=None,
        trainer=TrainerConfig(
            lr=1e-3,
            seq_len=16,
            batch_size=2,
            num_steps=1,
            warmup_steps=1,
            output_dir=str(tmp_path / "unused_ckpt"),
        ),
    )
    pretrain_yaml = tmp_path / "pretrain.yaml"
    pretrain_dump = config.model_dump()
    pretrain_dump["trainer"].pop("betas")
    _write_yaml(pretrain_yaml, pretrain_dump)
    return config, pretrain_yaml


def _weights_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "weights_only"
    save_model(Kestrel(_tiny_model_config()), checkpoint)
    return checkpoint


def _tool_definition(name: str = "get_weather") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def _write_eval_bundle(eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    assistant_rows = [
        {
            "source": "assistant_eval",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hello world"},
            ],
        },
        {
            "source": "assistant_eval",
            "messages": [
                {"role": "user", "content": "what"},
                {"role": "assistant", "content": "world hello"},
            ],
        },
    ]
    math_rows = [
        {
            "source": "gsm8k_eval",
            "messages": [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "1+1=2\nFinal answer: 2"},
            ],
        }
    ]
    tool_row = {
        "source": "tool_eval_seen",
        "tools": [_tool_definition()],
        "messages": [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Weather in Seoul?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Seoul"}},
                    }
                ],
            },
        ],
    }
    unseen_tool_row = json.loads(json.dumps(tool_row))
    unseen_tool_row["source"] = "tool_eval_unseen"
    no_call_row = {
        "source": "tool_eval_no_call",
        "tools": [_tool_definition()],
        "messages": [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ],
    }
    missing_info_row = {
        "source": "tool_eval_missing_info",
        "tools": [_tool_definition()],
        "messages": [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Which city?"},
        ],
    }

    files = {
        "assistant_eval.jsonl": assistant_rows,
        "gsm8k_eval.jsonl": math_rows,
        "tool_eval_seen.jsonl": [tool_row],
        "tool_eval_unseen.jsonl": [unseen_tool_row],
        "tool_eval_no_call.jsonl": [no_call_row],
        "tool_eval_missing_info.jsonl": [missing_info_row],
    }
    for filename, rows in files.items():
        with (eval_dir / filename).open("w", encoding="utf-8") as fin:
            for row in rows:
                fin.write(json.dumps(row) + "\n")


def _eval_config_yaml(
    tmp_path: Path,
    *,
    tokenizer_path: Path,
    weights: Path,
    pretrain_yaml: Path,
    eval_dir: Path,
    output: Path,
    checkpoints: list[dict[str, str]] | None = None,
    perplexity_enabled: bool = False,
) -> Path:
    config = {
        "model": str(tmp_path / "model.yaml"),
        "tokenizer": str(tokenizer_path),
        "checkpoints": checkpoints or [{"name": "pretrain", "path": str(weights)}],
        "data": {"dir": str(eval_dir), "max_rows_per_set": 1},
        "generation": {
            "max_tokens": 8,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "progress_every": 0,
        },
        "perplexity": {
            "enabled": perplexity_enabled,
            "pretrain_config": str(pretrain_yaml),
            "split": "val",
            "max_tokens": 32,
        },
        "output": str(output),
        "on_missing_checkpoint": "skip",
    }
    path = tmp_path / "eval_sft.yaml"
    _write_yaml(path, config)
    return path


@pytest.fixture(scope="module")
def eval_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("eval_sft")
    tokenizer_path = _tiny_tokenizer(tmp_path)
    corpus_config = _tiny_corpus(tmp_path)
    _, pretrain_yaml = _tiny_pretrain_config(tmp_path, tokenizer_path, corpus_config)
    weights = _weights_checkpoint(tmp_path)
    eval_dir = tmp_path / "eval"
    _write_eval_bundle(eval_dir)
    output = tmp_path / "scorecard.json"
    config_yaml = _eval_config_yaml(
        tmp_path,
        tokenizer_path=tokenizer_path,
        weights=weights,
        pretrain_yaml=pretrain_yaml,
        eval_dir=eval_dir,
        output=output,
    )
    return {
        "tmp_path": tmp_path,
        "tokenizer_path": tokenizer_path,
        "pretrain_yaml": pretrain_yaml,
        "weights": weights,
        "eval_dir": eval_dir,
        "output": output,
        "config_yaml": config_yaml,
    }


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("kestrel_run_eval_sft", "scripts/run_eval_sft.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_generated_tool_call_accepts_valid_call() -> None:
    tools = [ToolDefinition.model_validate(_tool_definition())]
    text = 'tool_call\n{"name":"get_weather","arguments":{"city":"Seoul"}}\ntool_call_end'
    parsed = parse_generated_tool_call(text, tools)
    assert parsed.attempted is True
    assert parsed.valid_json is True
    assert parsed.schema_valid is True
    assert parsed.name == "get_weather"
    assert parsed.arguments == {"city": "Seoul"}


def test_parse_generated_tool_call_reports_invalid_json() -> None:
    tools = [ToolDefinition.model_validate(_tool_definition())]
    parsed = parse_generated_tool_call("tool_call\nnot-json\ntool_call_end", tools)
    assert parsed.attempted is True
    assert parsed.valid_json is False
    assert parsed.schema_valid is False


def test_parse_generated_tool_call_reports_unknown_tool() -> None:
    tools = [ToolDefinition.model_validate(_tool_definition())]
    text = 'tool_call\n{"name":"forecast","arguments":{"city":"Seoul"}}\ntool_call_end'
    parsed = parse_generated_tool_call(text, tools)
    assert parsed.valid_json is True
    assert parsed.schema_valid is False
    assert parsed.name == "forecast"


def test_extract_final_number_prefers_final_answer() -> None:
    assert _extract_final_number("Final answer: 1,234") == 1234.0
    assert _extract_final_number("The total is 5\nFinal answer: 7") == 7.0
    assert _extract_final_number("no number") is None


def test_has_obvious_repetition_detects_degenerate_text() -> None:
    assert _has_obvious_repetition("hello " * 20) is True
    assert _has_obvious_repetition("the quick brown fox jumps over the lazy dog") is False


def test_committed_eval_sft_config_loads() -> None:
    config = load_config("configs/kestrel/50m/eval_sft.yaml", SFTEvalConfig)
    assert [entry.name for entry in config.checkpoints] == [
        "pretrain",
        "sft_5k",
        "sft_20k",
        "sft_50k",
    ]
    assert config.data.dir == "data/sft/eval"
    assert config.on_missing_checkpoint == "skip"


def test_evaluate_sft_produces_scorecard(eval_env: dict[str, Any]) -> None:
    config = load_config(eval_env["config_yaml"], SFTEvalConfig)
    scorecard = evaluate_sft(config)

    assert scorecard.format_version == 1
    result = scorecard.checkpoints[0]
    assert result.status == "ok"
    assert result.error is None
    assert result.assistant is not None and result.assistant.rows == 1
    assert result.math is not None and result.math.rows == 1
    assert result.tool is not None
    assert result.tool.seen.rows == 1
    assert result.tool.unseen.rows == 1
    assert result.tool.no_call.rows == 1
    assert result.tool.missing_info.rows == 1

    output_path = write_scorecard(scorecard, eval_env["tmp_path"] / "written_scorecard.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["checkpoints"][0]["name"] == "pretrain"
    assert payload["checkpoints"][0]["math"]["rows"] == 1


def test_evaluate_sft_is_read_only(eval_env: dict[str, Any]) -> None:
    checkpoint = eval_env["weights"]

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in checkpoint.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    config = load_config(eval_env["config_yaml"], SFTEvalConfig)
    evaluate_sft(config)
    assert snapshot() == before


def test_scorecard_compares_multiple_checkpoints(eval_env: dict[str, Any]) -> None:
    sft_checkpoint = eval_env["tmp_path"] / "sft_checkpoint"
    save_model(Kestrel(_tiny_model_config()), sft_checkpoint)

    config = load_config(eval_env["config_yaml"], SFTEvalConfig)
    config.checkpoints = [
        SFTCheckpointEntry(name="pretrain", path=str(eval_env["weights"])),
        SFTCheckpointEntry(name="sft", path=str(sft_checkpoint)),
    ]

    scorecard = evaluate_sft(config)

    assert [result.name for result in scorecard.checkpoints] == ["pretrain", "sft"]
    assert all(result.status == "ok" for result in scorecard.checkpoints)
    assert all(result.assistant is not None for result in scorecard.checkpoints)
    assert all(result.math is not None for result in scorecard.checkpoints)
    assert all(result.tool is not None for result in scorecard.checkpoints)


def test_missing_checkpoint_is_skipped(eval_env: dict[str, Any]) -> None:
    config = load_config(eval_env["config_yaml"], SFTEvalConfig)
    config.checkpoints.append(
        SFTCheckpointEntry(name="missing", path=str(eval_env["tmp_path"] / "missing"))
    )

    scorecard = evaluate_sft(config)

    assert [result.status for result in scorecard.checkpoints] == ["ok", "missing"]
    missing = scorecard.checkpoints[1]
    assert missing.error is not None
    assert "weights.npz" in missing.error


def test_missing_checkpoint_error_raises(eval_env: dict[str, Any]) -> None:
    config = load_config(eval_env["config_yaml"], SFTEvalConfig)
    config.checkpoints = [
        SFTCheckpointEntry(name="missing", path=str(eval_env["tmp_path"] / "missing"))
    ]
    config.on_missing_checkpoint = "error"

    with pytest.raises(FileNotFoundError, match=r"weights\.npz"):
        evaluate_sft(config)


def test_perplexity_metrics_use_pretrain_eval(eval_env: dict[str, Any]) -> None:
    config = load_config(eval_env["config_yaml"], SFTEvalConfig)
    config.perplexity.enabled = True
    config.perplexity.max_tokens = 32

    metrics = _perplexity_metrics(config, eval_env["weights"])

    assert metrics is not None
    assert metrics.tokens > 0
    assert math.isfinite(metrics.loss)
    assert metrics.perplexity == pytest.approx(math.exp(metrics.loss))


def test_cli_writes_scorecard(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_cli()
    output_path = eval_env["tmp_path"] / "cli_scorecard.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_sft.py",
            "--config",
            str(eval_env["config_yaml"]),
            "--output",
            str(output_path),
            "--skip-perplexity",
        ],
    )

    module.main()

    out = capsys.readouterr().out
    assert f"scorecard: {output_path}" in out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["checkpoints"][0]["status"] == "ok"
