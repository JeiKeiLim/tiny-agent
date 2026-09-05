"""Tests for external pretrain benchmark evaluation (TASK-011)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from kestrel.eval.pretrain_benchmarks import (
    BpbAccumulator,
    benchmark_names,
    evaluate_language_modeling,
    evaluate_multiple_choice,
    evaluate_selected_benchmarks,
    mcq_case,
    parse_only,
    selected_specs,
    write_scorecard,
)
from kestrel.model.config import ModelConfig
from kestrel.model.io import save as save_model
from kestrel.model.kestrel import Kestrel
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train as train_tokenizer
from kestrel.train.pretrain import PretrainConfig
from kestrel.train.trainer import TrainerConfig

BASE = "the quick brown fox jumps over the lazy dog. "


def _tiny_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "tok_corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(BASE * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(tmp_path / "tok"),
        special_tokens=["im_start", "im_end"],
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


def _tiny_pretrain_config(tmp_path: Path, tokenizer_path: Path) -> tuple[PretrainConfig, Path]:
    model_yaml = tmp_path / "model.yaml"
    _write_yaml(model_yaml, _tiny_model_config().model_dump())

    config = PretrainConfig(
        model=str(model_yaml),
        tokenizer=str(tokenizer_path),
        corpus=str(tmp_path / "unused_corpus.yaml"),
        total_tokens=None,
        trainer=TrainerConfig(
            lr=1e-3,
            seq_len=16,
            batch_size=1,
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(dataset_dir: Path, files: list[str]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"files": files}),
        encoding="utf-8",
    )


def _make_lm_dataset(data_dir: Path) -> None:
    dataset_dir = data_dir / "wikitext2"
    _write_manifest(dataset_dir, ["test.jsonl"])
    _write_jsonl(
        dataset_dir / "test.jsonl",
        [
            {"page": BASE * 4},
            {"page": BASE * 3},
            {"page": ""},
        ],
    )


def _make_hellaswag_dataset(data_dir: Path) -> None:
    dataset_dir = data_dir / "hellaswag"
    _write_manifest(dataset_dir, ["validation.jsonl"])
    _write_jsonl(
        dataset_dir / "validation.jsonl",
        [
            {
                "ctx": "the quick brown fox",
                "endings": ["jumps", "sleeps", "eats", "runs"],
                "label": 0,
            },
            {
                "ctx": "the lazy dog",
                "endings": ["barks", "naps", "howls", "yawns"],
                "label": 1,
            },
        ],
    )


def _make_piqa_dataset(data_dir: Path) -> None:
    dataset_dir = data_dir / "piqa"
    _write_manifest(dataset_dir, ["piqa_validation.parquet"])
    table = pa.table(
        {
            "goal": ["the quick brown fox", "the lazy dog"],
            "sol1": ["jumps", "barks"],
            "sol2": ["sleeps", "naps"],
            "label": [0, 1],
        }
    )
    pq.write_table(table, dataset_dir / "piqa_validation.parquet")


@pytest.fixture(scope="module")
def bench_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("pretrain_benchmarks")
    tokenizer_path = _tiny_tokenizer(tmp_path)
    config, config_yaml = _tiny_pretrain_config(tmp_path, tokenizer_path)
    data_dir = tmp_path / "data"
    _make_lm_dataset(data_dir)
    _make_hellaswag_dataset(data_dir)
    _make_piqa_dataset(data_dir)
    return {
        "tmp_path": tmp_path,
        "config": config,
        "config_yaml": config_yaml,
        "checkpoint": _weights_checkpoint(tmp_path),
        "data_dir": data_dir,
    }


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kestrel_run_eval_pretrain_benchmarks",
        "scripts/run_eval_pretrain_benchmarks.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bpb_accumulator_derived_values() -> None:
    acc = BpbAccumulator(nats=math.log(8.0), tokens=8, bytes=32, examples=2)
    assert acc.loss == pytest.approx(math.log(8.0) / 8.0)
    assert acc.perplexity == pytest.approx(8.0 ** (1.0 / 8.0))
    assert acc.bits_per_token == pytest.approx(math.log(8.0) / 8.0 / math.log(2.0))
    assert acc.bpb == pytest.approx(math.log(8.0) / (math.log(2.0) * 32.0))


def test_parse_only_rejects_unknown_names() -> None:
    assert parse_only(None) is None
    assert parse_only("hellaswag, piqa") == {"hellaswag", "piqa"}
    with pytest.raises(ValueError, match="unknown benchmark"):
        parse_only("does-not-exist")


def test_selected_specs_respect_filters() -> None:
    all_names = set(benchmark_names())
    assert all_names == {spec.name for spec in selected_specs(False, None)}

    large = selected_specs(True, None)
    assert all(spec.large is False for spec in large)

    only = selected_specs(False, {"hellaswag", "wikitext2"})
    assert [spec.name for spec in only] == ["hellaswag", "wikitext2"]


def test_mcq_case_mappings() -> None:
    hellaswag = mcq_case(
        "hellaswag",
        {"ctx": "ctx", "endings": ["a", "b"], "label": 1},
    )
    assert hellaswag is not None
    assert hellaswag.choices == ("a", "b")
    assert hellaswag.label == 1

    arc = mcq_case(
        "arc_easy",
        {
            "question": "q",
            "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
    )
    assert arc is not None
    assert arc.label == 2

    winogrande = mcq_case(
        "winogrande",
        {"sentence": "so _ won", "option1": "a", "option2": "b", "answer": "2"},
    )
    assert winogrande is not None
    assert winogrande.context == "so "
    assert winogrande.choices == ("a won", "b won")
    assert winogrande.label == 1

    sciq = mcq_case(
        "sciq",
        {
            "question": "q",
            "distractor1": "a",
            "distractor2": "b",
            "distractor3": "c",
            "correct_answer": "d",
        },
        seed=0,
    )
    assert sciq is not None
    assert sciq.choices[sciq.label] == "d"

    with pytest.raises(ValueError, match="no multiple-choice mapping"):
        mcq_case("wikitext2", {})


def test_evaluate_language_modeling_reports_bpb(bench_env: dict[str, Any]) -> None:
    from tokenizers import Tokenizer

    config = bench_env["config"]
    model = Kestrel(_tiny_model_config())
    model.load_weights(str(bench_env["checkpoint"] / "weights.npz"))
    tokenizer = Tokenizer.from_file(config.tokenizer)

    result = evaluate_language_modeling(
        model,
        tokenizer,
        bench_env["data_dir"] / "wikitext2",
        name="wikitext2",
        text_field="page",
        context_length=16,
        max_tokens=64,
    )

    assert result.status == "ok"
    metrics = result.metrics
    assert metrics["tokens"] > 0
    assert metrics["bytes"] > 0
    assert metrics["examples"] == 2
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["bpb"])
    assert metrics["bpb"] == pytest.approx(
        metrics["loss"] / math.log(2.0) * metrics["tokens"] / metrics["bytes"]
    )


def test_evaluate_multiple_choice_reports_accuracy(bench_env: dict[str, Any]) -> None:
    from tokenizers import Tokenizer

    config = bench_env["config"]
    model = Kestrel(_tiny_model_config())
    model.load_weights(str(bench_env["checkpoint"] / "weights.npz"))
    tokenizer = Tokenizer.from_file(config.tokenizer)

    result = evaluate_multiple_choice(
        model,
        tokenizer,
        bench_env["data_dir"] / "hellaswag",
        name="hellaswag",
        context_length=16,
        max_examples=2,
    )

    assert result.status == "ok"
    assert result.metrics["examples"] == 2
    assert 0.0 <= result.metrics["acc"] <= 100.0
    assert 0.0 <= result.metrics["acc_norm"] <= 100.0


def test_evaluate_selected_benchmarks_writes_scorecard(bench_env: dict[str, Any]) -> None:
    scorecard = evaluate_selected_benchmarks(
        pretrain_config=bench_env["config"],
        checkpoint=bench_env["checkpoint"],
        data_dir=bench_env["data_dir"],
        specs=selected_specs(False, {"wikitext2", "hellaswag", "piqa"}),
        max_tokens=32,
        max_examples=2,
    )

    assert [result.name for result in scorecard.results] == [
        "hellaswag",
        "piqa",
        "wikitext2",
    ]
    assert all(result.status == "ok" for result in scorecard.results)

    output = bench_env["tmp_path"] / "scorecard.json"
    write_scorecard(scorecard, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert len(payload["results"]) == 3


def test_evaluate_selected_benchmarks_missing_behavior(bench_env: dict[str, Any]) -> None:
    specs = selected_specs(False, {"wikitext2", "mmlu"})

    with pytest.raises(FileNotFoundError, match="benchmark dataset not found"):
        evaluate_selected_benchmarks(
            pretrain_config=bench_env["config"],
            checkpoint=bench_env["checkpoint"],
            data_dir=bench_env["data_dir"],
            specs=specs,
            allow_missing=False,
        )

    scorecard = evaluate_selected_benchmarks(
        pretrain_config=bench_env["config"],
        checkpoint=bench_env["checkpoint"],
        data_dir=bench_env["data_dir"],
        specs=specs,
        allow_missing=True,
    )
    by_name = {result.name: result for result in scorecard.results}
    assert by_name["wikitext2"].status == "ok"
    assert by_name["mmlu"].status == "missing"


def test_cli_json_output(
    bench_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    output = bench_env["tmp_path"] / "cli_scorecard.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_pretrain_benchmarks.py",
            "--pretrain-config",
            str(bench_env["config_yaml"]),
            "--checkpoint",
            str(bench_env["checkpoint"]),
            "--data-dir",
            str(bench_env["data_dir"]),
            "--only",
            "wikitext2,hellaswag",
            "--max-tokens",
            "32",
            "--max-examples",
            "2",
            "--output",
            str(output),
            "--json",
        ],
    )
    module.main()

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert output.is_file()
    assert [result["name"] for result in payload["results"]] == ["hellaswag", "wikitext2"]
    assert all(result["status"] == "ok" for result in payload["results"])


def test_cli_missing_dataset_is_error(
    bench_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_pretrain_benchmarks.py",
            "--pretrain-config",
            str(bench_env["config_yaml"]),
            "--checkpoint",
            str(bench_env["checkpoint"]),
            "--data-dir",
            str(bench_env["data_dir"]),
            "--only",
            "mmlu",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert "benchmark dataset not found" in str(excinfo.value)


def test_cli_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli()
    monkeypatch.setattr(sys, "argv", ["run_eval_pretrain_benchmarks.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--data-dir" in out
    assert "--max-tokens" in out
    assert "--max-examples" in out
