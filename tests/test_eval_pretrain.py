"""Tests for pretrain checkpoint evaluation (src/kestrel/eval/pretrain.py).

A tiny model, tiny in-test tokenizer, and tiny local corpus are assembled in a
module-scoped temp directory so the evaluation tests stay fast while exercising
the real corpus builder, dataset, model I/O, and CLI code paths.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import mlx.optimizers as optim
import pytest
import yaml

from kestrel.corpus.builder import build as build_corpus
from kestrel.corpus.config import ComponentConfig, CorpusConfig, LocalSourceConfig
from kestrel.eval.pretrain import (
    EvalMetrics,
    LossAccumulator,
    _format_progress,
    _manifest_domain_tokens,
    _manifest_total_tokens,
    evaluate_checkpoint,
)
from kestrel.model.config import ModelConfig
from kestrel.model.io import save as save_model
from kestrel.model.kestrel import Kestrel
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train as train_tokenizer
from kestrel.train.checkpoint import save_full_checkpoint
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


def _full_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "full"
    save_full_checkpoint(
        Kestrel(_tiny_model_config()), optim.AdamW(learning_rate=1e-3), checkpoint, {"step": 0}
    )
    return checkpoint


@pytest.fixture(scope="module")
def eval_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("eval_pretrain")
    tokenizer_path = _tiny_tokenizer(tmp_path)
    corpus_config = _tiny_corpus(tmp_path)
    config, config_yaml = _tiny_pretrain_config(tmp_path, tokenizer_path, corpus_config)
    return {
        "tmp_path": tmp_path,
        "config": config,
        "config_yaml": config_yaml,
        "weights": _weights_checkpoint(tmp_path),
        "full": _full_checkpoint(tmp_path),
    }


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kestrel_eval_pretrain", "scripts/eval_pretrain.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loss_accumulator_is_token_weighted() -> None:
    acc = LossAccumulator()
    acc.add(2.0, 1)
    acc.add(4.0, 3)
    assert acc.tokens == 4
    assert acc.batches == 2
    assert acc.loss == pytest.approx(1.5)
    with pytest.raises(ValueError, match="no tokens"):
        _ = LossAccumulator().loss


def test_eval_metrics_derived_values() -> None:
    metrics = EvalMetrics(loss=math.log(2.0), tokens=10, batches=2)
    assert metrics.perplexity == pytest.approx(2.0)
    assert metrics.bits_per_token == pytest.approx(1.0)
    assert metrics.to_dict()["tokens"] == 10


def test_manifest_token_estimates() -> None:
    manifest = {
        "total_token_count": None,
        "total_estimated_token_count": 100,
        "total_doc_count": 10,
        "files": [
            {
                "domain": "web",
                "path": "web.jsonl",
                "token_count": None,
                "estimated_token_count": 60,
                "doc_count": 4,
            }
        ],
    }
    assert _manifest_total_tokens(manifest) == 120
    assert _manifest_domain_tokens(manifest, "web", "jsonl") == 68
    assert _manifest_domain_tokens(manifest, "code", "jsonl") is None
    assert _manifest_total_tokens(None) is None


def test_format_progress_caps_at_100_percent() -> None:
    acc = LossAccumulator()
    acc.add(1.0, 120)
    assert _format_progress(acc, 100) == "120/~100 tokens (100.0%)"
    assert _format_progress(acc, None) == "120 tokens"

    large = LossAccumulator()
    large.add(1.0, 16_368)
    assert _format_progress(large, 100_000) == "16,368/~100,000 tokens (16.4%)"


def test_evaluate_checkpoint_reports_mixed_metrics(eval_env: dict[str, Any]) -> None:
    result = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=128,
    )
    assert math.isfinite(result.mixed.loss)
    assert result.mixed.tokens > 0
    assert result.mixed.perplexity == pytest.approx(math.exp(result.mixed.loss))
    assert result.mixed.bits_per_token == pytest.approx(result.mixed.loss / math.log(2.0))


def test_evaluate_checkpoint_reports_domains(eval_env: dict[str, Any]) -> None:
    result = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=128,
    )
    assert set(result.domains) == set(DOMAINS)
    for metrics in result.domains.values():
        assert math.isfinite(metrics.loss)
        assert metrics.tokens > 0


@pytest.mark.parametrize("checkpoint_key", ["weights", "full"])
def test_evaluate_checkpoint_accepts_weights_only_and_full(
    eval_env: dict[str, Any], checkpoint_key: str
) -> None:
    result = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env[checkpoint_key],
        max_tokens=64,
    )
    assert result.mixed.tokens > 0
    assert math.isfinite(result.mixed.loss)


def test_evaluate_checkpoint_respects_max_tokens(eval_env: dict[str, Any]) -> None:
    full = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=None,
    )
    capped = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=64,
    )
    assert full.mixed.tokens > capped.mixed.tokens
    assert capped.mixed.tokens <= 64


def test_evaluate_checkpoint_progress_prints_to_stderr(
    eval_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=64,
        progress_every_tokens=30,
    )
    err = capsys.readouterr().err
    assert "[eval] mixed:" in err
    assert "[eval] domain:web:" in err
    assert "/~" in err
    assert "%" in err


def test_evaluate_checkpoint_is_read_only(eval_env: dict[str, Any]) -> None:
    checkpoint = eval_env["weights"]

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in checkpoint.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=checkpoint,
        max_tokens=64,
    )
    assert snapshot() == before


def test_generate_samples_do_not_change_loss(eval_env: dict[str, Any]) -> None:
    base = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=64,
        generate_samples=False,
    )
    with_samples = evaluate_checkpoint(
        pretrain_config=eval_env["config"],
        checkpoint=eval_env["weights"],
        max_tokens=64,
        generate_samples=True,
    )
    assert base.mixed == with_samples.mixed
    assert with_samples.samples is not None
    assert len(with_samples.samples) == 3


def test_cli_prints_metrics(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pretrain.py",
            "--pretrain-config",
            str(eval_env["config_yaml"]),
            "--checkpoint",
            str(eval_env["weights"]),
            "--max-tokens",
            "64",
        ],
    )
    module.main()
    out = capsys.readouterr().out
    assert "checkpoint:" in out
    assert "loss:" in out
    assert "perplexity:" in out
    assert "bits/token:" in out
    assert "tokens:" in out


def test_cli_json_output(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pretrain.py",
            "--pretrain-config",
            str(eval_env["config_yaml"]),
            "--checkpoint",
            str(eval_env["weights"]),
            "--max-tokens",
            "64",
            "--json",
            "--generate",
        ],
    )
    module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["split"] == "val"
    assert "loss" in payload["mixed"]
    assert set(payload["domains"]) == set(DOMAINS)
    assert len(payload["samples"]) == 3


def test_cli_progress_keeps_json_stdout_parseable(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pretrain.py",
            "--pretrain-config",
            str(eval_env["config_yaml"]),
            "--checkpoint",
            str(eval_env["weights"]),
            "--max-tokens",
            "64",
            "--progress-every-tokens",
            "30",
            "--json",
        ],
    )
    module.main()
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["mixed"]["tokens"] > 0
    assert "[eval] mixed:" in err


def test_cli_warns_on_train_split(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pretrain.py",
            "--pretrain-config",
            str(eval_env["config_yaml"]),
            "--checkpoint",
            str(eval_env["weights"]),
            "--split",
            "train",
            "--max-tokens",
            "64",
        ],
    )
    module.main()
    assert "warning: evaluating the train split" in capsys.readouterr().err


def test_cli_missing_checkpoint_gives_clear_error(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pretrain.py",
            "--pretrain-config",
            str(eval_env["config_yaml"]),
            "--checkpoint",
            str(eval_env["tmp_path"] / "missing"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert "weights.npz" in str(excinfo.value)


def test_cli_missing_config_gives_clear_error(
    eval_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_pretrain.py",
            "--pretrain-config",
            str(eval_env["tmp_path"] / "missing.yaml"),
            "--checkpoint",
            str(eval_env["weights"]),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert "pretrain config not found" in str(excinfo.value)


def test_cli_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli()
    monkeypatch.setattr(sys, "argv", ["eval_pretrain.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert excinfo.value.code == 0
    assert "--max-tokens" in capsys.readouterr().out
