"""Tests for the pretrain phase (src/kestrel/train/pretrain.py).

A tiny model + tiny in-test tokenizer + tiny local corpus are assembled in
``tmp_path`` so the end-to-end pretrain runs fast (no gitignored artifacts, no
1GB corpus). The model vocab (400) matches the tokenizer vocab (400). The
corpus lines are unique (indexed) so the deterministic hash split produces both
a train and a val slice, while staying highly repetitive so the tiny model can
drive the loss down in a few dozen steps.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.corpus.config import ComponentConfig, CorpusConfig, LocalSourceConfig
from kestrel.data.pretrain_dataset import (
    PretrainDataset,
    PretrainDatasetConfig,
    PretrainDatasetIterator,
)
from kestrel.model.config import ModelConfig
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train as train_tokenizer
from kestrel.train import pretrain as pretrain_module
from kestrel.train.checkpoint import read_checkpoint_state
from kestrel.train.pretrain import PretrainConfig, pretrain
from kestrel.train.trainer import TrainerConfig, TrainResult

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
        context_length=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_size=64,
        intermediate_size=128,
    )


def _write_yaml(path: Path, obj: object) -> None:
    path.write_text(yaml.safe_dump(obj), encoding="utf-8")


def _tiny_pretrain_config(tmp_path: Path, tok: Path, num_steps: int = 30) -> PretrainConfig:
    model_yaml = tmp_path / "model.yaml"
    _write_yaml(model_yaml, _tiny_model_config().model_dump())

    src_dir = tmp_path / "corpus_src"
    src_dir.mkdir()
    lines = [f"line {i}: {BASE}" for i in range(500)]
    (src_dir / "web.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    corpus_cfg = CorpusConfig(
        total_bytes=100_000,
        seed=0,
        output_dir=str(tmp_path / "corpus_out"),
        val_fraction=0.1,
        test_fraction=0.0,
        min_component_fill=0.0,
        components=[
            ComponentConfig(
                name="web",
                source=LocalSourceConfig(type="local", path=str(src_dir / "web.txt")),
                fraction=1.0,
            )
        ],
    )
    corpus_yaml = tmp_path / "corpus.yaml"
    _write_yaml(corpus_yaml, corpus_cfg.model_dump())

    return PretrainConfig(
        model=str(model_yaml),
        tokenizer=str(tok),
        corpus=str(corpus_yaml),
        total_tokens=1024,
        trainer=TrainerConfig(
            lr=1e-3,
            seq_len=16,
            batch_size=2,
            num_steps=num_steps,
            warmup_steps=5,
            eval_every=10,
            eval_iters=2,
            log_every=100,
            save_every=100,
            output_dir=str(tmp_path / "ckpt"),
        ),
    )


def test_pretrain_end_to_end(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_pretrain_config(tmp_path, tok)
    result = pretrain(config)
    assert math.isfinite(result.final_loss)
    first = result.history[0][1]
    last = result.history[-1][1]
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"
    assert (tmp_path / "ckpt" / "final" / "weights.npz").exists()


def test_pretrain_auto_num_steps_uses_estimated_steps(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_pretrain_config(tmp_path, tok, num_steps=0)
    result = pretrain(config)
    assert result.num_steps == 32
    assert result.schedule_steps == 32
    assert math.isfinite(result.final_loss)
    assert (tmp_path / "ckpt" / "final" / "weights.npz").exists()


def test_pretrain_config_strict() -> None:
    PretrainConfig(model="m.yaml", tokenizer="t.json", corpus="c.yaml")  # valid
    with pytest.raises(ValidationError):
        PretrainConfig(model="m.yaml", tokenizer="t.json", corpus="c.yaml", bogus=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PretrainConfig(  # type: ignore[arg-type]
            model="m.yaml", tokenizer="t.json", corpus="c.yaml", total_tokens="lots"
        )


def test_50m_pretrain_yaml_loads() -> None:
    config = load_config("configs/kestrel/50m/pretrain.yaml", PretrainConfig)
    assert config.corpus == "configs/kestrel/corpus.yaml"
    assert config.total_tokens == 1013504000
    assert config.trainer.batch_size == 8
    assert config.trainer.seq_len == 1024
    assert config.trainer.num_steps == 0
    assert config.trainer.save_every == 2000
    assert config.trainer.output_dir == "checkpoints/pretrain/50m"


def test_150m_pretrain_yaml_loads() -> None:
    config = load_config("configs/kestrel/150m/pretrain.yaml", PretrainConfig)
    assert config.corpus == "configs/kestrel/corpus.yaml"
    assert config.total_tokens is None
    assert config.trainer.batch_size == 4
    assert config.trainer.seq_len == 1024
    assert config.trainer.num_steps == 0
    assert config.trainer.save_every == 2000
    assert config.trainer.output_dir == "checkpoints/pretrain/150m"


CORPUS_12G_TRAIN_MANIFEST = Path("data/corpus-12g/train/manifest.json")
TOKENIZER_PATH = Path("checkpoints/tokenizer/tokenizer.json")


@pytest.mark.skipif(
    not CORPUS_12G_TRAIN_MANIFEST.exists() or not TOKENIZER_PATH.exists(),
    reason="data/corpus-12g and the trained tokenizer are not present",
)
def test_12g_150m_estimated_steps_matches_manifest() -> None:
    manifest = json.loads(CORPUS_12G_TRAIN_MANIFEST.read_text(encoding="utf-8"))
    dataset = PretrainDataset(
        PretrainDatasetConfig(
            input="data/corpus-12g/train",
            tokenizer_path=str(TOKENIZER_PATH),
            context_length=1024,
            batch_size=4,
            total_tokens=None,
            seed=0,
        )
    )
    expected = manifest["total_estimated_token_count"] // (4 * 1024)
    assert abs(dataset.estimated_steps() - expected) / expected < 0.05


@pytest.mark.skipif(
    not CORPUS_12G_TRAIN_MANIFEST.exists() or not TOKENIZER_PATH.exists(),
    reason="data/corpus-12g and the trained tokenizer are not present",
)
def test_12g_50m_estimated_steps_uses_token_cap() -> None:
    dataset = PretrainDataset(
        PretrainDatasetConfig(
            input="data/corpus-12g/train",
            tokenizer_path=str(TOKENIZER_PATH),
            context_length=1024,
            batch_size=8,
            total_tokens=1013504000,
            seed=0,
        )
    )
    expected = 1013504000 // (8 * 1024)
    assert abs(dataset.estimated_steps() - expected) / expected < 0.05


# --- resume ---


def test_pretrain_resume_from_completed_final_is_noop(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_pretrain_config(tmp_path, tok, num_steps=5)
    first = pretrain(config)
    assert first.num_steps == 5

    config.resume = str(tmp_path / "ckpt" / "final")
    second = pretrain(config)

    assert second.num_steps == 5
    state = read_checkpoint_state(tmp_path / "ckpt" / "final")
    assert state["step"] == 5


def test_pretrain_resume_rejects_incompatible_trainer_config(tmp_path: Path) -> None:
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_pretrain_config(tmp_path, tok, num_steps=5)
    pretrain(config)

    config.resume = str(tmp_path / "ckpt" / "final")
    config.trainer.batch_size = 4
    with pytest.raises(ValueError, match="trainer config"):
        pretrain(config)


class _CrashAfterIterator:
    def __init__(self, inner: PretrainDatasetIterator, crash_after: int) -> None:
        self._inner = inner
        self._crash_after = crash_after
        self._count = 0

    def __iter__(self) -> _CrashAfterIterator:
        return self

    def __next__(self) -> tuple[Any, Any, Any]:
        batch = next(self._inner)
        self._count += 1
        if self._count >= self._crash_after:
            msg = "simulated crash"
            raise RuntimeError(msg)
        return batch

    def state_dict(self) -> dict[str, Any]:
        return self._inner.state_dict()

    def close(self) -> None:
        self._inner.close()


class _CrashAfterPretrainDataset:
    def __init__(self, inner: PretrainDataset, crash_after: int) -> None:
        self._inner = inner
        self._crash_after = crash_after

    def estimated_steps(self) -> int:
        return self._inner.estimated_steps()

    def iterator(self) -> _CrashAfterIterator:
        return _CrashAfterIterator(self._inner.iterator(), self._crash_after)

    def load_iterator(self, state: dict[str, Any]) -> PretrainDatasetIterator:
        return self._inner.load_iterator(state)


def test_pretrain_resume_continues_after_simulated_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tok = _tiny_tokenizer(tmp_path)
    config = _tiny_pretrain_config(tmp_path, tok, num_steps=10)
    config.trainer.save_every = 5

    real_dataset = pretrain_module._dataset

    def fake_dataset(
        cfg: PretrainConfig, corpus_cfg: CorpusConfig, split: str, total_tokens: int | None
    ) -> PretrainDataset:
        dataset = real_dataset(cfg, corpus_cfg, split, total_tokens)
        if split == "train":
            # Return five batches, then crash on the sixth ``next()`` call so
            # the step-5 checkpoint has already been written.
            return cast(PretrainDataset, _CrashAfterPretrainDataset(dataset, 6))
        return dataset

    monkeypatch.setattr(pretrain_module, "_dataset", fake_dataset)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pretrain(config)

    checkpoint_dir = tmp_path / "ckpt" / "step_000005"
    assert checkpoint_dir.is_dir()
    state = read_checkpoint_state(checkpoint_dir)
    assert state["step"] == 5

    config.resume = str(checkpoint_dir)
    result = pretrain(config)

    assert result.num_steps == 10
    assert len(result.history) == 5
    final_state = read_checkpoint_state(tmp_path / "ckpt" / "final")
    assert final_state["step"] == 10


def test_run_pretrain_cli_resume_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location("kestrel_run_pretrain", "scripts/run_pretrain.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: dict[str, Any] = {}

    def fake_pretrain(config: PretrainConfig, config_path: str | Path | None = None) -> TrainResult:
        calls["config"] = config
        calls["config_path"] = config_path
        return TrainResult(final_loss=1.0, num_steps=1)

    monkeypatch.setattr(module, "pretrain", fake_pretrain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pretrain.py",
            "--config",
            "configs/kestrel/50m/pretrain.yaml",
            "--resume",
            "checkpoints/pretrain/50m/step_000010",
        ],
    )

    module.main()

    assert calls["config"].resume == "checkpoints/pretrain/50m/step_000010"
    assert calls["config_path"] == "configs/kestrel/50m/pretrain.yaml"
    assert "steps:       1" in capsys.readouterr().out
