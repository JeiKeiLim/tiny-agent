"""Tests for the SFT phase (src/kestrel/train/sft.py).

A tiny model + tiny in-test tokenizer + tiny local SFT JSONL are assembled in
``tmp_path`` so the end-to-end SFT run is fast and does not depend on
gitignored checkpoints or the real 50M mixture.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import pytest
import yaml
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.data.sft_dataset import SFTDataset, SFTDatasetConfig
from kestrel.model.config import ModelConfig
from kestrel.model.io import load as load_model
from kestrel.model.io import save as save_model
from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS, TokenizerConfig
from kestrel.tokenizer.train import train as train_tokenizer
from kestrel.train import sft as sft_module
from kestrel.train.checkpoint import read_checkpoint_state
from kestrel.train.sft import SFTConfig, sft
from kestrel.train.trainer import TrainerConfig, TrainResult, _batch_loss

BASE = "the quick brown fox jumps over the lazy dog. "


def _train_tiny_tokenizer(output_root: Path) -> Path:
    corpus = output_root / "tok_corpus"
    corpus.mkdir()
    (corpus / "sft.txt").write_text(BASE * 500 + "hello world assistant user tool " * 500)
    config = TokenizerConfig(
        vocab_size=400,
        train_dir=str(corpus),
        output_dir=str(output_root / "tok"),
        special_tokens=list(DEFAULT_SPECIAL_TOKENS),
        eos_token=DEFAULT_SPECIAL_TOKENS[1],
    )
    return train_tokenizer(config)


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _train_tiny_tokenizer(tmp_path_factory.mktemp("sft_tokenizer"))


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


def _sft_row(index: int, source: str = "assistant_public") -> dict[str, object]:
    return {
        "source": source,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": f"hello world {index}"},
        ],
    }


def _write_sft_jsonl(path: Path, rows: int) -> None:
    with path.open("w", encoding="utf-8") as fin:
        for index in range(rows):
            fin.write(json.dumps(_sft_row(index)) + "\n")


def _sft_config(tmp_path: Path, tok: Path, num_steps: int = 10) -> SFTConfig:
    model_yaml = tmp_path / "model.yaml"
    _write_yaml(model_yaml, _tiny_model_config().model_dump())

    pretrain_ckpt = tmp_path / "pretrain_final"
    save_model(load_model(_tiny_model_config()), pretrain_ckpt)

    data = tmp_path / "sft.jsonl"
    _write_sft_jsonl(data, 20)

    dataset = SFTDatasetConfig(
        input=str(data),
        tokenizer_path=str(tok),
        context_length=32,
        batch_size=2,
        seed=0,
        max_examples=None,
        preserve_source_ratios=True,
        epochs=1,
    )
    trainer = TrainerConfig(
        lr=1e-3,
        seq_len=32,
        batch_size=2,
        num_steps=num_steps,
        warmup_steps=2,
        eval_every=0,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path / "ckpt"),
        use_loss_mask=True,
    )
    return SFTConfig(
        model=str(model_yaml),
        checkpoint=str(pretrain_ckpt),
        dataset=dataset,
        trainer=trainer,
    )


def test_sft_config_is_strict(tmp_path: Path, tiny_tokenizer: Path) -> None:
    config = _sft_config(tmp_path, tiny_tokenizer)
    assert config.trainer.use_loss_mask is True

    with pytest.raises(ValidationError):
        SFTConfig(  # type: ignore[call-arg]
            model="m.yaml",
            checkpoint="ckpt",
            dataset=SFTDatasetConfig(input="x.jsonl", tokenizer_path="t.json"),
            bogus=1,
        )
    with pytest.raises(ValidationError):
        SFTConfig(  # type: ignore[arg-type]
            model="m.yaml",
            checkpoint="ckpt",
            dataset=SFTDatasetConfig(input="x.jsonl", tokenizer_path="t.json", context_length=0),
        )


def test_sft_rejects_shape_mismatch(tmp_path: Path, tiny_tokenizer: Path) -> None:
    config = _sft_config(tmp_path, tiny_tokenizer)
    config.trainer.seq_len = 64
    with pytest.raises(ValueError, match="context_length"):
        sft(config)

    config = _sft_config(tmp_path, tiny_tokenizer)
    config.trainer.use_loss_mask = False
    with pytest.raises(ValueError, match="use_loss_mask"):
        sft(config)


def test_50m_sft_yaml_loads() -> None:
    config = load_config("configs/kestrel/50m/sft.yaml", SFTConfig)
    assert config.checkpoint == "checkpoints/pretrain/50m/final"
    assert config.dataset.context_length == 1024
    assert config.dataset.batch_size == 8
    assert config.dataset.epochs == 1
    assert config.trainer.seq_len == 1024
    assert config.trainer.batch_size == 8
    assert config.trainer.use_loss_mask is True
    assert config.trainer.output_dir == "checkpoints/sft/50m"


def test_150m_sft_yaml_loads() -> None:
    config = load_config("configs/kestrel/150m/sft.yaml", SFTConfig)
    assert config.checkpoint == "checkpoints/pretrain/150m/final"
    assert config.dataset.context_length == 1024
    assert config.dataset.batch_size == 4
    assert config.trainer.seq_len == 1024
    assert config.trainer.batch_size == 4
    assert config.trainer.use_loss_mask is True
    assert config.trainer.output_dir == "checkpoints/sft/150m"


def test_batch_loss_applies_loss_mask() -> None:
    model = load_model(_tiny_model_config())
    x = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    target = mx.array([[2, 3, 4, 4]], dtype=mx.int32)
    mask = mx.array([[0, 1, 0, 0]], dtype=mx.int32)

    loss = _batch_loss(model, x, target, mask, use_loss_mask=True)
    assert math.isfinite(float(loss.item()))

    masked_target = mx.array([[2, 9, 8, 7]], dtype=mx.int32)
    masked_loss = _batch_loss(model, x, masked_target, mask, use_loss_mask=True)
    assert float(loss.item()) == float(masked_loss.item())

    zero_mask = mx.zeros((1, 4), dtype=mx.int32)
    zero_loss = _batch_loss(model, x, target, zero_mask, use_loss_mask=True)
    assert float(zero_loss.item()) == 0.0


def test_sft_end_to_end(tmp_path: Path, tiny_tokenizer: Path) -> None:
    config = _sft_config(tmp_path, tiny_tokenizer, num_steps=10)
    result = sft(config)

    assert math.isfinite(result.final_loss)
    first = result.history[0][1]
    last = result.history[-1][1]
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"
    assert (tmp_path / "ckpt" / "final" / "weights.npz").exists()


def test_sft_resume_from_completed_final_is_noop(tmp_path: Path, tiny_tokenizer: Path) -> None:
    config = _sft_config(tmp_path, tiny_tokenizer, num_steps=5)
    first = sft(config)
    assert first.num_steps == 5

    config.resume = str(tmp_path / "ckpt" / "final")
    second = sft(config)

    assert second.num_steps == 5
    state = read_checkpoint_state(tmp_path / "ckpt" / "final")
    assert state["step"] == 5


def test_sft_resume_rejects_incompatible_trainer_config(
    tmp_path: Path, tiny_tokenizer: Path
) -> None:
    config = _sft_config(tmp_path, tiny_tokenizer, num_steps=5)
    sft(config)

    config.resume = str(tmp_path / "ckpt" / "final")
    config.trainer.lr = 2e-3
    with pytest.raises(ValueError, match="trainer config"):
        sft(config)


class _CrashAfterSFTIterator:
    def __init__(self, inner: Any, crash_after: int) -> None:
        self._inner = inner
        self._crash_after = crash_after
        self._count = 0

    def __iter__(self) -> _CrashAfterSFTIterator:
        return self

    def __next__(self) -> tuple[mx.array, mx.array, mx.array]:
        batch = next(self._inner)
        self._count += 1
        if self._count >= self._crash_after:
            msg = "simulated crash"
            raise RuntimeError(msg)
        return batch

    def state_dict(self) -> dict[str, Any]:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._inner.load_state_dict(state)

    def close(self) -> None:
        self._inner.close()


class _CrashAfterSFTDataset:
    def __init__(self, inner: SFTDataset, crash_after: int) -> None:
        self._inner = inner
        self._crash_after = crash_after

    def estimated_steps(self) -> int:
        return self._inner.estimated_steps()

    def iterator(self) -> _CrashAfterSFTIterator:
        return _CrashAfterSFTIterator(self._inner.iterator(), self._crash_after)

    def load_iterator(self, state: dict[str, Any]) -> Any:
        return self._inner.load_iterator(state)


def test_sft_resume_continues_after_simulated_crash(
    tmp_path: Path, tiny_tokenizer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _sft_config(tmp_path, tiny_tokenizer, num_steps=10)
    config.trainer.save_every = 5

    real_dataset = sft_module._train_dataset

    def fake_dataset(cfg: SFTConfig) -> SFTDataset:
        dataset = real_dataset(cfg)
        return cast(SFTDataset, _CrashAfterSFTDataset(dataset, 6))

    monkeypatch.setattr(sft_module, "_train_dataset", fake_dataset)
    with pytest.raises(RuntimeError, match="simulated crash"):
        sft(config)

    checkpoint_dir = tmp_path / "ckpt" / "step_000005"
    assert checkpoint_dir.is_dir()
    state = read_checkpoint_state(checkpoint_dir)
    assert state["step"] == 5

    config.resume = str(checkpoint_dir)
    result = sft(config)

    assert result.num_steps == 10
    assert len(result.history) == 5
    final_state = read_checkpoint_state(tmp_path / "ckpt" / "final")
    assert final_state["step"] == 10


def test_run_sft_cli_resume_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location("kestrel_run_sft", "scripts/run_sft.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: dict[str, Any] = {}

    def fake_sft(config: SFTConfig, config_path: str | Path | None = None) -> TrainResult:
        calls["config"] = config
        calls["config_path"] = config_path
        return TrainResult(final_loss=1.0, num_steps=1)

    monkeypatch.setattr(module, "sft", fake_sft)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sft.py",
            "--config",
            "configs/kestrel/50m/sft.yaml",
            "--resume",
            "checkpoints/sft/50m/step_000010",
        ],
    )

    module.main()

    assert calls["config"].resume == "checkpoints/sft/50m/step_000010"
    assert calls["config_path"] == "configs/kestrel/50m/sft.yaml"
    assert "steps:       1" in capsys.readouterr().out
