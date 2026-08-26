"""Tests for the pretrain phase (src/kestrel/train/pretrain.py).

A tiny model + tiny in-test tokenizer + tiny local corpus are assembled in
``tmp_path`` so the end-to-end pretrain runs fast (no gitignored artifacts, no
1GB corpus). The model vocab (400) matches the tokenizer vocab (400). The
corpus lines are unique (indexed) so the deterministic hash split produces both
a train and a val slice, while staying highly repetitive so the tiny model can
drive the loss down in a few dozen steps.
"""

import json
import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.corpus.config import ComponentConfig, CorpusConfig, LocalSourceConfig
from kestrel.data.pretrain_dataset import PretrainDataset, PretrainDatasetConfig
from kestrel.model.config import ModelConfig
from kestrel.tokenizer.config import TokenizerConfig
from kestrel.tokenizer.train import train as train_tokenizer
from kestrel.train.pretrain import PretrainConfig, pretrain
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
