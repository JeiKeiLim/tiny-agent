"""Tests for Kestrel model I/O: load(config, checkpoint) + save(model, path)."""

import mlx.core as mx
from mlx.utils import tree_flatten

from kestrel.common.config import load_config
from kestrel.model.config import ModelConfig
from kestrel.model.io import load, save
from kestrel.model.kestrel import Kestrel

MODEL_50M = "configs/kestrel/50m/model.yaml"


def test_load_random_init():
    config = load_config(MODEL_50M, ModelConfig)
    model = load(config)
    assert isinstance(model, Kestrel)
    for _, p in tree_flatten(model.parameters()):
        assert mx.all(mx.isfinite(p)).item()


def test_save_checkpoint_dir_convention(tmp_path):
    config = load_config(MODEL_50M, ModelConfig)
    model = load(config)
    ckpt = tmp_path / "pretrain" / "kestrel-50m"
    save(model, ckpt)
    assert ckpt.is_dir()
    assert (ckpt / "weights.npz").is_file()


def test_save_load_round_trip(tmp_path):
    config = load_config(MODEL_50M, ModelConfig)
    model = load(config)
    ckpt = tmp_path / "pretrain" / "kestrel-50m"
    save(model, ckpt)
    reloaded = load(config, ckpt)
    orig = dict(tree_flatten(model.parameters()))
    new = dict(tree_flatten(reloaded.parameters()))
    assert set(orig) == set(new)
    for name in orig:
        assert mx.array_equal(orig[name], new[name]), f"weight mismatch: {name}"
