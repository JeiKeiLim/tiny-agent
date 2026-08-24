"""Tests for the corpus config model and builder.

The models + builder are the code under test. Real config *values* are tunable
data, so tests use tiny synthetic fixtures (small local files, a mocked
HuggingFace stream) and only check that the real config is well-formed and
loadable.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kestrel.common.config import load_config
from kestrel.corpus.builder import build
from kestrel.corpus.config import (
    ComponentConfig,
    CorpusConfig,
    HfSourceConfig,
    LocalSourceConfig,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _local_component(name: str, path: str, fraction: float = 1.0) -> ComponentConfig:
    return ComponentConfig(
        name=name,
        fraction=fraction,
        source=LocalSourceConfig(type="local", path=path),
    )


# --- config model ---


def test_valid_config_loads(tmp_path):
    p = _write(
        tmp_path,
        "corpus.yaml",
        "total_bytes: 1000\n"
        "seed: 42\n"
        "output_dir: /tmp/corpus\n"
        "components:\n"
        "  - name: web\n"
        "    fraction: 0.7\n"
        "    source:\n"
        "      type: local\n"
        "      path: /tmp/web.txt\n"
        "  - name: code\n"
        "    fraction: 0.3\n"
        "    source:\n"
        "      type: hf\n"
        "      dataset: some/ds\n"
        "      text_field: text\n",
    )
    cfg = load_config(p, CorpusConfig)
    assert cfg.total_bytes == 1000
    assert cfg.seed == 42
    assert [c.name for c in cfg.components] == ["web", "code"]
    assert isinstance(cfg.components[0].source, LocalSourceConfig)
    assert cfg.components[0].source.path == "/tmp/web.txt"
    assert isinstance(cfg.components[1].source, HfSourceConfig)
    assert cfg.components[1].source.dataset == "some/ds"
    assert cfg.val_fraction == 0.1  # default
    assert cfg.test_fraction == 0.0  # default


def test_fractions_must_sum_to_one(tmp_path):
    p = _write(
        tmp_path,
        "corpus.yaml",
        "total_bytes: 1000\n"
        "components:\n"
        "  - name: web\n"
        "    fraction: 0.5\n"
        "    source:\n"
        "      type: local\n"
        "      path: /tmp/web.txt\n"
        "  - name: code\n"
        "    fraction: 0.3\n"
        "    source:\n"
        "      type: local\n"
        "      path: /tmp/code.txt\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, CorpusConfig)


def test_rejects_unknown_key(tmp_path):
    p = _write(
        tmp_path,
        "corpus.yaml",
        "total_bytes: 1000\n"
        "bogus: 1\n"
        "components:\n"
        "  - name: web\n"
        "    fraction: 1.0\n"
        "    source:\n"
        "      type: local\n"
        "      path: /tmp/web.txt\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, CorpusConfig)


def test_rejects_bad_source_type(tmp_path):
    p = _write(
        tmp_path,
        "corpus.yaml",
        "total_bytes: 1000\n"
        "components:\n"
        "  - name: web\n"
        "    fraction: 1.0\n"
        "    source:\n"
        "      type: s3\n"
        "      path: /tmp/web.txt\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, CorpusConfig)


def test_split_fractions_must_fit(tmp_path):
    p = _write(
        tmp_path,
        "corpus.yaml",
        "total_bytes: 1000\n"
        "val_fraction: 0.8\n"
        "test_fraction: 0.3\n"
        "components:\n"
        "  - name: web\n"
        "    fraction: 1.0\n"
        "    source:\n"
        "      type: local\n"
        "      path: /tmp/web.txt\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, CorpusConfig)


def test_real_corpus_config_loads():
    cfg = load_config(CONFIGS / "kestrel" / "corpus.yaml", CorpusConfig)
    assert isinstance(cfg, CorpusConfig)
    assert sum(c.fraction for c in cfg.components) == pytest.approx(1.0)


# --- builder: local ---


def test_build_local_assembles_mix(tmp_path):
    line = "x" * 9 + "\n"  # 10 bytes per line
    big = line * 20  # 200 bytes each, bigger than every target
    pa = _write(tmp_path, "a.txt", big)
    pb = _write(tmp_path, "b.txt", big)
    pc = _write(tmp_path, "c.txt", big)
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=100,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[
            _local_component("a", str(pa), 0.5),
            _local_component("b", str(pb), 0.3),
            _local_component("c", str(pc), 0.2),
        ],
    )
    results = build(cfg)
    assert set(results) == {"a", "b", "c"}
    for name in ("a", "b", "c"):
        assert (out_dir / "train" / f"{name}.txt").exists()
    assert results["a"] <= 50
    assert results["b"] <= 30
    assert results["c"] <= 20
    assert abs(sum(results.values()) - 100) <= 5


def test_build_local_exhausted_source(tmp_path):
    line = "x" * 9 + "\n"  # 10 bytes per line
    small = line * 3  # 30 bytes, smaller than the 100-byte target
    pa = _write(tmp_path, "a.txt", small)
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=100,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[_local_component("a", str(pa))],
    )
    results = build(cfg)
    assert results["a"] == 30
    assert (out_dir / "train" / "a.txt").read_text() == small


# --- builder: hf (mocked stream, no network) ---


def test_build_hf_streams_to_target(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]  # each -> 5 bytes with \n
    cfg = CorpusConfig(
        total_bytes=10,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[
            ComponentConfig(
                name="web",
                fraction=1.0,
                source=HfSourceConfig(type="hf", dataset="some/ds", text_field="text"),
            ),
        ],
    )
    with (
        patch("kestrel.corpus.builder.load_dataset", return_value={"train": rows}),
        patch("truststore.inject_into_ssl"),
    ):
        results = build(cfg)
    # target 10: "aaaa\n" (5) + "bbbb\n" (5) -> stop before "cccc"
    assert results["web"] == 10
    assert (out_dir / "train" / "web.txt").read_text() == "aaaa\nbbbb\n"


def test_build_hf_jsonl_when_no_text_field(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"prompt": "hi", "completion": "there"}, {"prompt": "a", "completion": "b"}]
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[
            ComponentConfig(
                name="jsonl",
                fraction=1.0,
                source=HfSourceConfig(type="hf", dataset="some/ds"),
            ),
        ],
    )
    with (
        patch("kestrel.corpus.builder.load_dataset", return_value={"train": rows}),
        patch("truststore.inject_into_ssl"),
    ):
        results = build(cfg)
    content = (out_dir / "train" / "jsonl.txt").read_text().splitlines()
    assert len(content) == 2
    assert json.loads(content[0]) == {"prompt": "hi", "completion": "there"}
    assert results["jsonl"] > 0


# --- builder: train/val(/test) split ---


def test_split_no_leakage_and_complete(tmp_path):
    lines = [f"line{i}" for i in range(200)]
    pa = _write(tmp_path, "a.txt", "".join(item + "\n" for item in lines))
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=100000,
        output_dir=str(out_dir),
        val_fraction=0.3,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    train = (out_dir / "train" / "a.txt").read_text().splitlines()
    val = (out_dir / "val" / "a.txt").read_text().splitlines()
    assert not (set(train) & set(val))  # no document in both splits
    assert sorted(train + val) == sorted(lines)  # nothing lost


def test_split_deterministic(tmp_path):
    lines = [f"line{i}" for i in range(200)]
    pa = _write(tmp_path, "a.txt", "".join(item + "\n" for item in lines))

    def build_into(out: Path) -> tuple[str, str]:
        cfg = CorpusConfig(
            total_bytes=100000,
            output_dir=str(out),
            val_fraction=0.3,
            components=[_local_component("a", str(pa))],
        )
        build(cfg)
        return (out / "train" / "a.txt").read_text(), (out / "val" / "a.txt").read_text()

    assert build_into(tmp_path / "out1") == build_into(tmp_path / "out2")


def test_split_ratio_approx(tmp_path):
    lines = [f"line{i}" for i in range(4000)]
    pa = _write(tmp_path, "a.txt", "".join(item + "\n" for item in lines))
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000000,
        output_dir=str(out_dir),
        val_fraction=0.25,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    train = (out_dir / "train" / "a.txt").read_text().splitlines()
    val = (out_dir / "val" / "a.txt").read_text().splitlines()
    val_frac = len(val) / (len(train) + len(val))
    assert abs(val_frac - 0.25) < 0.05


def test_test_split_carved_out(tmp_path):
    lines = [f"line{i}" for i in range(4000)]
    pa = _write(tmp_path, "a.txt", "".join(item + "\n" for item in lines))
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000000,
        output_dir=str(out_dir),
        val_fraction=0.1,
        test_fraction=0.1,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    train = (out_dir / "train" / "a.txt").read_text().splitlines()
    val = (out_dir / "val" / "a.txt").read_text().splitlines()
    test = (out_dir / "test" / "a.txt").read_text().splitlines()
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert sorted(train + val + test) == sorted(lines)


def test_no_test_dir_when_zero(tmp_path):
    pa = _write(tmp_path, "a.txt", "line0\nline1\n")
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        val_fraction=0.5,
        test_fraction=0.0,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    assert (out_dir / "train").is_dir()
    assert (out_dir / "val").is_dir()
    assert not (out_dir / "test").exists()
