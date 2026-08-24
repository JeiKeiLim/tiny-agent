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
        components=[
            ComponentConfig(
                name="a",
                fraction=0.5,
                source=LocalSourceConfig(type="local", path=str(pa)),
            ),
            ComponentConfig(
                name="b",
                fraction=0.3,
                source=LocalSourceConfig(type="local", path=str(pb)),
            ),
            ComponentConfig(
                name="c",
                fraction=0.2,
                source=LocalSourceConfig(type="local", path=str(pc)),
            ),
        ],
    )
    results = build(cfg)
    assert set(results) == {"a", "b", "c"}
    for name in ("a", "b", "c"):
        assert (out_dir / f"{name}.txt").exists()
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
        components=[
            ComponentConfig(
                name="a",
                fraction=1.0,
                source=LocalSourceConfig(type="local", path=str(pa)),
            ),
        ],
    )
    results = build(cfg)
    assert results["a"] == 30
    assert (out_dir / "a.txt").read_text() == small


# --- builder: hf (mocked stream, no network) ---


def test_build_hf_streams_to_target(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]  # each -> 5 bytes with \n
    cfg = CorpusConfig(
        total_bytes=10,
        output_dir=str(out_dir),
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
    assert (out_dir / "web.txt").read_text() == "aaaa\nbbbb\n"


def test_build_hf_jsonl_when_no_text_field(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"prompt": "hi", "completion": "there"}, {"prompt": "a", "completion": "b"}]
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
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
    content = (out_dir / "jsonl.txt").read_text().splitlines()
    assert len(content) == 2
    assert json.loads(content[0]) == {"prompt": "hi", "completion": "there"}
    assert results["jsonl"] > 0
