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


def _write_jsonl(tmp_path: Path, name: str, docs: list[str]) -> Path:
    p = tmp_path / name
    p.write_text(
        "".join(json.dumps({"text": doc}, ensure_ascii=False) + "\n" for doc in docs),
        encoding="utf-8",
    )
    return p


def _local_component(name: str, path: str, fraction: float = 1.0) -> ComponentConfig:
    return ComponentConfig(
        name=name,
        fraction=fraction,
        source=LocalSourceConfig(type="local", path=path),
    )


def _read_jsonl_docs(path: Path) -> list[str]:
    docs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            docs.append(json.loads(line)["text"])
    return docs


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
    assert cfg.output_format == "jsonl"
    assert cfg.tokenizer_path is None
    assert [c.name for c in cfg.components] == ["web", "code"]
    assert isinstance(cfg.components[0].source, LocalSourceConfig)
    assert cfg.components[0].source.path == "/tmp/web.txt"
    assert isinstance(cfg.components[1].source, HfSourceConfig)
    assert cfg.components[1].source.dataset == "some/ds"
    assert cfg.val_fraction == 0.1  # default
    assert cfg.test_fraction == 0.0  # default


def test_rejects_bad_output_format(tmp_path):
    p = _write(
        tmp_path,
        "corpus.yaml",
        "total_bytes: 1000\n"
        "output_format: parquet\n"
        "components:\n"
        "  - name: web\n"
        "    fraction: 1.0\n"
        "    source:\n"
        "      type: local\n"
        "      path: /tmp/web.txt\n",
    )
    with pytest.raises(ValidationError):
        load_config(p, CorpusConfig)


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
    assert cfg.output_format == "jsonl"
    assert sum(c.fraction for c in cfg.components) == pytest.approx(1.0)


# --- builder: local ---


def test_build_local_default_jsonl_preserves_documents(tmp_path):
    pa = _write(tmp_path, "a.txt", "alpha\nbeta\n")
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    path = out_dir / "train" / "a.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["domain"] for row in rows] == ["a", "a"]
    assert [row["text"] for row in rows] == ["alpha", "beta"]


def test_build_local_assembles_mix(tmp_path):
    line = "x" * 9 + "\n"
    big = line * 200
    pa = _write(tmp_path, "a.txt", big)
    pb = _write(tmp_path, "b.txt", big)
    pc = _write(tmp_path, "c.txt", big)
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
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
        assert (out_dir / "train" / f"{name}.jsonl").exists()
    assert results["a"] <= 500 + 100
    assert results["b"] <= 300 + 100
    assert results["c"] <= 200 + 100
    assert sum(results.values()) > 800


def test_build_local_exhausted_source(tmp_path):
    pa = _write(tmp_path, "a.txt", "a\nb\nc\n")
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[_local_component("a", str(pa))],
    )
    results = build(cfg)
    path = out_dir / "train" / "a.jsonl"
    assert _read_jsonl_docs(path) == ["a", "b", "c"]
    assert results["a"] == len(path.read_text(encoding="utf-8").encode())


def test_build_local_jsonl_preserves_multiline_document(tmp_path):
    code = "def f():\n    return 1\n"
    pa = _write_jsonl(tmp_path, "a.jsonl", [code, "other"])
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        val_fraction=0.0,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    path = out_dir / "train" / "a.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    assert _read_jsonl_docs(path) == [code, "other"]


def test_build_local_txt_output_is_legacy(tmp_path):
    pa = _write(tmp_path, "a.txt", "alpha\nbeta\n")
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        output_format="txt",
        val_fraction=0.0,
        components=[_local_component("a", str(pa))],
    )
    build(cfg)
    assert (out_dir / "train" / "a.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


# --- builder: hf (mocked stream, no network) ---


def test_build_hf_jsonl_preserves_multiline_document(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"text": "line1\nline2"}, {"text": "other"}]
    cfg = CorpusConfig(
        total_bytes=1000,
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
        build(cfg)
    path = out_dir / "train" / "web.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    assert [json.loads(line)["text"] for line in raw] == ["line1\nline2", "other"]


def test_build_hf_streams_to_target(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]
    cfg = CorpusConfig(
        total_bytes=1,
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
    path = out_dir / "train" / "web.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 1
    assert results["web"] == len(raw[0].encode()) + 1


def test_build_hf_jsonl_when_no_text_field(tmp_path):
    out_dir = tmp_path / "out"
    rows = [{"prompt": "hi", "completion": "there"}]
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
        build(cfg)
    path = out_dir / "train" / "jsonl.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 1
    outer = json.loads(raw[0])
    assert json.loads(outer["text"]) == rows[0]


# --- builder: manifest ---


def test_manifest_counts_and_estimated_tokens(tmp_path):
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
    for split in ("train", "val"):
        manifest = json.loads((out_dir / split / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["split"] == split
        assert manifest["output_format"] == "jsonl"
        file_entry = manifest["files"][0]
        docs = _read_jsonl_docs(out_dir / split / "a.jsonl")
        assert file_entry["doc_count"] == len(docs)
        assert file_entry["byte_count"] == (out_dir / split / "a.jsonl").stat().st_size
        assert file_entry["token_count"] is None
        assert file_entry["estimated_token_count"] == file_entry["byte_count"] // 4
        assert manifest["total_doc_count"] == len(docs)
        assert manifest["total_token_count"] is None
        assert manifest["total_estimated_token_count"] == manifest["total_byte_count"] // 4


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
    train = _read_jsonl_docs(out_dir / "train" / "a.jsonl")
    val = _read_jsonl_docs(out_dir / "val" / "a.jsonl")
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
        return (
            (out / "train" / "a.jsonl").read_text(encoding="utf-8"),
            (out / "val" / "a.jsonl").read_text(encoding="utf-8"),
        )

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
    train = _read_jsonl_docs(out_dir / "train" / "a.jsonl")
    val = _read_jsonl_docs(out_dir / "val" / "a.jsonl")
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
    train = _read_jsonl_docs(out_dir / "train" / "a.jsonl")
    val = _read_jsonl_docs(out_dir / "val" / "a.jsonl")
    test = _read_jsonl_docs(out_dir / "test" / "a.jsonl")
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


# --- builder: idempotent rebuild ---


def _idempotent_fixture(tmp_path: Path) -> tuple[Path, CorpusConfig]:
    pa = _write(tmp_path, "a.txt", "alpha\nbeta\n")
    out_dir = tmp_path / "out"
    cfg = CorpusConfig(
        total_bytes=1000,
        output_dir=str(out_dir),
        val_fraction=0.5,
        components=[_local_component("a", str(pa))],
    )
    return out_dir, cfg


def test_build_skips_when_corpus_complete(tmp_path):
    _, cfg = _idempotent_fixture(tmp_path)
    first = build(cfg)

    with patch("kestrel.corpus.builder._build_local", return_value=0) as mock:
        second = build(cfg)

    mock.assert_not_called()
    assert second == first


def test_build_rebuilds_when_config_fingerprint_changes(tmp_path):
    _, cfg = _idempotent_fixture(tmp_path)
    build(cfg)
    stale = cfg.model_copy(update={"seed": cfg.seed + 1})

    with patch("kestrel.corpus.builder._build_local", return_value=0) as mock:
        build(stale)

    mock.assert_called()


def test_build_rebuilds_when_manifest_missing(tmp_path):
    out_dir, cfg = _idempotent_fixture(tmp_path)
    build(cfg)
    (out_dir / "train" / "manifest.json").unlink()

    with patch("kestrel.corpus.builder._build_local", return_value=0) as mock:
        build(cfg)

    mock.assert_called()


def test_build_rebuilds_when_corpus_file_missing(tmp_path):
    out_dir, cfg = _idempotent_fixture(tmp_path)
    build(cfg)
    (out_dir / "train" / "a.jsonl").unlink()

    with patch("kestrel.corpus.builder._build_local", return_value=0) as mock:
        build(cfg)

    mock.assert_called()


def test_build_rebuilds_when_corpus_file_size_mismatches(tmp_path):
    out_dir, cfg = _idempotent_fixture(tmp_path)
    build(cfg)
    path = out_dir / "train" / "a.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")

    with patch("kestrel.corpus.builder._build_local", return_value=0) as mock:
        build(cfg)

    mock.assert_called()


def test_build_force_rebuilds_existing_corpus(tmp_path):
    _, cfg = _idempotent_fixture(tmp_path)
    build(cfg)

    with patch("kestrel.corpus.builder._build_local", return_value=0) as mock:
        build(cfg, force=True)

    mock.assert_called()
