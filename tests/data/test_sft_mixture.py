"""Tests for the M2 SFT mixture builder."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import pytest

from kestrel.data.sft_mixture import MixtureConfig, MixtureRecipe, build_mixture

_SOURCES = (
    "assistant_public",
    "gsm8k_math",
    "tool_local",
    "tool_public",
    "internal_llm",
)


def _write_source(input_dir: Path, source: str, count: int) -> None:
    path = input_dir / f"{source}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fin:
        for index in range(count):
            fin.write(json.dumps({"source": source, "index": index}) + "\n")


def _write_all(input_dir: Path, counts: dict[str, int]) -> None:
    for source in _SOURCES:
        _write_source(input_dir, source, counts.get(source, 0))


def _recipe(
    assistant_public: int = 0,
    gsm8k_math: int = 0,
    tool_local: int = 0,
    tool_public: int = 0,
    internal_llm: int = 0,
) -> MixtureRecipe:
    return MixtureRecipe(
        assistant_public=assistant_public,
        gsm8k_math=gsm8k_math,
        tool_local=tool_local,
        tool_public=tool_public,
        internal_llm=internal_llm,
    )


def _config(
    input_dir: Path,
    output_dir: Path,
    *,
    recipe: MixtureRecipe,
    fallback_recipe: MixtureRecipe | None = None,
    deficit_policy: Literal["allow", "fail", "redistribute"] = "allow",
    fallback_when_internal_missing: bool = True,
    seed: int = 11,
) -> MixtureConfig:
    return MixtureConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        seed=seed,
        recipe=recipe,
        deficit_policy=deficit_policy,
        fallback_when_internal_missing=fallback_when_internal_missing,
        fallback_recipe=fallback_recipe,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_counts(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row["source"]) for row in rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_build_mixture_default_uses_requested_counts(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 5,
            "gsm8k_math": 3,
            "tool_local": 4,
            "tool_public": 2,
            "internal_llm": 2,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
    )

    manifest = build_mixture(config)
    rows = _read_jsonl(Path(manifest.output_path))

    assert manifest.recipe_used == "default"
    assert manifest.requested_total == 6
    assert manifest.total_rows == 6
    assert len(rows) == 6
    assert _source_counts(rows) == {
        "assistant_public": 2,
        "gsm8k_math": 1,
        "tool_local": 1,
        "tool_public": 1,
        "internal_llm": 1,
    }

    manifest_data = json.loads(Path(manifest.manifest_path).read_text(encoding="utf-8"))
    assert manifest_data["sources"]["assistant_public"]["requested_rows"] == 2
    assert manifest_data["sources"]["assistant_public"]["selected_rows"] == 2
    assert manifest_data["sources"]["assistant_public"]["deficit_rows"] == 0


def test_build_mixture_fallback_when_internal_missing(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 5,
            "gsm8k_math": 3,
            "tool_local": 4,
            "tool_public": 5,
            "internal_llm": 0,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
        fallback_recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=2,
        ),
    )

    manifest = build_mixture(config)
    rows = _read_jsonl(Path(manifest.output_path))

    assert manifest.recipe_used == "fallback"
    assert manifest.total_rows == 6
    assert _source_counts(rows) == {
        "assistant_public": 2,
        "gsm8k_math": 1,
        "tool_local": 1,
        "tool_public": 2,
    }


def test_build_mixture_does_not_fallback_when_disabled(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 5,
            "gsm8k_math": 3,
            "tool_local": 4,
            "tool_public": 2,
            "internal_llm": 0,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
        fallback_recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=2,
        ),
        fallback_when_internal_missing=False,
    )

    manifest = build_mixture(config)

    assert manifest.recipe_used == "default"
    assert manifest.total_rows == 5
    assert manifest.sources["internal_llm"].deficit_rows == 1


def test_build_mixture_records_deficit_when_source_is_short(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 1,
            "gsm8k_math": 3,
            "tool_local": 4,
            "tool_public": 2,
            "internal_llm": 2,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
    )

    manifest = build_mixture(config)

    assert manifest.total_rows == 5
    assert manifest.sources["assistant_public"].available_rows == 1
    assert manifest.sources["assistant_public"].selected_rows == 1
    assert manifest.sources["assistant_public"].deficit_rows == 1


def test_build_mixture_records_missing_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    for source in _SOURCES:
        if source != "gsm8k_math":
            _write_source(input_dir, source, 3)
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=1,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
    )

    manifest = build_mixture(config)

    missing = manifest.sources["gsm8k_math"]
    assert manifest.total_rows == 4
    assert missing.available_rows == 0
    assert missing.selected_rows == 0
    assert missing.deficit_rows == 1
    assert missing.sha256 is None


def test_build_mixture_fails_on_deficit_when_policy_is_fail(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 1,
            "gsm8k_math": 3,
            "tool_local": 4,
            "tool_public": 2,
            "internal_llm": 2,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
        deficit_policy="fail",
    )

    with pytest.raises(ValueError, match="assistant_public"):
        build_mixture(config)


def test_build_mixture_redistributes_deficit_to_surplus_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 1,
            "gsm8k_math": 5,
            "tool_local": 0,
            "tool_public": 0,
            "internal_llm": 0,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(assistant_public=2, gsm8k_math=2),
        deficit_policy="redistribute",
    )

    manifest = build_mixture(config)
    rows = _read_jsonl(Path(manifest.output_path))

    assert manifest.total_rows == 4
    assert _source_counts(rows) == {"assistant_public": 1, "gsm8k_math": 3}
    assert manifest.sources["assistant_public"].deficit_rows == 1
    assert manifest.sources["gsm8k_math"].extra_rows == 1


def test_build_mixture_redistribution_fails_when_surplus_is_insufficient(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 1,
            "gsm8k_math": 1,
            "tool_local": 0,
            "tool_public": 0,
            "internal_llm": 0,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(assistant_public=2, gsm8k_math=2),
        deficit_policy="redistribute",
    )

    with pytest.raises(ValueError, match="redistribution"):
        build_mixture(config)


def test_build_mixture_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    _write_all(
        input_dir,
        {
            "assistant_public": 10,
            "gsm8k_math": 10,
            "tool_local": 10,
            "tool_public": 10,
            "internal_llm": 10,
        },
    )
    recipe = _recipe(
        assistant_public=4,
        gsm8k_math=3,
        tool_local=3,
        tool_public=2,
        internal_llm=2,
    )

    output_dir = tmp_path / "mixture"
    first = build_mixture(_config(input_dir, output_dir, recipe=recipe, seed=17))
    second = build_mixture(_config(input_dir, output_dir, recipe=recipe, seed=17))

    assert first.output_sha256 == second.output_sha256
    assert first.config_sha256 == second.config_sha256


def test_build_mixture_manifest_records_hashes(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "mixture"
    _write_all(
        input_dir,
        {
            "assistant_public": 5,
            "gsm8k_math": 3,
            "tool_local": 4,
            "tool_public": 2,
            "internal_llm": 2,
        },
    )
    config = _config(
        input_dir,
        output_dir,
        recipe=_recipe(
            assistant_public=2,
            gsm8k_math=1,
            tool_local=1,
            tool_public=1,
            internal_llm=1,
        ),
    )

    manifest = build_mixture(config)

    assert manifest.sources["assistant_public"].sha256 == _sha256_file(
        input_dir / "assistant_public.jsonl"
    )
    assert manifest.output_sha256 == _sha256_file(Path(manifest.output_path))
    assert len(manifest.config_sha256) == 64
