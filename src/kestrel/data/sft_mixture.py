"""Build the M2 SFT training mixture from prepared per-source JSONL files."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from kestrel.common.config import BaseConfig

_SOURCE_ORDER: tuple[str, ...] = (
    "assistant_public",
    "gsm8k_math",
    "tool_local",
    "tool_public",
    "internal_llm",
)


class MixtureRecipe(BaseConfig):
    """Requested row count for each SFT source in one mixture."""

    assistant_public: int = Field(default=0, ge=0)
    gsm8k_math: int = Field(default=0, ge=0)
    tool_local: int = Field(default=0, ge=0)
    tool_public: int = Field(default=0, ge=0)
    internal_llm: int = Field(default=0, ge=0)


class MixtureConfig(BaseConfig):
    """Strict settings for building the SFT mixture."""

    input_dir: str
    output_dir: str
    seed: int
    recipe: MixtureRecipe
    output_name: str = "sft-50k.jsonl"
    manifest_name: str = "manifest.json"
    deficit_policy: Literal["allow", "fail", "redistribute"] = "allow"
    fallback_when_internal_missing: bool = True
    fallback_recipe: MixtureRecipe | None = None


class MixtureSourceManifest(BaseConfig):
    """Manifest entry for one source used in the SFT mixture."""

    source: str
    input_path: str
    requested_rows: int
    available_rows: int
    selected_rows: int
    deficit_rows: int
    extra_rows: int
    sha256: str | None


class MixtureManifest(BaseConfig):
    """Manifest for one built SFT mixture."""

    recipe_used: Literal["default", "fallback"]
    seed: int
    config_sha256: str
    output_path: str
    manifest_path: str
    output_sha256: str
    requested_total: int
    total_rows: int
    sources: dict[str, MixtureSourceManifest]


@dataclass
class _SourcePool:
    source: str
    path: Path
    lines: list[str]
    sha256: str | None
    requested_rows: int
    selected_lines: list[str]
    extra_lines: list[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fin:
        return [line.rstrip("\n") for line in fin if line.strip()]


def _has_rows(path: Path) -> bool:
    return bool(_load_lines(path))


def _config_sha256(config: MixtureConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _choose_recipe(config: MixtureConfig) -> tuple[MixtureRecipe, Literal["default", "fallback"]]:
    internal_path = Path(config.input_dir) / "internal_llm.jsonl"
    internal_missing = config.recipe.internal_llm > 0 and not _has_rows(internal_path)
    if (
        config.fallback_when_internal_missing
        and config.fallback_recipe is not None
        and internal_missing
    ):
        return config.fallback_recipe, "fallback"
    return config.recipe, "default"


def _build_pool(
    source: str,
    input_dir: str,
    requested_rows: int,
    seed: int,
    source_index: int,
) -> _SourcePool:
    path = Path(input_dir) / f"{source}.jsonl"
    lines = _load_lines(path)
    sha256 = _sha256_file(path) if path.exists() else None
    shuffled = list(lines)
    random.Random(seed + source_index).shuffle(shuffled)
    selected_lines = shuffled[:requested_rows]
    extra_lines = shuffled[requested_rows:] if requested_rows > 0 else []
    return _SourcePool(
        source=source,
        path=path,
        lines=lines,
        sha256=sha256,
        requested_rows=requested_rows,
        selected_lines=selected_lines,
        extra_lines=extra_lines,
    )


def _selected_rows(pool: _SourcePool) -> int:
    return len(pool.selected_lines)


def _deficit_rows(pool: _SourcePool) -> int:
    return max(0, pool.requested_rows - _selected_rows(pool))


def _redistribute_deficit(pools: list[_SourcePool], deficit: int) -> int:
    remaining = deficit
    for pool in pools:
        if remaining <= 0:
            break
        take = min(remaining, len(pool.extra_lines))
        pool.selected_lines.extend(pool.extra_lines[:take])
        pool.extra_lines = pool.extra_lines[take:]
        remaining -= take
    return remaining


def _source_manifest(pool: _SourcePool) -> MixtureSourceManifest:
    selected_rows = _selected_rows(pool)
    return MixtureSourceManifest(
        source=pool.source,
        input_path=str(pool.path),
        requested_rows=pool.requested_rows,
        available_rows=len(pool.lines),
        selected_rows=selected_rows,
        deficit_rows=max(0, pool.requested_rows - selected_rows),
        extra_rows=max(0, selected_rows - pool.requested_rows),
        sha256=pool.sha256,
    )


def build_mixture(config: MixtureConfig) -> MixtureManifest:
    """Combine per-source SFT JSONL files into one shuffled mixture."""
    recipe, recipe_used = _choose_recipe(config)
    pools = [
        _build_pool(source, config.input_dir, getattr(recipe, source), config.seed, index)
        for index, source in enumerate(_SOURCE_ORDER)
    ]

    requested_total = sum(pool.requested_rows for pool in pools)
    deficit = requested_total - sum(_selected_rows(pool) for pool in pools)
    if deficit > 0:
        if config.deficit_policy == "fail":
            short_sources = [pool.source for pool in pools if _deficit_rows(pool) > 0]
            msg = f"SFT mixture source deficit: {', '.join(short_sources)}"
            raise ValueError(msg)
        if config.deficit_policy == "redistribute":
            remaining = _redistribute_deficit(pools, deficit)
            if remaining > 0:
                msg = f"SFT mixture redistribution left {remaining} rows short"
                raise ValueError(msg)

    combined = [line for pool in pools for line in pool.selected_lines]
    random.Random(config.seed + 10_000).shuffle(combined)

    output_path = Path(config.output_dir) / config.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fin:
        for line in combined:
            fin.write(line + "\n")

    manifest = MixtureManifest(
        recipe_used=recipe_used,
        seed=config.seed,
        config_sha256=_config_sha256(config),
        output_path=str(output_path),
        manifest_path=str(Path(config.output_dir) / config.manifest_name),
        output_sha256=_sha256_file(output_path),
        requested_total=requested_total,
        total_rows=len(combined),
        sources={pool.source: _source_manifest(pool) for pool in pools},
    )
    manifest_path = Path(config.output_dir) / config.manifest_name
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
