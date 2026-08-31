"""Shared M2 SFT data preparation: configs, sampling, filtering, and manifests."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import Field
from tokenizers import Tokenizer

from kestrel.common.config import BaseConfig
from kestrel.data.sft_chat import render_sft
from kestrel.data.sft_prepare_gsm8k import convert_gsm8k_row, load_gsm8k_rows
from kestrel.data.sft_prepare_public import convert_smol_row, load_smol_rows
from kestrel.data.sft_schema import SFTRow


class AssistantSourceConfig(BaseConfig):
    """Settings for the public assistant SFT source."""

    dataset_id: str = "HuggingFaceTB/smol-smoltalk"
    split: str = "train"
    source: str = "assistant_public"
    target_rows: int = Field(default=22_500, gt=0)


class Gsm8kSourceConfig(BaseConfig):
    """Settings for the GSM8K math SFT source."""

    dataset_id: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    source: str = "gsm8k_math"
    target_rows: int = Field(default=7_500, gt=0)


class SFTDataConfig(BaseConfig):
    """Strict settings for M2 SFT data preparation."""

    output_dir: str = "data/sft/raw"
    tokenizer_path: str = "checkpoints/tokenizer/tokenizer.json"
    context_length: int = Field(default=1024, gt=1)
    seed: int = 42
    assistant: AssistantSourceConfig = Field(default_factory=AssistantSourceConfig)
    gsm8k: Gsm8kSourceConfig = Field(default_factory=Gsm8kSourceConfig)


class SourceManifest(BaseConfig):
    """Manifest entry for one prepared SFT source."""

    source: str
    dataset_id: str
    split: str
    seed: int
    requested_rows: int
    candidate_rows: int
    written_rows: int
    filtered_rows: int
    output_path: str
    sha256: str
    dataset_config: str | None = None


def reservoir_sample(rows: Iterator[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically sample up to ``k`` rows from a single-pass stream."""
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < k:
            reservoir.append(row)
        else:
            replace_at = rng.randint(0, index)
            if replace_at < k:
                reservoir[replace_at] = row
    return reservoir


def _passes_context_filter(row: SFTRow, tokenizer: Tokenizer, context_length: int) -> bool:
    rendered = render_sft(row, tokenizer)
    return 2 <= len(rendered.token_ids) <= context_length and any(rendered.loss_mask)


def _write_jsonl(path: Path, rows: list[SFTRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fin:
        for row in rows:
            fin.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return len(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_manifest(output_dir: str | Path, manifest: SourceManifest) -> Path:
    path = Path(output_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data[manifest.source] = manifest.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def prepare_rows(
    *,
    source: str,
    dataset_id: str,
    split: str,
    seed: int,
    target_rows: int,
    tokenizer_path: str,
    context_length: int,
    output_dir: str,
    load_rows: Callable[[], Iterator[dict[str, Any]]],
    convert_row: Callable[[dict[str, Any]], SFTRow | None],
    dataset_config: str | None = None,
) -> SourceManifest:
    """Stream, sample, convert, filter, and write one SFT source."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    candidates = reservoir_sample(load_rows(), target_rows, seed)

    rows: list[SFTRow] = []
    filtered_rows = 0
    for raw in candidates:
        row = convert_row(raw)
        if row is None or not _passes_context_filter(row, tokenizer, context_length):
            filtered_rows += 1
            continue
        rows.append(row)

    output_path = Path(output_dir) / f"{source}.jsonl"
    written_rows = _write_jsonl(output_path, rows)

    manifest = SourceManifest(
        source=source,
        dataset_id=dataset_id,
        split=split,
        seed=seed,
        requested_rows=target_rows,
        candidate_rows=len(candidates),
        written_rows=written_rows,
        filtered_rows=filtered_rows,
        output_path=str(output_path),
        sha256=_sha256_file(output_path),
        dataset_config=dataset_config,
    )
    _update_manifest(output_dir, manifest)
    return manifest


def prepare_assistant(config: SFTDataConfig) -> SourceManifest:
    """Prepare the public assistant source."""
    source = config.assistant
    return prepare_rows(
        source=source.source,
        dataset_id=source.dataset_id,
        split=source.split,
        seed=config.seed,
        target_rows=source.target_rows,
        tokenizer_path=config.tokenizer_path,
        context_length=config.context_length,
        output_dir=config.output_dir,
        load_rows=lambda: load_smol_rows(source.dataset_id, source.split),
        convert_row=lambda raw: convert_smol_row(raw, source.source),
    )


def prepare_gsm8k(config: SFTDataConfig) -> SourceManifest:
    """Prepare the GSM8K math source."""
    source = config.gsm8k
    return prepare_rows(
        source=source.source,
        dataset_id=source.dataset_id,
        split=source.split,
        seed=config.seed,
        target_rows=source.target_rows,
        tokenizer_path=config.tokenizer_path,
        context_length=config.context_length,
        output_dir=config.output_dir,
        load_rows=lambda: load_gsm8k_rows(source.dataset_id, source.dataset_config, source.split),
        convert_row=lambda raw: convert_gsm8k_row(raw, source.source),
        dataset_config=source.dataset_config,
    )


def prepare_all(config: SFTDataConfig) -> dict[str, SourceManifest]:
    """Prepare both public SFT sources."""
    manifests = [prepare_assistant(config), prepare_gsm8k(config)]
    return {manifest.source: manifest for manifest in manifests}
