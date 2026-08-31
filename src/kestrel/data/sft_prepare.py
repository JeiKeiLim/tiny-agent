"""Shared M2 SFT data preparation: configs, sampling, filtering, and manifests."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import Field
from tokenizers import Tokenizer

from kestrel.common.config import BaseConfig
from kestrel.data.sft_chat import render_sft
from kestrel.data.sft_internal_llm import (
    InternalLLMConfig,
    create_llm_client,
    generate_internal_llm_rows,
)
from kestrel.data.sft_mixture import MixtureConfig
from kestrel.data.sft_prepare_gsm8k import convert_gsm8k_row, load_gsm8k_rows
from kestrel.data.sft_prepare_public import convert_smol_row, load_smol_rows
from kestrel.data.sft_public_tool import PublicToolNormalizer, load_public_tool_rows
from kestrel.data.sft_schema import SFTRow
from kestrel.data.sft_tool_generator import (
    ToolEvalBreakdown,
    ToolGeneratorConfig,
    ToolTrainBreakdown,
    generate_tool_eval,
    generate_tool_train,
)
from kestrel.tools.schema_sampler import m2_eval_tool_names


class AssistantSourceConfig(BaseConfig):
    """Settings for the public assistant SFT source."""

    dataset_id: str = "HuggingFaceTB/smol-smoltalk"
    split: str = "train"
    source: str = "assistant_public"
    target_rows: int = Field(default=22_500, gt=0)
    max_candidate_rows: int | None = Field(default=None, gt=0)


class Gsm8kSourceConfig(BaseConfig):
    """Settings for the GSM8K math SFT source."""

    dataset_id: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    source: str = "gsm8k_math"
    target_rows: int = Field(default=7_500, gt=0)


class ToolSourceConfig(BaseConfig):
    """Settings for the local rule-based tool SFT source."""

    source: str = "tool_local"
    min_tools: int = Field(default=3, ge=2, le=5)
    max_tools: int = Field(default=5, ge=3, le=5)
    train: ToolTrainBreakdown = Field(default_factory=ToolTrainBreakdown)
    eval: ToolEvalBreakdown = Field(default_factory=ToolEvalBreakdown)


class PublicToolSourceConfig(BaseConfig):
    """Settings for the public tool-calling SFT source."""

    dataset_id: str = "argilla/apigen-function-calling"
    split: str = "train"
    source: str = "tool_public"
    target_rows: int = Field(default=5_000, gt=0)
    max_tools: int = Field(default=5, ge=1, le=5)
    max_query_chars: int = Field(default=512, gt=0)
    max_tool_chars: int = Field(default=2_048, gt=0)
    max_list_items: int = Field(default=10, gt=0)


class SFTDataEvalConfig(BaseConfig):
    """Strict settings for the held-out SFT eval bundle."""

    output_dir: str = "data/sft/eval"
    seed: int = 42
    assistant_dataset_id: str = "HuggingFaceTB/smol-smoltalk"
    assistant_split: str = "test"
    assistant_source: str = "assistant_eval"
    assistant_target_rows: int = Field(default=200, gt=0)
    assistant_max_candidate_rows: int | None = Field(default=10_000, gt=0)
    gsm8k_dataset_id: str = "openai/gsm8k"
    gsm8k_dataset_config: str = "main"
    gsm8k_split: str = "test"
    gsm8k_source: str = "gsm8k_eval"
    gsm8k_target_rows: int = Field(default=500, gt=0)
    tool_eval: bool = True


class SFTDataConfig(BaseConfig):
    """Strict settings for M2 SFT data preparation."""

    output_dir: str = "data/sft/raw"
    tokenizer_path: str = "checkpoints/tokenizer/tokenizer.json"
    context_length: int = Field(default=1024, gt=1)
    seed: int = 42
    assistant: AssistantSourceConfig = Field(default_factory=AssistantSourceConfig)
    gsm8k: Gsm8kSourceConfig = Field(default_factory=Gsm8kSourceConfig)
    tool: ToolSourceConfig = Field(default_factory=ToolSourceConfig)
    public_tool: PublicToolSourceConfig = Field(default_factory=PublicToolSourceConfig)
    internal_llm: InternalLLMConfig = Field(default_factory=InternalLLMConfig)
    mixture: MixtureConfig | None = None
    eval: SFTDataEvalConfig | None = None


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
    dropped_rows: int = 0
    model_env: str | None = None
    prompt_version: str | None = None
    generated_counts: dict[str, int] | None = None


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
    max_candidate_rows: int | None = None,
) -> SourceManifest:
    """Stream, convert, filter, and write one SFT source until target rows are valid."""
    tokenizer = Tokenizer.from_file(tokenizer_path)

    rows: list[SFTRow] = []
    candidate_rows = 0
    filtered_rows = 0
    for raw in load_rows():
        if max_candidate_rows is not None and candidate_rows >= max_candidate_rows:
            break
        candidate_rows += 1
        row = convert_row(raw)
        if row is None or not _passes_context_filter(row, tokenizer, context_length):
            filtered_rows += 1
            continue
        rows.append(row)
        if len(rows) == target_rows:
            break

    output_path = Path(output_dir) / f"{source}.jsonl"
    written_rows = _write_jsonl(output_path, rows)

    manifest = SourceManifest(
        source=source,
        dataset_id=dataset_id,
        split=split,
        seed=seed,
        requested_rows=target_rows,
        candidate_rows=candidate_rows,
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
        load_rows=lambda: load_smol_rows(source.dataset_id, source.split, config.seed),
        convert_row=lambda raw: convert_smol_row(raw, source.source),
        max_candidate_rows=source.max_candidate_rows,
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
        load_rows=lambda: load_gsm8k_rows(
            source.dataset_id, source.dataset_config, source.split, config.seed
        ),
        convert_row=lambda raw: convert_gsm8k_row(raw, source.source),
        dataset_config=source.dataset_config,
    )


def write_tool_split(
    *,
    output_dir: str,
    source: str,
    split: str,
    seed: int,
    rows: list[SFTRow],
    tokenizer: Tokenizer,
    context_length: int,
) -> SourceManifest:
    kept_rows: list[SFTRow] = []
    filtered_rows = 0
    for row in rows:
        if _passes_context_filter(row, tokenizer, context_length):
            kept_rows.append(row)
        else:
            filtered_rows += 1

    output_path = Path(output_dir) / f"{source}.jsonl"
    written_rows = _write_jsonl(output_path, kept_rows)
    manifest = SourceManifest(
        source=source,
        dataset_id="local-rule-based-tool-generator",
        split=split,
        seed=seed,
        requested_rows=len(rows),
        candidate_rows=len(rows),
        written_rows=written_rows,
        filtered_rows=filtered_rows,
        output_path=str(output_path),
        sha256=_sha256_file(output_path),
    )
    _update_manifest(output_dir, manifest)
    return manifest


def prepare_tool(config: SFTDataConfig) -> dict[str, SourceManifest]:
    """Generate and write the local rule-based tool train and eval splits."""
    generator_config = ToolGeneratorConfig(
        seed=config.seed,
        min_tools=config.tool.min_tools,
        max_tools=config.tool.max_tools,
        train=config.tool.train,
        eval=config.tool.eval,
    )
    train_rows = generate_tool_train(generator_config).all_rows
    random.Random(config.seed).shuffle(train_rows)
    eval_rows = generate_tool_eval(generator_config)
    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    splits = {
        "train": (config.tool.source, "train", train_rows),
        "eval_seen": ("tool_eval_seen", "eval_seen", list(eval_rows.seen)),
        "eval_unseen": ("tool_eval_unseen", "eval_unseen", list(eval_rows.unseen)),
        "eval_no_call": ("tool_eval_no_call", "eval_no_call", list(eval_rows.no_call)),
        "eval_missing_info": (
            "tool_eval_missing_info",
            "eval_missing_info",
            list(eval_rows.missing_info),
        ),
    }
    manifests = [
        write_tool_split(
            output_dir=config.output_dir,
            source=source,
            split=split,
            seed=config.seed,
            rows=rows,
            tokenizer=tokenizer,
            context_length=config.context_length,
        )
        for split, (source, manifest_split, rows) in splits.items()
    ]
    return {manifest.source: manifest for manifest in manifests}


def _reservoir_append(
    reservoir: list[SFTRow], rng: random.Random, count: int, row: SFTRow, k: int
) -> None:
    if len(reservoir) < k:
        reservoir.append(row)
        return
    replace_at = rng.randint(0, count - 1)
    if replace_at < k:
        reservoir[replace_at] = row


def prepare_public_tool(config: SFTDataConfig) -> SourceManifest:
    """Prepare the public tool-calling source with distilabel rows preferred."""
    source = config.public_tool
    normalizer = PublicToolNormalizer(
        source=source.source,
        max_tools=source.max_tools,
        max_query_chars=source.max_query_chars,
        max_tool_chars=source.max_tool_chars,
        max_list_items=source.max_list_items,
        excluded_tool_names=m2_eval_tool_names(),
    )
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    target_rows = source.target_rows
    distilabel_rows: list[SFTRow] = []
    other_rows: list[SFTRow] = []
    distilabel_count = 0
    other_count = 0
    dropped_rows = 0
    rng_distilabel = random.Random(config.seed)
    rng_other = random.Random(config.seed + 1)

    for raw in load_public_tool_rows(source.dataset_id, source.split):
        row = normalizer.convert(raw)
        if row is None:
            dropped_rows += 1
            continue
        if raw.get("origin") == "distilabel":
            distilabel_count += 1
            _reservoir_append(distilabel_rows, rng_distilabel, distilabel_count, row, target_rows)
        else:
            other_count += 1
            _reservoir_append(other_rows, rng_other, other_count, row, target_rows)

    random.Random(config.seed + 2).shuffle(distilabel_rows)
    random.Random(config.seed + 3).shuffle(other_rows)
    sampled = distilabel_rows[:target_rows]
    if len(sampled) < target_rows:
        sampled.extend(other_rows[: target_rows - len(sampled)])

    kept_rows: list[SFTRow] = []
    filtered_rows = 0
    for row in sampled:
        if _passes_context_filter(row, tokenizer, config.context_length):
            kept_rows.append(row)
        else:
            filtered_rows += 1

    output_path = Path(config.output_dir) / f"{source.source}.jsonl"
    written_rows = _write_jsonl(output_path, kept_rows)
    manifest = SourceManifest(
        source=source.source,
        dataset_id=source.dataset_id,
        split=source.split,
        seed=config.seed,
        requested_rows=target_rows,
        candidate_rows=len(sampled),
        written_rows=written_rows,
        filtered_rows=filtered_rows,
        output_path=str(output_path),
        sha256=_sha256_file(output_path),
        dropped_rows=dropped_rows,
    )
    _update_manifest(config.output_dir, manifest)
    return manifest


def prepare_internal_llm(config: SFTDataConfig) -> SourceManifest | None:
    """Prepare the optional internal LLM source, or return None when disabled."""
    source = config.internal_llm
    if not source.enabled:
        return None

    client = create_llm_client(source)

    def _report_progress(state: dict[str, int]) -> None:
        print(
            "internal_llm: "
            f"assistant {state['assistant']}/{source.assistant_rows}, "
            f"math {state['math']}/{source.math_rows}, "
            f"tool {state['tool']}/{source.tool_rows}",
            file=sys.stderr,
        )

    debug_callback: Callable[[str, str, str], None] | None = None
    if source.debug_drops:

        def _report_drop(category: str, reason: str, detail: str) -> None:
            message = f"internal_llm debug: {category} dropped reason={reason}"
            if detail:
                message += f" detail={detail!r}"
            print(message, file=sys.stderr)

        debug_callback = _report_drop

    rows, generated_counts = generate_internal_llm_rows(
        source, client, config.seed, _report_progress, debug_callback
    )
    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    kept_rows: list[SFTRow] = []
    filtered_rows = 0
    for row in rows:
        if _passes_context_filter(row, tokenizer, config.context_length):
            kept_rows.append(row)
        else:
            filtered_rows += 1

    requested_rows = source.assistant_rows + source.math_rows + source.tool_rows
    output_path = Path(config.output_dir) / f"{source.source}.jsonl"
    written_rows = _write_jsonl(output_path, kept_rows)
    manifest = SourceManifest(
        source=source.source,
        dataset_id=f"internal-llm:{source.model_env}",
        split="generated",
        seed=config.seed,
        requested_rows=requested_rows,
        candidate_rows=len(rows),
        written_rows=written_rows,
        filtered_rows=filtered_rows,
        output_path=str(output_path),
        sha256=_sha256_file(output_path),
        dropped_rows=max(0, requested_rows - len(rows)),
        model_env=source.model_env,
        prompt_version=source.prompt_version,
        generated_counts=generated_counts,
    )
    _update_manifest(config.output_dir, manifest)
    return manifest


def prepare_all(config: SFTDataConfig) -> dict[str, SourceManifest]:
    """Prepare the public SFT sources, local tool source, and optional internal LLM."""
    manifests = [
        prepare_assistant(config),
        prepare_gsm8k(config),
        prepare_public_tool(config),
        *prepare_tool(config).values(),
    ]
    internal_manifest = prepare_internal_llm(config)
    if internal_manifest is not None:
        manifests.append(internal_manifest)
    return {manifest.source: manifest for manifest in manifests}
