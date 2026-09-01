"""Inference-only SFT evaluation and pretrain-vs-SFT scorecard."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field
from tokenizers import Tokenizer

from kestrel.common.config import BaseConfig, load_config
from kestrel.data.chat import extract_assistant_content
from kestrel.data.sft_chat import TOOL_CALL, completion_prompt_text
from kestrel.data.sft_schema import AssistantMessage, SFTRow
from kestrel.eval.pretrain import evaluate_checkpoint
from kestrel.eval.tool_calling import (
    GeneratedToolCall,
    NoCallMetrics,
    ToolSetMetrics,
    make_no_call_metrics,
    make_tool_set_metrics,
    parse_generated_tool_call,
    tool_set_partial_accuracy,
)
from kestrel.model.config import ModelConfig
from kestrel.model.generate import generate
from kestrel.model.io import load as load_model
from kestrel.train.pretrain import PretrainConfig


class SFTCheckpointEntry(BaseConfig):
    """One named checkpoint to include in the SFT scorecard."""

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)


class SFTGenerationConfig(BaseConfig):
    """Decoding settings for SFT eval generation."""

    max_tokens: int = Field(default=256, gt=0)
    temperature: float = 0.0
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    progress_every: int = Field(default=100, ge=0)
    clear_cache_every: int = Field(default=64, ge=0)


class SFTEvalDataConfig(BaseConfig):
    """Held-out SFT eval bundle locations."""

    dir: str = "data/sft/eval"
    assistant_file: str = "assistant_eval.jsonl"
    gsm8k_file: str = "gsm8k_eval.jsonl"
    tool_seen_file: str = "tool_eval_seen.jsonl"
    tool_unseen_file: str = "tool_eval_unseen.jsonl"
    tool_no_call_file: str = "tool_eval_no_call.jsonl"
    tool_missing_info_file: str = "tool_eval_missing_info.jsonl"
    max_rows_per_set: int | None = Field(default=None, gt=0)


class SFTPerplexityConfig(BaseConfig):
    """Optional held-out pretrain perplexity measurement."""

    enabled: bool = True
    pretrain_config: str = "configs/kestrel/50m/pretrain.yaml"
    split: str = "val"
    max_tokens: int = Field(default=100_000, gt=0)


class SFTEvalConfig(BaseConfig):
    """Strict settings for the SFT eval harness."""

    model: str
    tokenizer: str
    checkpoints: list[SFTCheckpointEntry] = Field(min_length=1)
    data: SFTEvalDataConfig = Field(default_factory=SFTEvalDataConfig)
    generation: SFTGenerationConfig = Field(default_factory=SFTGenerationConfig)
    perplexity: SFTPerplexityConfig = Field(default_factory=SFTPerplexityConfig)
    output: str = "data/sft/eval/scorecard.json"
    on_missing_checkpoint: Literal["skip", "error"] = "skip"


@dataclass(frozen=True)
class AssistantMetrics:
    rows: int
    non_empty_rate: float
    no_tool_call_rate: float
    no_repetition_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rows": self.rows,
            "non_empty_rate": self.non_empty_rate,
            "no_tool_call_rate": self.no_tool_call_rate,
            "no_repetition_rate": self.no_repetition_rate,
        }


@dataclass(frozen=True)
class MathMetrics:
    rows: int
    exact_match_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {"rows": self.rows, "exact_match_rate": self.exact_match_rate}


@dataclass(frozen=True)
class ToolMetrics:
    seen: ToolSetMetrics
    unseen: ToolSetMetrics
    no_call: NoCallMetrics
    missing_info: NoCallMetrics

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            "seen": self.seen.to_dict(),
            "unseen": self.unseen.to_dict(),
            "no_call": self.no_call.to_dict(),
            "missing_info": self.missing_info.to_dict(),
        }


@dataclass(frozen=True)
class PerplexityMetrics:
    loss: float
    perplexity: float
    bits_per_token: float
    tokens: int
    batches: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
            "tokens": self.tokens,
            "batches": self.batches,
        }


@dataclass(frozen=True)
class CheckpointEvalResult:
    name: str
    path: str
    status: Literal["ok", "missing"]
    assistant: AssistantMetrics | None
    math: MathMetrics | None
    tool: ToolMetrics | None
    perplexity: PerplexityMetrics | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "assistant": self.assistant.to_dict() if self.assistant is not None else None,
            "math": self.math.to_dict() if self.math is not None else None,
            "tool": self.tool.to_dict() if self.tool is not None else None,
            "perplexity": self.perplexity.to_dict() if self.perplexity is not None else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class SFTEvalScorecard:
    format_version: int
    model: str
    tokenizer: str
    data_dir: str
    checkpoints: tuple[CheckpointEvalResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "data_dir": self.data_dir,
            "checkpoints": [result.to_dict() for result in self.checkpoints],
        }


@dataclass(frozen=True)
class EvalCase:
    category: str
    row: SFTRow
    target_index: int
    expected_tool_name: str | None = None
    expected_arguments: dict[str, Any] | None = None
    expected_content: str | None = None


@dataclass(frozen=True)
class GeneratedOutput:
    raw: str
    attempted_tool_call: bool
    content: str | None
    tool_call: GeneratedToolCall | None


def _load_rows(path: Path, limit: int | None) -> list[SFTRow]:
    if not path.is_file():
        msg = f"SFT eval file not found: {path}"
        raise FileNotFoundError(msg)

    rows: list[SFTRow] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(SFTRow.model_validate(json.loads(stripped)))
            except json.JSONDecodeError as exc:
                msg = f"{path} line {line_number}: invalid JSON: {exc}"
                raise ValueError(msg) from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _content_cases(rows: list[SFTRow], category: str, *, last: bool) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for row in rows:
        candidates = [
            (index, message)
            for index, message in enumerate(row.messages)
            if isinstance(message, AssistantMessage) and message.content is not None
        ]
        if not candidates:
            continue
        index, message = candidates[-1] if last else candidates[0]
        cases.append(
            EvalCase(
                category=category,
                row=row,
                target_index=index,
                expected_content=cast(str, message.content),
            )
        )
    return cases


def _tool_call_cases(rows: list[SFTRow], category: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for row in rows:
        for index, message in enumerate(row.messages):
            if not isinstance(message, AssistantMessage) or message.tool_calls is None:
                continue
            call = message.tool_calls[0]
            cases.append(
                EvalCase(
                    category=category,
                    row=row,
                    target_index=index,
                    expected_tool_name=call.function.name,
                    expected_arguments=dict(call.function.arguments),
                )
            )
            break
    return cases


def _load_cases(config: SFTEvalConfig) -> dict[str, list[EvalCase]]:
    data_dir = Path(config.data.dir)
    limit = config.data.max_rows_per_set
    return {
        "assistant": _content_cases(
            _load_rows(data_dir / config.data.assistant_file, limit), "assistant", last=True
        ),
        "math": _content_cases(
            _load_rows(data_dir / config.data.gsm8k_file, limit), "math", last=True
        ),
        "tool_seen": _tool_call_cases(
            _load_rows(data_dir / config.data.tool_seen_file, limit), "tool_seen"
        ),
        "tool_unseen": _tool_call_cases(
            _load_rows(data_dir / config.data.tool_unseen_file, limit), "tool_unseen"
        ),
        "tool_no_call": _content_cases(
            _load_rows(data_dir / config.data.tool_no_call_file, limit),
            "tool_no_call",
            last=False,
        ),
        "tool_missing_info": _content_cases(
            _load_rows(data_dir / config.data.tool_missing_info_file, limit),
            "tool_missing_info",
            last=False,
        ),
    }


def _prompt_text(case: EvalCase, tokenizer: Tokenizer) -> str:
    prompt_row = SFTRow.model_validate(
        {
            "source": case.row.source,
            "tools": [tool.model_dump(mode="json") for tool in case.row.tools],
            "messages": [
                message.model_dump(mode="json")
                for message in case.row.messages[: case.target_index]
            ],
        }
    )
    return completion_prompt_text(prompt_row, tokenizer)


def _generate_output(
    model: Any,
    tokenizer: Tokenizer,
    case: EvalCase,
    config: SFTGenerationConfig,
) -> GeneratedOutput:
    prompt = _prompt_text(case, tokenizer)
    raw = generate(
        model,
        tokenizer,
        prompt,
        config.max_tokens,
        temp=config.temperature,
        repetition_penalty=config.repetition_penalty,
        clear_cache_every=config.clear_cache_every,
    )
    if TOOL_CALL in raw:
        return GeneratedOutput(
            raw=raw,
            attempted_tool_call=True,
            content=None,
            tool_call=parse_generated_tool_call(raw, case.row.tools),
        )
    return GeneratedOutput(
        raw=raw,
        attempted_tool_call=False,
        content=extract_assistant_content(raw),
        tool_call=None,
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _has_obvious_repetition(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 12:
        return False
    if len(set(words)) / len(words) < 0.2:
        return True
    shingles = [tuple(words[index : index + 5]) for index in range(len(words) - 4)]
    return len(set(shingles)) * 2 < len(shingles)


def _assistant_metrics(outputs: list[GeneratedOutput]) -> AssistantMetrics:
    total = len(outputs)
    non_empty = sum(1 for output in outputs if (output.content or "").strip())
    no_tool_call = sum(1 for output in outputs if not output.attempted_tool_call)
    no_repetition = sum(
        1 for output in outputs if not _has_obvious_repetition(output.content or "")
    )
    return AssistantMetrics(
        rows=total,
        non_empty_rate=_rate(non_empty, total),
        no_tool_call_rate=_rate(no_tool_call, total),
        no_repetition_rate=_rate(no_repetition, total),
    )


def _extract_final_number(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"final answer\s*[:\-]?\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    candidate = match.group(1) if match is not None else text
    candidate = candidate.replace(",", "")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", candidate)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _math_metrics(cases: list[EvalCase], outputs: list[GeneratedOutput]) -> MathMetrics:
    total = 0
    correct = 0
    for case, output in zip(cases, outputs, strict=True):
        expected = _extract_final_number(case.expected_content or "")
        if expected is None:
            continue
        total += 1
        actual = _extract_final_number(output.content or "")
        if actual is not None and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
            correct += 1
    return MathMetrics(rows=total, exact_match_rate=_rate(correct, total))


def _tool_set_metrics(cases: list[EvalCase], outputs: list[GeneratedOutput]) -> ToolSetMetrics:
    total = len(cases)
    valid_json = 0
    schema_valid = 0
    tool_selection = 0
    argument_exact = 0
    argument_partial_sum = 0.0
    for case, output in zip(cases, outputs, strict=True):
        call = output.tool_call or GeneratedToolCall(
            attempted=False,
            valid_json=False,
            name=None,
            arguments=None,
            schema_valid=False,
            error="no tool-call marker",
        )
        if call.valid_json:
            valid_json += 1
        if call.schema_valid:
            schema_valid += 1
        if call.valid_json and call.name == case.expected_tool_name:
            tool_selection += 1
            actual = call.arguments or {}
            expected = case.expected_arguments or {}
            if actual == expected:
                argument_exact += 1
            argument_partial_sum += tool_set_partial_accuracy(expected, actual)
    return make_tool_set_metrics(
        rows=total,
        valid_json=valid_json,
        schema_valid=schema_valid,
        tool_selection=tool_selection,
        argument_exact=argument_exact,
        argument_partial_sum=argument_partial_sum,
    )


def _no_call_metrics(outputs: list[GeneratedOutput]) -> NoCallMetrics:
    total = len(outputs)
    no_tool_call = sum(1 for output in outputs if not output.attempted_tool_call)
    non_empty = sum(1 for output in outputs if (output.content or "").strip())
    correct = sum(
        1 for output in outputs if not output.attempted_tool_call and (output.content or "").strip()
    )
    return make_no_call_metrics(
        rows=total,
        no_tool_call=no_tool_call,
        non_empty=non_empty,
        correct=correct,
    )


def _perplexity_metrics(config: SFTEvalConfig, checkpoint: Path) -> PerplexityMetrics | None:
    if not config.perplexity.enabled:
        return None
    pretrain_config = load_config(config.perplexity.pretrain_config, PretrainConfig)
    result = evaluate_checkpoint(
        pretrain_config=pretrain_config,
        checkpoint=checkpoint,
        split=config.perplexity.split,
        max_tokens=config.perplexity.max_tokens,
        progress_every_tokens=0,
    )
    return PerplexityMetrics(
        loss=result.mixed.loss,
        perplexity=result.mixed.perplexity,
        bits_per_token=result.mixed.bits_per_token,
        tokens=result.mixed.tokens,
        batches=result.mixed.batches,
    )


def _evaluate_checkpoint(
    entry: SFTCheckpointEntry,
    *,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    config: SFTEvalConfig,
    cases: dict[str, list[EvalCase]],
) -> CheckpointEvalResult:
    checkpoint = Path(entry.path)
    if not (checkpoint / "weights.npz").is_file():
        message = f"checkpoint weights not found: {checkpoint / 'weights.npz'}"
        if config.on_missing_checkpoint == "error":
            raise FileNotFoundError(message)
        return CheckpointEvalResult(
            name=entry.name,
            path=entry.path,
            status="missing",
            assistant=None,
            math=None,
            tool=None,
            perplexity=None,
            error=message,
        )

    model = load_model(model_config, checkpoint)
    outputs_by_category: dict[str, list[GeneratedOutput]] = {}
    for category, category_cases in cases.items():
        outputs: list[GeneratedOutput] = []
        for index, case in enumerate(category_cases):
            outputs.append(_generate_output(model, tokenizer, case, config.generation))
            if (
                config.generation.progress_every > 0
                and (index + 1) % config.generation.progress_every == 0
            ):
                print(
                    f"[sft-eval] {entry.name} {category}: {index + 1}/{len(category_cases)}",
                    file=sys.stderr,
                    flush=True,
                )
        outputs_by_category[category] = outputs

    return CheckpointEvalResult(
        name=entry.name,
        path=entry.path,
        status="ok",
        assistant=_assistant_metrics(outputs_by_category["assistant"]),
        math=_math_metrics(cases["math"], outputs_by_category["math"]),
        tool=ToolMetrics(
            seen=_tool_set_metrics(cases["tool_seen"], outputs_by_category["tool_seen"]),
            unseen=_tool_set_metrics(cases["tool_unseen"], outputs_by_category["tool_unseen"]),
            no_call=_no_call_metrics(outputs_by_category["tool_no_call"]),
            missing_info=_no_call_metrics(outputs_by_category["tool_missing_info"]),
        ),
        perplexity=_perplexity_metrics(config, checkpoint),
        error=None,
    )


def evaluate_sft(config: SFTEvalConfig) -> SFTEvalScorecard:
    """Evaluate every configured checkpoint and return a JSON-serializable scorecard."""
    model_config = load_config(config.model, ModelConfig)
    tokenizer = Tokenizer.from_file(config.tokenizer)
    cases = _load_cases(config)

    results = tuple(
        _evaluate_checkpoint(
            entry,
            model_config=model_config,
            tokenizer=tokenizer,
            config=config,
            cases=cases,
        )
        for entry in config.checkpoints
    )
    return SFTEvalScorecard(
        format_version=1,
        model=config.model,
        tokenizer=config.tokenizer,
        data_dir=config.data.dir,
        checkpoints=results,
    )


def write_scorecard(scorecard: SFTEvalScorecard, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output_path
