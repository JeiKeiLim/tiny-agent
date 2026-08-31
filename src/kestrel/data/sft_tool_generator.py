"""Seeded local rule-based tool SFT generator for M2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from random import Random
from typing import Any, Self

from pydantic import Field, model_validator

from kestrel.common.config import BaseConfig
from kestrel.data.sft_chat import validate_arguments
from kestrel.data.sft_schema import SFTRow, ToolDefinition
from kestrel.tools.schema_sampler import (
    TRAIN_TOOL_FAMILIES,
    UNSEEN_TOOL_FAMILIES,
    SampledTool,
    ToolFamilySpec,
    ToolTask,
)

SYSTEM_PROMPT = (
    "You are Kestrel, a helpful assistant. Use the provided tools when they are necessary. "
    "Call exactly one tool when a tool is needed. If required information is missing, ask for "
    "clarification. If no tool is needed, answer directly."
)

NO_TOOL_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("What is the capital of France?", "The capital of France is Paris."),
    (
        "Please give a one-sentence definition of JSON.",
        "JSON is a text format for structured data using key-value pairs, lists, and primitives.",
    ),
    ("Say hello in a friendly way.", "Hello! How can I help you today?"),
    (
        "What color do you get when mixing blue and yellow?",
        "Mixing blue and yellow gives green.",
    ),
    ("Please remind me to drink water.", "Okay. Remember to drink water regularly."),
)


class ToolTrainBreakdown(BaseConfig):
    """Locked 10k local tool train breakdown."""

    direct: int = Field(default=6_000, ge=0)
    no_tool: int = Field(default=1_500, ge=0)
    distractor_heavy: int = Field(default=1_000, ge=0)
    missing_info: int = Field(default=750, ge=0)
    hard_variation: int = Field(default=750, ge=0)


class ToolEvalBreakdown(BaseConfig):
    """Local tool eval set sizes."""

    seen: int = Field(default=500, ge=0)
    unseen: int = Field(default=500, ge=0)
    no_call: int = Field(default=250, ge=0)
    missing_info: int = Field(default=250, ge=0)


class ToolGeneratorConfig(BaseConfig):
    """Strict settings for the local rule-based tool SFT generator."""

    seed: int = 42
    min_tools: int = Field(default=3, ge=2, le=5)
    max_tools: int = Field(default=5, ge=3, le=5)
    train: ToolTrainBreakdown = Field(default_factory=ToolTrainBreakdown)
    eval: ToolEvalBreakdown = Field(default_factory=ToolEvalBreakdown)

    @model_validator(mode="after")
    def _check_tool_range(self) -> Self:
        if self.min_tools > self.max_tools:
            msg = "min_tools must be less than or equal to max_tools"
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class ToolTrainRows:
    """Categorized local tool train rows."""

    direct: tuple[SFTRow, ...]
    no_tool: tuple[SFTRow, ...]
    distractor_heavy: tuple[SFTRow, ...]
    missing_info: tuple[SFTRow, ...]
    hard_variation: tuple[SFTRow, ...]

    @property
    def all_rows(self) -> list[SFTRow]:
        return [
            *self.direct,
            *self.no_tool,
            *self.distractor_heavy,
            *self.missing_info,
            *self.hard_variation,
        ]


@dataclass(frozen=True)
class ToolEvalRows:
    """Categorized local tool eval rows."""

    seen: tuple[SFTRow, ...]
    unseen: tuple[SFTRow, ...]
    no_call: tuple[SFTRow, ...]
    missing_info: tuple[SFTRow, ...]

    @property
    def all_rows(self) -> list[SFTRow]:
        return [*self.seen, *self.unseen, *self.no_call, *self.missing_info]


def _make_row(source: str, tools: list[ToolDefinition], messages: list[dict[str, Any]]) -> SFTRow:
    payload: dict[str, Any] = {
        "source": source,
        "tools": [tool.model_dump(mode="json") for tool in tools],
        "messages": messages,
    }
    return SFTRow.model_validate(payload)


def _validate_tool_call(definition: ToolDefinition, arguments: dict[str, Any]) -> None:
    errors = validate_arguments(definition.function.parameters, arguments)
    if errors:
        msg = f"generated arguments failed validation: {'; '.join(errors)}"
        raise ValueError(msg)


def _tool_call_messages(definition: ToolDefinition, task: ToolTask) -> list[dict[str, Any]]:
    name = definition.function.name
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": task.arguments},
                }
            ],
        },
        {"role": "tool", "name": name, "content": task.tool_result},
        {"role": "assistant", "content": task.final_answer},
    ]


def _build_tools(
    rng: Random,
    families: tuple[ToolFamilySpec, ...],
    relevant_family: ToolFamilySpec,
    relevant_definition: ToolDefinition,
    tool_count: int,
) -> list[ToolDefinition]:
    distractors = [
        family.sample_definition(rng)
        for family in rng.sample(
            [family for family in families if family.family != relevant_family.family],
            k=tool_count - 1,
        )
    ]
    tools = [relevant_definition, *distractors]
    rng.shuffle(tools)
    return tools


def _direct_row(
    source: str,
    rng: Random,
    config: ToolGeneratorConfig,
    families: tuple[ToolFamilySpec, ...],
    prompt_prefix: str = "",
    force_max_tools: bool = False,
) -> SFTRow:
    family = rng.choice(families)
    sampled: SampledTool = family.sample(rng)
    if force_max_tools:
        tool_count = config.max_tools
    else:
        tool_count = rng.randint(config.min_tools, config.max_tools)
    tools = _build_tools(rng, families, family, sampled.definition, tool_count)
    user_prompt = sampled.task.user_prompt
    if prompt_prefix:
        user_prompt = f"{prompt_prefix} {user_prompt}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        *_tool_call_messages(sampled.definition, sampled.task),
    ]
    _validate_tool_call(sampled.definition, sampled.task.arguments)
    return _make_row(source, tools, messages)


def _generate_direct(
    source: str,
    rng: Random,
    config: ToolGeneratorConfig,
    count: int,
    families: tuple[ToolFamilySpec, ...],
    prompt_prefix: str = "",
    force_max_tools: bool = False,
) -> list[SFTRow]:
    return [
        _direct_row(source, rng, config, families, prompt_prefix, force_max_tools)
        for _ in range(count)
    ]


def _no_tool_row(
    source: str,
    rng: Random,
    config: ToolGeneratorConfig,
    families: tuple[ToolFamilySpec, ...],
) -> SFTRow:
    tool_count = rng.randint(config.min_tools, config.max_tools)
    tools = [family.sample_definition(rng) for family in rng.sample(list(families), k=tool_count)]
    rng.shuffle(tools)
    user_prompt, answer = rng.choice(NO_TOOL_EXAMPLES)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]
    return _make_row(source, tools, messages)


def _generate_no_tool(
    source: str,
    rng: Random,
    config: ToolGeneratorConfig,
    count: int,
    families: tuple[ToolFamilySpec, ...],
) -> list[SFTRow]:
    return [_no_tool_row(source, rng, config, families) for _ in range(count)]


def _missing_info_row(
    source: str,
    rng: Random,
    config: ToolGeneratorConfig,
    families: tuple[ToolFamilySpec, ...],
) -> SFTRow:
    family = rng.choice(families)
    definition = family.sample_definition(rng)
    tool_count = rng.randint(config.min_tools, config.max_tools)
    tools = _build_tools(rng, families, family, definition, tool_count)
    user_prompt, answer = rng.choice(family.missing_info)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]
    return _make_row(source, tools, messages)


def _generate_missing_info(
    source: str,
    rng: Random,
    config: ToolGeneratorConfig,
    count: int,
    families: tuple[ToolFamilySpec, ...],
) -> list[SFTRow]:
    return [_missing_info_row(source, rng, config, families) for _ in range(count)]


def generate_tool_train(config: ToolGeneratorConfig) -> ToolTrainRows:
    """Generate the local tool train split using a seeded RNG."""
    rng = Random(config.seed)
    breakdown = config.train
    return ToolTrainRows(
        direct=tuple(
            _generate_direct("tool_local", rng, config, breakdown.direct, TRAIN_TOOL_FAMILIES)
        ),
        no_tool=tuple(
            _generate_no_tool("tool_local", rng, config, breakdown.no_tool, TRAIN_TOOL_FAMILIES)
        ),
        distractor_heavy=tuple(
            _generate_direct(
                "tool_local",
                rng,
                config,
                breakdown.distractor_heavy,
                TRAIN_TOOL_FAMILIES,
                prompt_prefix="Many tools are available.",
                force_max_tools=True,
            )
        ),
        missing_info=tuple(
            _generate_missing_info(
                "tool_local", rng, config, breakdown.missing_info, TRAIN_TOOL_FAMILIES
            )
        ),
        hard_variation=tuple(
            _generate_direct(
                "tool_local",
                rng,
                config,
                breakdown.hard_variation,
                TRAIN_TOOL_FAMILIES,
                prompt_prefix="Some available tools may look relevant.",
                force_max_tools=True,
            )
        ),
    )


def generate_tool_eval(config: ToolGeneratorConfig) -> ToolEvalRows:
    """Generate local tool eval splits using a seeded RNG offset from train."""
    rng = Random(config.seed + 1)
    breakdown = config.eval
    return ToolEvalRows(
        seen=tuple(
            _generate_direct("tool_eval_seen", rng, config, breakdown.seen, TRAIN_TOOL_FAMILIES)
        ),
        unseen=tuple(
            _generate_direct(
                "tool_eval_unseen", rng, config, breakdown.unseen, UNSEEN_TOOL_FAMILIES
            )
        ),
        no_call=tuple(
            _generate_no_tool(
                "tool_eval_no_call", rng, config, breakdown.no_call, TRAIN_TOOL_FAMILIES
            )
        ),
        missing_info=tuple(
            _generate_missing_info(
                "tool_eval_missing_info", rng, config, breakdown.missing_info, TRAIN_TOOL_FAMILIES
            )
        ),
    )


def row_tool_names(rows: Iterable[SFTRow]) -> set[str]:
    """Collect every tool name referenced by a set of SFT rows."""
    names: set[str] = set()
    for row in rows:
        names.update(tool.function.name for tool in row.tools)
    return names
