"""Metrics for generated M2 tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kestrel.data.sft_chat import TOOL_CALL, extract_tool_call_payload, validate_arguments
from kestrel.data.sft_schema import ToolDefinition


@dataclass(frozen=True)
class GeneratedToolCall:
    """Parsed result of one generated assistant turn."""

    attempted: bool
    valid_json: bool
    name: str | None
    arguments: dict[str, Any] | None
    schema_valid: bool
    error: str | None = None


def parse_generated_tool_call(text: str, tools: list[ToolDefinition]) -> GeneratedToolCall:
    """Parse a generated assistant turn as a tool call without raising."""
    if TOOL_CALL not in text:
        return GeneratedToolCall(
            attempted=False,
            valid_json=False,
            name=None,
            arguments=None,
            schema_valid=False,
            error="no tool-call marker",
        )

    payload = extract_tool_call_payload(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return GeneratedToolCall(
            attempted=True,
            valid_json=False,
            name=None,
            arguments=None,
            schema_valid=False,
            error=f"invalid tool-call JSON: {exc}",
        )

    if not isinstance(data, dict):
        return GeneratedToolCall(
            attempted=True,
            valid_json=False,
            name=None,
            arguments=None,
            schema_valid=False,
            error="tool-call payload must be a JSON object",
        )

    name = data.get("name")
    arguments = data.get("arguments")
    if not isinstance(name, str) or not name:
        return GeneratedToolCall(
            attempted=True,
            valid_json=False,
            name=None,
            arguments=None,
            schema_valid=False,
            error="tool-call payload requires a non-empty string name",
        )
    if not isinstance(arguments, dict):
        return GeneratedToolCall(
            attempted=True,
            valid_json=False,
            name=name,
            arguments=None,
            schema_valid=False,
            error="tool-call payload requires an object arguments field",
        )

    tool = next((item for item in tools if item.function.name == name), None)
    if tool is None:
        return GeneratedToolCall(
            attempted=True,
            valid_json=True,
            name=name,
            arguments=arguments,
            schema_valid=False,
            error=f"unknown tool name: {name}",
        )

    errors = validate_arguments(tool.function.parameters, arguments)
    if errors:
        return GeneratedToolCall(
            attempted=True,
            valid_json=True,
            name=name,
            arguments=arguments,
            schema_valid=False,
            error="tool-call arguments failed schema validation: " + "; ".join(errors),
        )

    return GeneratedToolCall(
        attempted=True,
        valid_json=True,
        name=name,
        arguments=arguments,
        schema_valid=True,
        error=None,
    )


@dataclass(frozen=True)
class ToolSetMetrics:
    """Aggregate metrics for a direct tool-call eval set."""

    rows: int
    valid_json_rate: float
    schema_valid_rate: float
    tool_selection_rate: float
    argument_exact_rate: float
    argument_partial_accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rows": self.rows,
            "valid_json_rate": self.valid_json_rate,
            "schema_valid_rate": self.schema_valid_rate,
            "tool_selection_rate": self.tool_selection_rate,
            "argument_exact_rate": self.argument_exact_rate,
            "argument_partial_accuracy": self.argument_partial_accuracy,
        }


@dataclass(frozen=True)
class NoCallMetrics:
    """Aggregate metrics for no-call and missing-info eval sets."""

    rows: int
    no_tool_call_rate: float
    non_empty_rate: float
    correct_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rows": self.rows,
            "no_tool_call_rate": self.no_tool_call_rate,
            "non_empty_rate": self.non_empty_rate,
            "correct_rate": self.correct_rate,
        }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def tool_set_partial_accuracy(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Fraction of expected arguments whose values match exactly."""
    if not expected:
        return 1.0 if not actual else 0.0
    matched = sum(1 for key, value in expected.items() if actual.get(key) == value)
    return matched / len(expected)


def make_tool_set_metrics(
    rows: int,
    valid_json: int,
    schema_valid: int,
    tool_selection: int,
    argument_exact: int,
    argument_partial_sum: float,
) -> ToolSetMetrics:
    return ToolSetMetrics(
        rows=rows,
        valid_json_rate=_rate(valid_json, rows),
        schema_valid_rate=_rate(schema_valid, rows),
        tool_selection_rate=_rate(tool_selection, rows),
        argument_exact_rate=_rate(argument_exact, rows),
        argument_partial_accuracy=argument_partial_sum / rows if rows else 0.0,
    )


def make_no_call_metrics(
    rows: int, no_tool_call: int, non_empty: int, correct: int
) -> NoCallMetrics:
    return NoCallMetrics(
        rows=rows,
        no_tool_call_rate=_rate(no_tool_call, rows),
        non_empty_rate=_rate(non_empty, rows),
        correct_rate=_rate(correct, rows),
    )
