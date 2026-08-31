"""Chat renderer and tool-call parser for Kestrel M2 SFT rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

from kestrel.data.sft_schema import SFTRow, ToolDefinition
from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS

IM_START = DEFAULT_SPECIAL_TOKENS[0]
IM_END = DEFAULT_SPECIAL_TOKENS[1]
IM_SYSTEM = DEFAULT_SPECIAL_TOKENS[2]
IM_USER = DEFAULT_SPECIAL_TOKENS[3]
IM_ASSISTANT = DEFAULT_SPECIAL_TOKENS[4]
TOOL_CALL = DEFAULT_SPECIAL_TOKENS[5]
TOOL_CALL_END = DEFAULT_SPECIAL_TOKENS[6]
TOOL_RESPONSE = DEFAULT_SPECIAL_TOKENS[7]
TOOL_RESPONSE_END = DEFAULT_SPECIAL_TOKENS[8]


@dataclass(frozen=True)
class RenderedSFT:
    """Tokenized SFT sequence with a per-token loss mask."""

    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    text: str


class ToolCallParseError(ValueError):
    """Raised when a rendered assistant tool call is invalid."""


@dataclass(frozen=True)
class ParsedToolCall:
    """Parsed and validated assistant tool call."""

    name: str
    arguments: dict[str, Any]


def _require_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        msg = f"tokenizer is missing required special token: {token}"
        raise ValueError(msg)
    return token_id


def render_sft(row: SFTRow, tokenizer: Tokenizer) -> RenderedSFT:
    """Render a logical SFT row into token IDs, loss mask, and text.

    Assistant content and assistant tool-call payloads are marked with loss
    mask ``1``. System, user, tool, role markers, and structural markers are
    masked with ``0``.
    """
    ids: list[int] = []
    mask: list[int] = []
    text_parts: list[str] = []

    def add_special(token: str, loss: int) -> None:
        ids.append(_require_token_id(tokenizer, token))
        mask.append(loss)
        text_parts.append(token)

    def add_newline(loss: int = 0) -> None:
        encoded = tokenizer.encode("\n", add_special_tokens=False).ids
        ids.extend(encoded)
        mask.extend([loss] * len(encoded))
        text_parts.append("\n")

    def add_text(text: str, loss: int) -> None:
        encoded = tokenizer.encode(text, add_special_tokens=False).ids
        ids.extend(encoded)
        mask.extend([loss] * len(encoded))
        text_parts.append(text)

    for message in row.messages:
        add_special(IM_START, 0)
        add_newline()

        if message.role == "system":
            add_special(IM_SYSTEM, 0)
            add_newline()
            add_text(message.content, 0)
            add_newline()
            add_special(IM_END, 0)
            add_newline()
        elif message.role == "user":
            add_special(IM_USER, 0)
            add_newline()
            add_text(message.content, 0)
            add_newline()
            add_special(IM_END, 0)
            add_newline()
        elif message.role == "tool":
            add_special(TOOL_RESPONSE, 0)
            add_newline()
            add_text(message.content, 0)
            add_newline()
            add_special(TOOL_RESPONSE_END, 0)
            add_newline()
            add_special(IM_END, 0)
            add_newline()
        else:
            add_special(IM_ASSISTANT, 0)
            add_newline()
            if message.content is not None:
                add_text(message.content, 1)
                add_newline()
                add_special(IM_END, 1)
                add_newline()
            else:
                tool_calls = message.tool_calls
                if tool_calls is None:
                    msg = "assistant message has neither content nor tool_calls"
                    raise ValueError(msg)
                call = tool_calls[0]
                payload = json.dumps(
                    {"name": call.function.name, "arguments": call.function.arguments},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                add_special(TOOL_CALL, 1)
                add_newline()
                add_text(payload, 1)
                add_newline()
                add_special(TOOL_CALL_END, 1)
                add_newline()
                add_special(IM_END, 1)
                add_newline()

    return RenderedSFT(tuple(ids), tuple(mask), "".join(text_parts))


def _extract_payload(text: str) -> str:
    start = text.find(TOOL_CALL)
    end = text.find(TOOL_CALL_END)
    if start != -1 and end != -1 and end > start:
        return text[start + len(TOOL_CALL) : end].strip()
    return text.strip()


def _type_matches(type_name: str, value: Any) -> bool:
    match type_name:
        case "string":
            return isinstance(value, str)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "null":
            return value is None
        case _:
            return True


def _validate_schema(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    if not isinstance(schema, dict):
        return []

    errors: list[str] = []

    if "enum" in schema:
        enum_values = schema["enum"]
        if isinstance(enum_values, list) and value not in enum_values:
            errors.append(f"{path} must be one of {enum_values!r}")

    declared_type = schema.get("type")
    if declared_type is not None:
        type_names = declared_type if isinstance(declared_type, list) else [declared_type]
        if not any(_type_matches(str(name), value) for name in type_names):
            errors.append(f"{path} must be of type {declared_type!r}")
            return errors

    if isinstance(value, dict):
        raw_properties = schema.get("properties", {})
        properties = raw_properties if isinstance(raw_properties, dict) else {}
        raw_required = schema.get("required", [])
        required = raw_required if isinstance(raw_required, list) else []

        for raw_key in required:
            key = str(raw_key)
            if key not in value:
                errors.append(f"{path} missing required argument {key!r}")

        for key, item in value.items():
            if key in properties:
                errors.extend(_validate_schema(properties[key], item, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path} has additional property {key!r}")

    if isinstance(value, list):
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path} must have at most {max_items} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_schema(items_schema, item, f"{path}[{index}]"))

    return errors


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Validate tool-call arguments against a JSON Schema object."""
    return _validate_schema(schema, arguments, "$")


def parse_tool_call(text: str, tools: list[ToolDefinition]) -> ParsedToolCall:
    """Parse and validate a rendered assistant tool call.

    Raises:
        ToolCallParseError: if the payload is missing, invalid JSON, uses an
            unknown tool name, or fails schema validation.
    """
    payload = _extract_payload(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        msg = f"invalid tool-call JSON: {exc}"
        raise ToolCallParseError(msg) from exc

    if not isinstance(data, dict):
        msg = "tool-call payload must be a JSON object"
        raise ToolCallParseError(msg)

    name = data.get("name")
    arguments = data.get("arguments")
    if not isinstance(name, str) or not name:
        msg = "tool-call payload requires a non-empty string name"
        raise ToolCallParseError(msg)
    if not isinstance(arguments, dict):
        msg = "tool-call payload requires an object arguments field"
        raise ToolCallParseError(msg)

    tool = next((item for item in tools if item.function.name == name), None)
    if tool is None:
        msg = f"unknown tool name: {name}"
        raise ToolCallParseError(msg)

    errors = validate_arguments(tool.function.parameters, arguments)
    if errors:
        msg = "tool-call arguments failed schema validation: " + "; ".join(errors)
        raise ToolCallParseError(msg)

    return ParsedToolCall(name=name, arguments=arguments)
