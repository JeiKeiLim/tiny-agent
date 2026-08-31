"""Normalizer for the public argilla/apigen-function-calling SFT source."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from kestrel.data.sft_chat import validate_arguments
from kestrel.data.sft_schema import FunctionSpec, SFTRow, ToolDefinition
from kestrel.data.sft_tool_generator import SYSTEM_PROMPT

_XLAM_LIST_TYPE = re.compile(r"list\[(\w+)\]")


def _parse_json_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _xlam_primitive_schema(type_name: str) -> dict[str, Any] | None:
    match type_name:
        case "str" | "string":
            return {"type": "string"}
        case "int" | "integer":
            return {"type": "integer"}
        case "float" | "number":
            return {"type": "number"}
        case "bool" | "boolean":
            return {"type": "boolean"}
        case _:
            return None


def _xlam_type_schema(raw_type: Any, max_list_items: int) -> dict[str, Any] | None:
    if not isinstance(raw_type, str):
        return None
    cleaned = raw_type.strip().lower()
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    list_match = _XLAM_LIST_TYPE.fullmatch(cleaned)
    if list_match is not None:
        item_schema = _xlam_primitive_schema(list_match.group(1))
        if item_schema is None:
            return None
        return {"type": "array", "items": item_schema, "maxItems": max_list_items}
    return _xlam_primitive_schema(cleaned)


def _with_description(schema: dict[str, Any], description: Any) -> dict[str, Any]:
    if isinstance(description, str) and description:
        return {**schema, "description": description}
    return schema


def _normalize_xlam_parameters(raw_parameters: Any, max_list_items: int) -> dict[str, Any] | None:
    if not isinstance(raw_parameters, dict):
        return None
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, raw_spec in raw_parameters.items():
        if not isinstance(raw_spec, dict):
            return None
        schema = _xlam_type_schema(raw_spec.get("type", "str"), max_list_items)
        if schema is None:
            return None
        properties[str(name)] = _with_description(schema, raw_spec.get("description"))
        if "default" not in raw_spec:
            required.append(str(name))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _normalize_json_property(property_schema: Any, max_list_items: int) -> dict[str, Any] | None:
    if not isinstance(property_schema, dict):
        return None
    property_type = property_schema.get("type")
    if isinstance(property_type, list):
        return None
    if property_type is None:
        if "enum" not in property_schema:
            return None
        property_type = "string"

    if property_type == "string":
        schema: dict[str, Any] = {"type": "string"}
        if "enum" in property_schema:
            enum_values = property_schema["enum"]
            if (
                not isinstance(enum_values, list)
                or not enum_values
                or not all(isinstance(value, str) for value in enum_values)
            ):
                return None
            schema["enum"] = enum_values
        return _with_description(schema, property_schema.get("description"))

    if property_type in {"integer", "number", "boolean"}:
        return _with_description({"type": property_type}, property_schema.get("description"))

    if property_type == "array":
        item_schema = _normalize_json_property(property_schema.get("items"), max_list_items)
        if item_schema is None or item_schema.get("type") in {"array", "object"}:
            return None
        raw_max_items = property_schema.get("maxItems", max_list_items)
        if (
            not isinstance(raw_max_items, int)
            or isinstance(raw_max_items, bool)
            or raw_max_items <= 0
        ):
            return None
        schema = {
            "type": "array",
            "items": item_schema,
            "maxItems": min(raw_max_items, max_list_items),
        }
        return _with_description(schema, property_schema.get("description"))

    return None


def _normalize_json_schema(parameters: Any, max_list_items: int) -> dict[str, Any] | None:
    if not isinstance(parameters, dict):
        return None
    if parameters.get("type") not in {None, "object"}:
        return None
    raw_properties = parameters.get("properties", {})
    if not isinstance(raw_properties, dict):
        return None
    raw_required = parameters.get("required", [])
    if not isinstance(raw_required, list):
        return None

    properties: dict[str, Any] = {}
    for name, property_schema in raw_properties.items():
        normalized = _normalize_json_property(property_schema, max_list_items)
        if normalized is None:
            return None
        properties[str(name)] = normalized

    required = [str(name) for name in raw_required if str(name) in properties]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _build_definition(
    name: str, description: str, parameters: dict[str, Any]
) -> ToolDefinition | None:
    try:
        return ToolDefinition(
            type="function",
            function=FunctionSpec(name=name, description=description, parameters=parameters),
        )
    except ValidationError:
        return None


def _parse_tool_definition(
    item: Any, *, max_tool_chars: int, max_list_items: int
) -> ToolDefinition | None:
    if not isinstance(item, dict):
        return None

    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        function = item["function"]
        name = function.get("name")
        description = function.get("description", "")
        if not isinstance(name, str) or not name or not isinstance(description, str):
            return None
        parameters = _normalize_json_schema(
            function.get("parameters", {"type": "object"}), max_list_items
        )
        if parameters is None:
            return None
        definition = _build_definition(name, description, parameters)
    elif isinstance(item.get("name"), str) and isinstance(item.get("parameters"), dict):
        name = item["name"]
        description = item.get("description", "")
        if not name or not isinstance(description, str):
            return None
        parameters = _normalize_xlam_parameters(item["parameters"], max_list_items)
        if parameters is None:
            return None
        definition = _build_definition(name, description, parameters)
    else:
        return None

    if definition is None:
        return None
    serialized = json.dumps(definition.model_dump(mode="json"), ensure_ascii=False)
    if len(serialized) > max_tool_chars:
        return None
    return definition


def _parse_tools(
    raw_tools: Any,
    *,
    max_tools: int,
    max_tool_chars: int,
    max_list_items: int,
    excluded_tool_names: frozenset[str],
) -> list[ToolDefinition] | None:
    tools = _parse_json_list(raw_tools)
    if not tools or len(tools) > max_tools:
        return None

    parsed: list[ToolDefinition] = []
    seen_names: set[str] = set()
    for item in tools:
        definition = _parse_tool_definition(
            item, max_tool_chars=max_tool_chars, max_list_items=max_list_items
        )
        if definition is None:
            return None
        name = definition.function.name
        if name in seen_names or name in excluded_tool_names:
            return None
        seen_names.add(name)
        parsed.append(definition)
    return parsed


def _arguments_are_flat(arguments: dict[str, Any]) -> bool:
    for value in arguments.values():
        if isinstance(value, dict):
            return False
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict | list) or not isinstance(item, str | int | float | bool):
                    return False
        elif not isinstance(value, str | int | float | bool):
            return False
    return True


def _parse_single_call(raw_answers: Any, tools: list[ToolDefinition]) -> dict[str, Any] | None:
    answers = _parse_json_list(raw_answers)
    if not answers or len(answers) != 1:
        return None
    call = answers[0]
    if not isinstance(call, dict):
        return None
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    if not _arguments_are_flat(arguments):
        return None
    tool = next((item for item in tools if item.function.name == name), None)
    if tool is None:
        return None
    if validate_arguments(tool.function.parameters, arguments):
        return None
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@dataclass
class PublicToolNormalizer:
    """Stateful row normalizer for public tool-calling data."""

    source: str
    max_tools: int = 5
    max_query_chars: int = 512
    max_tool_chars: int = 2048
    max_list_items: int = 10
    excluded_tool_names: frozenset[str] = frozenset()
    _seen_ids: set[int] = field(default_factory=set)
    _seen_hash_ids: set[str] = field(default_factory=set)
    _seen_queries: set[str] = field(default_factory=set)

    def convert(self, raw: dict[str, Any]) -> SFTRow | None:
        """Convert one raw public tool row, or return ``None`` if it should be dropped."""
        query = raw.get("query")
        if not isinstance(query, str):
            return None
        stripped_query = query.strip()
        normalized_query = _normalize_query(stripped_query)
        if not normalized_query or len(stripped_query) > self.max_query_chars:
            return None
        if normalized_query in self._seen_queries:
            return None

        row_id = raw.get("id")
        if isinstance(row_id, bool) or not isinstance(row_id, int):
            row_id = None
        if row_id is not None and row_id in self._seen_ids:
            return None

        hash_id = raw.get("hash_id")
        if not isinstance(hash_id, str) or not hash_id:
            hash_id = None
        if hash_id is not None and hash_id in self._seen_hash_ids:
            return None

        tools = _parse_tools(
            raw.get("tools"),
            max_tools=self.max_tools,
            max_tool_chars=self.max_tool_chars,
            max_list_items=self.max_list_items,
            excluded_tool_names=self.excluded_tool_names,
        )
        if tools is None:
            return None

        tool_call = _parse_single_call(raw.get("answers"), tools)
        if tool_call is None:
            return None

        try:
            row = SFTRow.model_validate(
                {
                    "source": self.source,
                    "tools": [tool.model_dump(mode="json") for tool in tools],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": stripped_query},
                        {"role": "assistant", "tool_calls": [tool_call]},
                    ],
                }
            )
        except ValidationError:
            return None

        if row_id is not None:
            self._seen_ids.add(row_id)
        if hash_id is not None:
            self._seen_hash_ids.add(hash_id)
        self._seen_queries.add(normalized_query)
        return row


def load_public_tool_rows(dataset_id: str, split: str) -> Iterator[dict[str, Any]]:
    """Load public tool rows from Hugging Face, using the local cache when available."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    return iter(dataset)
