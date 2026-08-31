"""Tests for the strict M2 SFT row schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kestrel.data.sft_schema import SFTRow


def _tool_row() -> dict[str, object]:
    return {
        "source": "tool_local",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "What is the weather in Seoul?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Seoul"}},
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
    }


def test_valid_non_tool_row() -> None:
    row = SFTRow.model_validate(
        {
            "source": "assistant_public",
            "messages": [
                {"role": "system", "content": "Helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
    )
    assert row.source == "assistant_public"
    assert row.tools == []


def test_valid_tool_row() -> None:
    row = SFTRow.model_validate(_tool_row())
    assert len(row.messages) == 4


def test_unknown_row_key_raises() -> None:
    data = _tool_row()
    data["unexpected"] = 1
    with pytest.raises(ValidationError, match="unexpected"):
        SFTRow.model_validate(data)


def test_unknown_message_key_raises() -> None:
    data = _tool_row()
    messages = data["messages"]
    assert isinstance(messages, list)
    messages[0] = {"role": "user", "content": "Hello", "extra": True}
    with pytest.raises(ValidationError, match="extra"):
        SFTRow.model_validate(data)


def test_strict_type_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        SFTRow.model_validate({"source": 1, "messages": [{"role": "user", "content": "hi"}]})


def test_assistant_requires_content_or_tool_calls() -> None:
    with pytest.raises(ValidationError, match="content or tool_calls"):
        SFTRow.model_validate(
            {
                "source": "assistant_public",
                "messages": [{"role": "user", "content": "hi"}, {"role": "assistant"}],
            }
        )


def test_assistant_cannot_have_content_and_tool_calls() -> None:
    with pytest.raises(ValidationError, match="both content and tool_calls"):
        SFTRow.model_validate(
            {
                "source": "tool_local",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "final",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": {}},
                            }
                        ],
                    },
                ],
            }
        )


def test_multiple_tool_calls_raise() -> None:
    with pytest.raises(ValidationError, match="exactly one tool call"):
        SFTRow.model_validate(
            {
                "source": "tool_local",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": {}},
                            },
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": {}},
                            },
                        ],
                    },
                ],
            }
        )


def test_tool_message_requires_matching_assistant_call() -> None:
    with pytest.raises(ValidationError, match="tool message must follow"):
        SFTRow.model_validate(
            {
                "source": "tool_local",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "name": "get_weather", "content": "sunny"},
                ],
            }
        )


def test_unknown_tool_call_name_raises() -> None:
    with pytest.raises(ValidationError, match="not defined in tools"):
        SFTRow.model_validate(
            {
                "source": "tool_local",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_forecast", "arguments": {}},
                            }
                        ],
                    },
                ],
            }
        )


def test_first_message_must_be_system_or_user() -> None:
    with pytest.raises(ValidationError, match="first message"):
        SFTRow.model_validate(
            {
                "source": "assistant_public",
                "messages": [{"role": "assistant", "content": "hi"}],
            }
        )
