"""Tests for the M2 SFT chat renderer and tool-call parser."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from kestrel.data.sft_chat import (
    IM_ASSISTANT,
    IM_END,
    IM_START,
    IM_SYSTEM,
    IM_USER,
    TOOL_CALL,
    TOOL_CALL_END,
    TOOL_RESPONSE,
    TOOL_RESPONSE_END,
    RenderedSFT,
    ToolCallParseError,
    parse_tool_call,
    render_sft,
    validate_arguments,
)
from kestrel.data.sft_schema import SFTRow, ToolDefinition

SPECIALS = [
    IM_START,
    IM_END,
    IM_SYSTEM,
    IM_USER,
    IM_ASSISTANT,
    TOOL_CALL,
    TOOL_CALL_END,
    TOOL_RESPONSE,
    TOOL_RESPONSE_END,
]


def _tiny_tokenizer() -> Tokenizer:
    vocab: dict[str, int] = {
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
        "sunny": 3,
        "final": 4,
        "answer": 5,
    }
    for index, token in enumerate(SPECIALS, start=6):
        vocab[token] = index
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _non_tool_row() -> SFTRow:
    return SFTRow.model_validate(
        {
            "source": "assistant_public",
            "messages": [
                {"role": "system", "content": "hello"},
                {"role": "user", "content": "world"},
                {"role": "assistant", "content": "hello world"},
            ],
        }
    )


def _tool_call_row() -> SFTRow:
    return SFTRow.model_validate(
        {
            "source": "tool_local",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Seoul"}},
                        }
                    ],
                },
            ],
        }
    )


def _tool_result_row() -> SFTRow:
    return SFTRow.model_validate(
        {
            "source": "tool_local",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "hello"},
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
                {"role": "assistant", "content": "hello"},
            ],
        }
    )


def test_render_non_tool_masks_only_assistant() -> None:
    tokenizer = _tiny_tokenizer()
    rendered = render_sft(_non_tool_row(), tokenizer)

    assert isinstance(rendered, RenderedSFT)
    assert len(rendered.token_ids) == len(rendered.loss_mask)

    ids = rendered.token_ids
    mask = rendered.loss_mask
    assert ids[0] == tokenizer.token_to_id(IM_START)
    assert tokenizer.token_to_id(IM_SYSTEM) in ids
    assert tokenizer.token_to_id(IM_USER) in ids

    assistant_index = ids.index(tokenizer.token_to_id(IM_ASSISTANT))
    assert mask[assistant_index] == 0
    assert mask[assistant_index + 1] == 1

    assistant_end_index = ids.index(tokenizer.token_to_id(IM_END), assistant_index)
    assert mask[assistant_end_index] == 1

    system_index = ids.index(tokenizer.token_to_id(IM_SYSTEM))
    assert mask[system_index] == 0
    user_index = ids.index(tokenizer.token_to_id(IM_USER))
    assert mask[user_index] == 0


def test_render_tool_call_marks_call_payload() -> None:
    tokenizer = _tiny_tokenizer()
    rendered = render_sft(_tool_call_row(), tokenizer)

    ids = rendered.token_ids
    mask = rendered.loss_mask
    tool_call_index = ids.index(tokenizer.token_to_id(TOOL_CALL))
    assert mask[tool_call_index] == 1
    assert mask[tool_call_index + 1] == 1

    tool_call_end_index = ids.index(tokenizer.token_to_id(TOOL_CALL_END))
    assert mask[tool_call_end_index] == 1

    assistant_end_index = ids.index(tokenizer.token_to_id(IM_END), tool_call_index)
    assert mask[assistant_end_index] == 1


def test_render_tool_result_masks_tool_but_marks_final_answer() -> None:
    tokenizer = _tiny_tokenizer()
    rendered = render_sft(_tool_result_row(), tokenizer)

    ids = rendered.token_ids
    mask = rendered.loss_mask

    tool_response_index = ids.index(tokenizer.token_to_id(TOOL_RESPONSE))
    assert mask[tool_response_index] == 0
    assert mask[tool_response_index + 1] == 0

    tool_response_end_index = ids.index(tokenizer.token_to_id(TOOL_RESPONSE_END))
    assert mask[tool_response_end_index] == 0

    assistant_index = ids.index(tokenizer.token_to_id(IM_ASSISTANT), tool_response_end_index)
    assert mask[assistant_index + 1] == 1


def test_render_text_contains_reserved_markers() -> None:
    tokenizer = _tiny_tokenizer()
    rendered = render_sft(_non_tool_row(), tokenizer)

    assert IM_START in rendered.text
    assert IM_SYSTEM in rendered.text
    assert IM_USER in rendered.text
    assert IM_ASSISTANT in rendered.text
    assert IM_END in rendered.text


def test_render_without_tools_omits_tool_block() -> None:
    tokenizer = _tiny_tokenizer()
    rendered = render_sft(_non_tool_row(), tokenizer)
    assert "Available tools:" not in rendered.text


def test_render_tools_adds_loss_masked_tool_block() -> None:
    tokenizer = _tiny_tokenizer()
    rendered = render_sft(_tool_call_row(), tokenizer)

    assert "Available tools:" in rendered.text
    assert '"get_weather"' in rendered.text

    ids = rendered.token_ids
    mask = rendered.loss_mask
    system_index = ids.index(tokenizer.token_to_id(IM_SYSTEM))
    tool_block_end = ids.index(tokenizer.token_to_id(IM_END), system_index)
    assert all(loss == 0 for loss in mask[system_index : tool_block_end + 1])


def _tools() -> list[ToolDefinition]:
    return [
        ToolDefinition.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                },
            }
        )
    ]


def test_parse_tool_call_with_markers() -> None:
    payload = '{"name":"get_weather","arguments":{"city":"Seoul"}}'
    text = f"{TOOL_CALL}\n{payload}\n{TOOL_CALL_END}"
    parsed = parse_tool_call(text, _tools())
    assert parsed.name == "get_weather"
    assert parsed.arguments == {"city": "Seoul"}


def test_parse_tool_call_without_markers() -> None:
    payload = '{"name":"get_weather","arguments":{"city":"Seoul","days":2}}'
    parsed = parse_tool_call(payload, _tools())
    assert parsed.arguments == {"city": "Seoul", "days": 2}


def test_parse_tool_call_invalid_json() -> None:
    text = f"{TOOL_CALL}\nnot-json\n{TOOL_CALL_END}"
    with pytest.raises(ToolCallParseError, match="invalid tool-call JSON"):
        parse_tool_call(text, _tools())


def test_parse_tool_call_unknown_tool() -> None:
    payload = '{"name":"get_forecast","arguments":{"city":"Seoul"}}'
    with pytest.raises(ToolCallParseError, match="unknown tool name"):
        parse_tool_call(payload, _tools())


def test_parse_tool_call_schema_invalid_arguments() -> None:
    payload = '{"name":"get_weather","arguments":{"days":2}}'
    with pytest.raises(ToolCallParseError, match="schema validation"):
        parse_tool_call(payload, _tools())


def test_parse_tool_call_arguments_must_be_object() -> None:
    payload = '{"name":"get_weather","arguments":"Seoul"}'
    with pytest.raises(ToolCallParseError, match="object arguments"):
        parse_tool_call(payload, _tools())


def test_validate_arguments_supports_flat_types_and_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "unit"],
        "additionalProperties": False,
    }
    assert validate_arguments(schema, {"city": "Seoul", "unit": "celsius"}) == []
    assert validate_arguments(schema, {"city": "Seoul", "unit": "kelvin"}) != []
    assert validate_arguments(schema, {"city": "Seoul", "unit": "celsius", "extra": 1}) != []
