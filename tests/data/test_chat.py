"""Tests for interactive SFT chat helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer

from kestrel.data.chat import (
    build_chat_prompt,
    extract_assistant_content,
    mask_special_tokens,
)
from kestrel.data.sft_chat import IM_ASSISTANT, IM_END, IM_START, IM_USER
from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS


def test_build_chat_prompt_contains_user_turn_but_not_assistant(
    tiny_sft_tokenizer_obj: Tokenizer,
) -> None:
    prompt = build_chat_prompt(
        [{"role": "user", "content": "What is 2+2?"}], tiny_sft_tokenizer_obj
    )

    assert IM_START in prompt
    assert IM_USER in prompt
    assert "What is 2+2?" in prompt
    assert IM_ASSISTANT not in prompt


def test_build_chat_prompt_includes_previous_assistant_turn(
    tiny_sft_tokenizer_obj: Tokenizer,
) -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello world"},
        {"role": "user", "content": "again"},
    ]

    prompt = build_chat_prompt(messages, tiny_sft_tokenizer_obj)

    assert prompt.count(IM_USER) == 2
    assert prompt.count(IM_ASSISTANT) == 1
    assert "hello world" in prompt
    assert "again" in prompt


def test_build_chat_prompt_supports_system_message(tiny_sft_tokenizer_obj: Tokenizer) -> None:
    prompt = build_chat_prompt(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        tiny_sft_tokenizer_obj,
    )

    assert "be concise" in prompt
    assert IM_USER in prompt


def test_build_chat_prompt_rejects_invalid_history(tiny_sft_tokenizer_obj: Tokenizer) -> None:
    with pytest.raises(ValidationError):
        build_chat_prompt([{"role": "assistant", "content": "hello"}], tiny_sft_tokenizer_obj)


def test_extract_assistant_content_removes_role_structure() -> None:
    raw = f"{IM_START}\n{IM_ASSISTANT}\nhello world\n{IM_END}\n"

    assert extract_assistant_content(raw) == "hello world"


def test_extract_assistant_content_handles_missing_assistant_marker() -> None:
    assert extract_assistant_content("  plain answer\n") == "plain answer"


def test_extract_assistant_content_stops_at_next_role_marker() -> None:
    raw = f"{IM_ASSISTANT}\nhello\n{IM_USER}\nnext turn"

    assert extract_assistant_content(raw) == "hello"


def test_mask_special_tokens_replaces_all_known_markers() -> None:
    text = f"{IM_START} {IM_USER} {IM_ASSISTANT} {IM_END}"

    masked = mask_special_tokens(text)

    for token in DEFAULT_SPECIAL_TOKENS:
        assert token not in masked
    assert "[SPECIAL_" in masked
