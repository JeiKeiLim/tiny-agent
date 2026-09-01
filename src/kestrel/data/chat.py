"""Helpers for interactive chat with Kestrel SFT checkpoints."""

from __future__ import annotations

from typing import Any

from tokenizers import Tokenizer

from kestrel.data.sft_chat import IM_ASSISTANT, completion_prompt_text
from kestrel.data.sft_schema import SFTRow
from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS


def build_chat_prompt(messages: list[dict[str, Any]], tokenizer: Tokenizer) -> str:
    """Render a chat history with the SFT template and assistant prefix."""
    row = SFTRow.model_validate({"source": "chat", "messages": messages})
    return completion_prompt_text(row, tokenizer)


def _cut_at_special_tokens(text: str) -> str:
    cutoff = len(text)
    for token in DEFAULT_SPECIAL_TOKENS:
        index = text.find(token)
        if index != -1:
            cutoff = min(cutoff, index)
    return text[:cutoff]


def extract_assistant_content(text: str) -> str:
    """Extract printable assistant content from generated chat text.

    If the generated text starts with the assistant role marker, the marker is
    removed. Any subsequent special-token boundary cuts the response.
    """
    assistant_index = text.find(IM_ASSISTANT)
    if assistant_index != -1:
        text = text[assistant_index + len(IM_ASSISTANT) :]
    return _cut_at_special_tokens(text).strip()


def mask_special_tokens(text: str) -> str:
    """Replace special tokens with safe placeholders for display."""
    masked = text
    for index, token in enumerate(DEFAULT_SPECIAL_TOKENS):
        masked = masked.replace(token, f"[SPECIAL_{index}]")
    return masked
