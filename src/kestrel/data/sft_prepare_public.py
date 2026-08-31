"""Public assistant data preparation for M2 SFT (Smol-SmolTalk)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

from kestrel.data.sft_schema import SFTRow

_ALLOWED_ROLES = {"system", "user", "assistant"}


def convert_smol_row(raw: dict[str, Any], source: str) -> SFTRow | None:
    """Convert one Smol-SmolTalk row to a Kestrel SFT row.

    Returns ``None`` for rows that are empty, use unsupported roles, contain
    blank messages, or have no assistant turn.
    """
    raw_messages = raw.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None

    messages: list[dict[str, str]] = []
    has_assistant = False
    for item in raw_messages:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role not in _ALLOWED_ROLES or not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
        if role == "assistant":
            has_assistant = True

    if not has_assistant:
        return None

    try:
        return SFTRow.model_validate({"source": source, "messages": messages})
    except ValidationError:
        return None


def load_smol_rows(dataset_id: str, split: str) -> Iterator[dict[str, Any]]:
    """Stream Smol-SmolTalk rows from Hugging Face."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    return iter(dataset)
