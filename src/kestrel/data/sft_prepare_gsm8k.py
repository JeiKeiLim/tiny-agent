"""GSM8K math data preparation for M2 SFT."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

from kestrel.data.sft_schema import SFTRow

_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")
_WHITESPACE = re.compile(r"\s+")


def _clean_gsm8k_text(text: str) -> str:
    """Remove calculator annotations and normalize line whitespace."""
    cleaned = _CALC_ANNOTATION.sub("", text)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line)


def convert_gsm8k_row(raw: dict[str, Any], source: str) -> SFTRow | None:
    """Convert one GSM8K row into a user problem + assistant CoT/final row."""
    question = raw.get("question")
    answer = raw.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None

    question = question.strip()
    if not question or "####" not in answer:
        return None

    reasoning_raw, final_raw = answer.split("####", 1)
    final_answer = final_raw.strip()
    if not final_answer:
        return None

    reasoning = _clean_gsm8k_text(reasoning_raw)
    if reasoning:
        assistant = f"{reasoning}\n\nFinal answer: {final_answer}"
    else:
        assistant = f"Final answer: {final_answer}"

    try:
        return SFTRow.model_validate(
            {
                "source": source,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    except ValidationError:
        return None


def load_gsm8k_rows(dataset_id: str, dataset_config: str, split: str) -> Iterator[dict[str, Any]]:
    """Stream GSM8K rows from Hugging Face."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, name=dataset_config, split=split, streaming=True)
    return iter(dataset)
