"""Tokenizer training configuration (Pydantic model loaded from YAML)."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from kestrel.common.config import BaseConfig

# ChatML + tool-call tokens, matching the Qwen2.5 / Mistral convention. Baked
# in at training time so their IDs are reserved in the vocab (im_end doubles
# as the EOS/stop token).
DEFAULT_SPECIAL_TOKENS = [
    "im_start",
    "im_end",
    "im_system",
    "im_user",
    "im_assistant",
    "tool_call",
    "tool_call_end",
    "tool_response",
    "tool_response_end",
]


class TokenizerConfig(BaseConfig):
    """How to train the byte-level BPE tokenizer (plan §7).

    The vocab holds 256 byte-tokens plus ``special_tokens`` plus merged tokens,
    so ``vocab_size`` must fit all of them. Special tokens are baked in at
    training time (not added later) so both model sizes share one tokenizer.
    """

    vocab_size: int = Field(default=16384, gt=0)
    train_dir: str = "data/tokenizer_train"
    output_dir: str = "checkpoints/tokenizer"
    eos_token: str = "im_end"
    min_frequency: int = Field(default=2, gt=0)
    initial_alphabet: str = '{}[]":,'
    special_tokens: list[str] = Field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))

    @model_validator(mode="after")
    def _check_special_tokens(self) -> Self:
        if len(set(self.special_tokens)) != len(self.special_tokens):
            msg = f"duplicate special tokens: {self.special_tokens}"
            raise ValueError(msg)
        if self.eos_token not in self.special_tokens:
            msg = f"eos_token {self.eos_token!r} must be one of special_tokens"
            raise ValueError(msg)
        if len(self.special_tokens) + 256 > self.vocab_size:
            msg = (
                f"vocab_size {self.vocab_size} must fit 256 byte tokens plus "
                f"{len(self.special_tokens)} special tokens"
            )
            raise ValueError(msg)
        return self
