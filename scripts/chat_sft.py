"""Interactive multi-turn chat CLI for Kestrel SFT checkpoints.

This is a manual inspection tool. It renders prompts with the same SFT chat
template used during training, but does not expose tool calling.

Usage:
    uv run python scripts/chat_sft.py --checkpoint checkpoints/sft/50m/final
    uv run python scripts/chat_sft.py --checkpoint ... --system "You are helpful."
"""

from __future__ import annotations

import argparse
from typing import Any

import truststore

truststore.inject_into_ssl()

from tokenizers import Tokenizer  # noqa: E402

from kestrel.common.config import load_config  # noqa: E402
from kestrel.data.chat import (  # noqa: E402
    build_chat_prompt,
    extract_assistant_content,
    mask_special_tokens,
)
from kestrel.model.config import ModelConfig  # noqa: E402
from kestrel.model.generate import generate  # noqa: E402
from kestrel.model.io import load  # noqa: E402

DEFAULT_CONFIG = "configs/kestrel/50m/model.yaml"
DEFAULT_TOKENIZER = "checkpoints/tokenizer/tokenizer.json"
EXIT_COMMANDS = {"exit", "quit", "q"}


def _read_user_line() -> str | None:
    try:
        return input("you> ")
    except EOFError:
        return None
    except KeyboardInterrupt:
        print("\ninterrupted")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with a Kestrel SFT checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="model config YAML")
    parser.add_argument(
        "--checkpoint",
        required=True,
        default=argparse.SUPPRESS,
        help="checkpoint directory",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer.json path")
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="maximum generated tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="repetition penalty")
    args = parser.parse_args()

    config = load_config(args.config, ModelConfig)
    model = load(config, args.checkpoint)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    history: list[dict[str, Any]] = []
    if args.system is not None:
        history.append({"role": "system", "content": args.system})

    print(f"chat with {args.checkpoint}; type exit, quit, or q to stop")
    while True:
        user_line = _read_user_line()
        if user_line is None:
            break
        user_content = user_line.strip()
        if not user_content:
            continue
        if user_content.lower() in EXIT_COMMANDS:
            break

        history.append({"role": "user", "content": user_content})
        prompt = build_chat_prompt(history, tokenizer)
        generated = generate(
            model,
            tokenizer,
            prompt,
            args.max_tokens,
            temp=args.temp,
            repetition_penalty=args.repetition_penalty,
        )
        assistant_content = extract_assistant_content(generated)
        if not assistant_content:
            assistant_content = "(empty response)"

        history.append({"role": "assistant", "content": assistant_content})
        print(f"model> {mask_special_tokens(assistant_content)}")


if __name__ == "__main__":
    main()
