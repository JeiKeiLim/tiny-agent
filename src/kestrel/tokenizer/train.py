"""Train the byte-level BPE tokenizer (plan §7).

Trains a BPE tokenizer on the prepared text sample (one ``.txt`` file per
source under ``train_dir``, built by ``kestrel.data.prepare_tokenizer_data``)
and saves the artifact to ``output_dir``. The artifact is a runtime output
(gitignored); this script + its config are the reproducible source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from kestrel.common.config import load_config
from kestrel.tokenizer.config import TokenizerConfig


def train(config: TokenizerConfig) -> Path:
    """Train the tokenizer on ``config.train_dir`` and save it.

    Returns the path of the saved artifact.
    """
    files = sorted(Path(config.train_dir).glob("*.txt"))
    if not files:
        msg = f"no .txt files found in {config.train_dir!r}"
        raise FileNotFoundError(msg)
    print(f"Training on {[f.name for f in files]}", flush=True)

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    # Seed the base alphabet with all 256 byte-chars so every byte has a token
    # regardless of corpus (GPT-2/Qwen convention). Guarantees a lossless
    # round-trip for any byte sequence, not just observed text.
    trainer = BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        initial_alphabet=list(ByteLevel.alphabet()),
        show_progress=True,
    )
    tokenizer.train([str(f) for f in files], trainer)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer (vocab {len(tokenizer.get_vocab())}) -> {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer.")
    parser.add_argument("--config", default="configs/tokenizer/train.yaml")
    args = parser.parse_args()
    config = load_config(args.config, TokenizerConfig)
    train(config)


if __name__ == "__main__":
    main()
