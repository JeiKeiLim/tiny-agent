"""Held-out SFT eval bundle preparation for the M2 scorecard."""

from __future__ import annotations

from tokenizers import Tokenizer

from kestrel.data.sft_prepare import (
    SFTDataConfig,
    SFTDataEvalConfig,
    SourceManifest,
    prepare_rows,
    write_tool_split,
)
from kestrel.data.sft_prepare_gsm8k import convert_gsm8k_row, load_gsm8k_rows
from kestrel.data.sft_prepare_public import convert_smol_row, load_smol_rows
from kestrel.data.sft_tool_generator import ToolGeneratorConfig, generate_tool_eval


def prepare_eval(config: SFTDataConfig) -> dict[str, SourceManifest]:
    """Prepare the held-out assistant, GSM8K, and local tool eval bundle."""
    eval_config = config.eval or SFTDataEvalConfig()
    manifests: list[SourceManifest] = []

    manifests.append(
        prepare_rows(
            source=eval_config.assistant_source,
            dataset_id=eval_config.assistant_dataset_id,
            split=eval_config.assistant_split,
            seed=eval_config.seed,
            target_rows=eval_config.assistant_target_rows,
            tokenizer_path=config.tokenizer_path,
            context_length=config.context_length,
            output_dir=eval_config.output_dir,
            load_rows=lambda: load_smol_rows(
                eval_config.assistant_dataset_id, eval_config.assistant_split, eval_config.seed
            ),
            convert_row=lambda raw: convert_smol_row(raw, eval_config.assistant_source),
            max_candidate_rows=eval_config.assistant_max_candidate_rows,
        )
    )

    manifests.append(
        prepare_rows(
            source=eval_config.gsm8k_source,
            dataset_id=eval_config.gsm8k_dataset_id,
            split=eval_config.gsm8k_split,
            seed=eval_config.seed,
            target_rows=eval_config.gsm8k_target_rows,
            tokenizer_path=config.tokenizer_path,
            context_length=config.context_length,
            output_dir=eval_config.output_dir,
            load_rows=lambda: load_gsm8k_rows(
                eval_config.gsm8k_dataset_id,
                eval_config.gsm8k_dataset_config,
                eval_config.gsm8k_split,
                eval_config.seed,
            ),
            convert_row=lambda raw: convert_gsm8k_row(raw, eval_config.gsm8k_source),
            dataset_config=eval_config.gsm8k_dataset_config,
        )
    )

    if eval_config.tool_eval:
        generator_config = ToolGeneratorConfig(
            seed=eval_config.seed,
            min_tools=config.tool.min_tools,
            max_tools=config.tool.max_tools,
            train=config.tool.train,
            eval=config.tool.eval,
        )
        eval_rows = generate_tool_eval(generator_config)
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
        splits = {
            "eval_seen": ("tool_eval_seen", "eval_seen", list(eval_rows.seen)),
            "eval_unseen": ("tool_eval_unseen", "eval_unseen", list(eval_rows.unseen)),
            "eval_no_call": ("tool_eval_no_call", "eval_no_call", list(eval_rows.no_call)),
            "eval_missing_info": (
                "tool_eval_missing_info",
                "eval_missing_info",
                list(eval_rows.missing_info),
            ),
        }
        manifests.extend(
            write_tool_split(
                output_dir=eval_config.output_dir,
                source=source,
                split=split,
                seed=eval_config.seed,
                rows=rows,
                tokenizer=tokenizer,
                context_length=config.context_length,
            )
            for split, (source, manifest_split, rows) in splits.items()
        )

    return {manifest.source: manifest for manifest in manifests}
