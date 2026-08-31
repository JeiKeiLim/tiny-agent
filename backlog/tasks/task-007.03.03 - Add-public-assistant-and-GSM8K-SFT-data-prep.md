---
id: TASK-007.03.03
title: Add public assistant and GSM8K SFT data prep
status: Done
assignee:
  - '@Jongkuk Lim'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 06:26'
labels:
  - sft
  - data
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.01
parent_task_id: TASK-007.03
priority: high
ordinal: 43000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement data prep for the public assistant and GSM8K math slices.

Depends on:
- TASK-007.03.01

Files:
- src/kestrel/data/sft_prepare.py or src/kestrel/data/sft_prepare_public.py
- src/kestrel/data/sft_prepare_gsm8k.py
- scripts/run_prepare_sft_public.py if a CLI entry point is needed
- tests/data/test_sft_prepare_public.py
- tests/data/test_sft_prepare_gsm8k.py
- configs/kestrel/50m/sft_data.yaml if config-driven data prep is introduced

Scope:
- Public assistant source: Smol-SmolTalk train.
- Sample 22,500 rows deterministically by seed.
- Convert conversations to the Kestrel logical SFT schema.
- Filter rows by max rendered token length.
- Tag rows with source=assistant_public.
- GSM8K source: GSM8K train split, about 7,473 rows.
- Convert each problem/solution into user problem + assistant CoT/final answer.
- Tag rows with source=gsm8k_math.
- Write per-source JSONL files under data/sft/raw/.
- Write a small manifest with source name, row count, seed, dataset ID, and hash.

Constraints:
- Unit tests must not require network access; use tiny local fixtures.
- Network download paths may be tested separately or manually.
- Keep the trainer independent of Hugging Face dataset internals.

Acceptance:
- assistant_public.jsonl contains the requested count or available fixture count in tests.
- gsm8k_math.jsonl contains chat/CoT rows with source tag.
- Manifest records counts/seeds/hashes.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Smol-SmolTalk rows are converted to the Kestrel logical schema
- [x] #2 GSM8K rows are converted to user problem + assistant CoT/final answer rows
- [x] #3 Output JSONL files include source tags and manifest metadata
- [x] #4 Unit tests use local fixtures and do not require network access
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/data/sft_prepare_public.py with convert_smol_row(raw, source) and load_smol_rows(dataset_id, split).
2. Add src/kestrel/data/sft_prepare_gsm8k.py with convert_gsm8k_row(raw, source) and load_gsm8k_rows(dataset_id, dataset_config, split).
3. Add src/kestrel/data/sft_prepare.py with strict SFTDataConfig, SourceManifest, deterministic reservoir sampling, JSONL writer, SHA256 manifest update, render-based token-length filter, prepare_rows, prepare_assistant, prepare_gsm8k, and prepare_all.
4. Add scripts/run_prepare_sft.py CLI with --config and --source assistant|gsm8k|all.
5. Add configs/kestrel/sft_data.yaml pointing to data/sft/raw, checkpoints/tokenizer/tokenizer.json, context_length 1024, seed 42, assistant target 22500, GSM8K target 7500.
6. Add tests/data/test_sft_prepare_public.py and tests/data/test_sft_prepare_gsm8k.py using local row fixtures and a tiny in-test tokenizer; no network access in unit tests.
7. Run make check and fix all lint/typecheck/test failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented src/kestrel/data/sft_prepare_public.py, src/kestrel/data/sft_prepare_gsm8k.py, src/kestrel/data/sft_prepare.py, scripts/run_prepare_sft.py, and configs/kestrel/sft_data.yaml.
HF loaders use streaming=True. prepare_rows uses deterministic reservoir sampling so the full Smol-SmolTalk split does not need to be materialized in memory.
Smol rows are converted to the Kestrel logical schema and tagged assistant_public. GSM8K rows are converted to user problem + assistant CoT/final answer, with calculator annotations removed.
Output is written to data/sft/raw/<source>.jsonl and data/sft/raw/manifest.json with source, dataset ID, split, seed, requested/candidate/written/filtered counts, output path, and SHA256.
Added tests/data/conftest.py, tests/data/test_sft_prepare_public.py, and tests/data/test_sft_prepare_gsm8k.py using local fixtures and a tiny tokenizer; unit tests do not require network access.
Validation: make check passed with 228 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added public assistant and GSM8K SFT data preparation: strict config, streaming HF loaders, deterministic sampling, render-based context filtering, JSONL outputs, manifest metadata, CLI, and local-fixture tests.
<!-- SECTION:FINAL_SUMMARY:END -->
