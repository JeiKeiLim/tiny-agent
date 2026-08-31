---
id: TASK-007.03.13
title: Prepare held-out SFT eval datasets
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 23:43'
updated_date: '2026-08-31 23:49'
labels:
  - sft
  - data
  - eval
milestone: m-2
dependencies:
  - TASK-007.03.03
  - TASK-007.03.04
references:
  - backlog/docs/doc-001
modified_files:
  - src/kestrel/data/sft_prepare.py
  - src/kestrel/data/sft_prepare_eval.py
  - scripts/run_prepare_sft.py
  - configs/kestrel/sft_data.yaml
  - tests/data/test_sft_prepare_eval.py
  - README.md
  - AGENTS.md
parent_task_id: TASK-007.03
priority: high
ordinal: 53000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the held-out SFT eval bundle consumed by TASK-007.03.09.

The 50k SFT mixture is training data only. The SFT scorecard needs a fixed eval bundle that is not part of the training mixture.

Outcome:
- add an eval data prep source to the SFT data pipeline
- write assistant, GSM8K, and local tool eval rows under data/sft/eval
- write data/sft/eval/manifest.json with source, split, row counts, and sha256 hashes

Files:
- src/kestrel/data/sft_prepare_eval.py
- src/kestrel/data/sft_prepare.py if shared config/manifest helpers need changes
- scripts/run_prepare_sft.py
- configs/kestrel/sft_data.yaml
- tests/data/test_sft_prepare_eval.py

Config:
- add eval section to configs/kestrel/sft_data.yaml
- fields:
  - output_dir: data/sft/eval
  - seed: 42
  - assistant_split: test
  - assistant_target_rows: 200
  - gsm8k_split: test
  - gsm8k_dataset_config: main
  - gsm8k_target_rows: 500
  - tool_eval: true
- committed config must use held-out splits and safe defaults

Behavior:
- assistant eval:
  - load HuggingFaceTB/smol-smoltalk test split if available
  - convert with existing convert_smol_row()
  - target 200 valid rows after context filtering
  - if test split is unavailable, fall back to a seed-held-out subset of train only if explicitly configured, but committed config must not use train
- gsm8k eval:
  - load openai/gsm8k test/main
  - convert with existing convert_gsm8k_row()
  - target up to 500 valid rows; GSM8K test has ~1319 rows so 500 is safe
- tool eval:
  - reuse generate_tool_eval() / prepare_tool() logic
  - write tool_eval_seen.jsonl, tool_eval_unseen.jsonl, tool_eval_no_call.jsonl, tool_eval_missing_info.jsonl under data/sft/eval
  - keep default sizes 500/500/250/250
  - unseen tool names must remain disjoint from train tool names
- manifest:
  - one entry per eval source
  - include dataset_id, split, seed, requested_rows, written_rows, filtered_rows, output_path, sha256

CLI:
- uv run python scripts/run_prepare_sft.py --source eval

Tests:
- tests/data/test_sft_prepare_eval.py
- assert committed config uses assistant test split and gsm8k test/main
- assert prepare_eval writes assistant/gsm8k/tool eval files and manifest
- assert assistant and gsm8k eval use target valid rows after filtering
- assert tool eval unseen names are disjoint from train names
- assert manifest hashes match written files

Design decisions:
- Keep eval data under data/sft/eval, not data/sft/raw, so training raw files and scorecard eval files do not get confused.
- Reuse existing row converters and context filter; do not create a new SFT row schema.
- Do not include eval rows in the SFT mixture.
- Tool eval rows already exist in data/sft/raw today, but this task makes the scorecard eval bundle explicit and self-contained.

Verification:
- make check passes
- running the eval prep command writes data/sft/eval/manifest.json and all expected JSONL files
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 run_prepare_sft.py --source eval writes assistant, GSM8K, and four local tool eval JSONL files under data/sft/eval
- [x] #2 assistant eval uses held-out Smol-SmolTalk test split in committed config and writes 200 valid rows unless source exhausts
- [x] #3 GSM8K eval uses test/main in committed config and writes up to 500 valid rows
- [x] #4 tool eval includes seen, unseen, no-call, and missing-info sets with unseen tool names disjoint from train tool names
- [x] #5 data/sft/eval/manifest.json includes source, split, row counts, output path, and sha256 for each eval file
- [x] #6 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add SFTDataEvalConfig to src/kestrel/data/sft_prepare.py so SFTDataConfig can reference it without a circular import.
2. Add src/kestrel/data/sft_prepare_eval.py with prepare_eval(), reusing prepare_rows(), load_smol_rows(), load_gsm8k_rows(), convert_smol_row(), convert_gsm8k_row(), generate_tool_eval(), and a public write_tool_split() helper.
3. Expose write_tool_split() from src/kestrel/data/sft_prepare.py and use it for both raw tool prep and eval tool prep.
4. Wire --source eval into scripts/run_prepare_sft.py.
5. Add committed eval config to configs/kestrel/sft_data.yaml using assistant test split and gsm8k test/main.
6. Add tests/data/test_sft_prepare_eval.py for config, held-out splits, target valid rows, tool eval names, manifest hashes, and CLI source selection.
7. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified HuggingFaceTB/smol-smoltalk exposes train/test and openai/gsm8k main exposes train/test.

Implemented SFTDataEvalConfig in src/kestrel/data/sft_prepare.py to avoid a circular import, prepare_eval() in src/kestrel/data/sft_prepare_eval.py, public write_tool_split() reuse, CLI --source eval, committed eval config, README/AGENTS updates, and tests. make check passes with 311 tests. Real eval prep wrote assistant_eval 200/200 from 376 candidates, gsm8k_eval 500/500, tool_eval_seen 500/500, tool_eval_unseen 500/500, tool_eval_no_call 250/250, and tool_eval_missing_info 250/250 under data/sft/eval with manifest.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added held-out SFT eval bundle preparation via run_prepare_sft.py --source eval. It writes assistant_eval from Smol-SmolTalk test, gsm8k_eval from GSM8K test/main, and the four local tool eval sets under data/sft/eval with manifest.json. Added SFTDataEvalConfig, prepare_eval(), tests, committed eval config, and README/AGENTS updates. Verified with make check: 311 tests passed, and real eval prep produced 200 assistant, 500 GSM8K, 500 seen, 500 unseen, 250 no-call, and 250 missing-info eval rows.
<!-- SECTION:FINAL_SUMMARY:END -->
