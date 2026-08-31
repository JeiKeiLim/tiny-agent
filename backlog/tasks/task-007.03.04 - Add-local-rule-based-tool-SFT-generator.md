---
id: TASK-007.03.04
title: Add local rule-based tool SFT generator
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 07:30'
labels:
  - sft
  - data
  - tool-calling
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.01
parent_task_id: TASK-007.03
priority: high
ordinal: 44000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the local rule-based tool-use generator for the 10k local slice.

Depends on:
- TASK-007.03.01

Files:
- src/kestrel/data/sft_tool_generator.py
- src/kestrel/tools/schema_sampler.py if separate schema sampling is useful
- tests/data/test_sft_tool_generator.py
- tests/tools/test_schema_sampler.py

Locked design:
- Tool domains: weather lookup, unit conversion, calculator, date/time math, simple lookup, document search, simple database/record lookup, inventory/record lookup.
- 3-5 tool definitions per prompt.
- Exactly one relevant tool; other tools are distractors.
- Flat arguments only: string, int, float, bool, enum, short list.
- No nested objects in M2.
- Deterministic mock JSON tool results.
- No side-effect tools.
- Train and eval tool schema families must be disjoint where unseen-schema eval is required.

10k train breakdown:
- 6,000 direct single-call examples with tool result and final answer
- 1,500 no-tool-needed examples
- 1,000 distractor-heavy examples
- 750 missing-information/clarification examples
- 750 hard variation examples

Eval output:
- seen-schema tool eval set
- unseen-schema tool eval set
- no-call eval set
- missing-info eval set

Acceptance:
- Generator is seeded and deterministic.
- All generated rows validate against the SFT schema.
- All tool-call arguments validate against the generated tool schemas.
- Unseen eval tool names/schemas do not appear in train output.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generator produces the locked 10k train breakdown
- [x] #2 Generated rows use the standard logical messages/tools/tool_calls schema
- [x] #3 Tool schemas are sampled and not hardcoded to one fixed registry
- [x] #4 Unseen-schema eval rows use tool names/schemas absent from train rows
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/tools/schema_sampler.py with train and unseen tool families for weather, unit conversion, calculator, date/time, lookup, document search, record lookup, and inventory.
2. Keep M2 tool schemas flat: string, integer, number, boolean, enum, and short list only; no nested argument objects.
3. Add src/kestrel/data/sft_tool_generator.py with strict Pydantic config for the locked 10k train breakdown and eval set sizes.
4. Generate direct, no-tool, distractor-heavy, missing-info, and hard-variation train rows using seeded random.Random.
5. Generate seen-schema, unseen-schema, no-call, and missing-info eval rows.
6. Use the existing SFTRow logical schema and validate every generated row plus every tool-call argument against its sampled JSON Schema.
7. Ensure unseen eval tool names and schemas are disjoint from train rows.
8. Add tests/data/test_sft_tool_generator.py and tests/tools/test_schema_sampler.py.
9. Run make check and fix all failures.
10. Wire the generator into SFTDataConfig, prepare_tool(), scripts/run_prepare_sft.py --source tool, and tests so it writes tool_local.jsonl plus eval JSONL files with manifest entries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added src/kestrel/tools/schema_sampler.py with 8 train and 8 unseen flat tool families covering weather, unit conversion, calculator, date/time, lookup, document search, record lookup, and inventory.
Added src/kestrel/data/sft_tool_generator.py with strict Pydantic generator config, seeded train/eval generation, and row/tool-name helpers.
Train generation implements the locked 10k breakdown: 6000 direct, 1500 no-tool, 1000 distractor-heavy, 750 missing-info, and 750 hard-variation rows.
Eval generation produces seen-schema, unseen-schema, no-call, and missing-info rows.
Every generated row is constructed through SFTRow validation, and every generated tool call is validated against its row-level JSON Schema.
Unseen eval rows use UNSEEN_TOOL_FAMILIES only, while train rows use TRAIN_TOOL_FAMILIES only, keeping unseen tool names disjoint from train output.
Added tests/tools/test_schema_sampler.py and tests/data/test_sft_tool_generator.py.
Validation: make check passed with 250 tests.

Reopened: the generator needs the obvious data-prep wiring so it can write tool_local.jsonl and eval JSONL files via scripts/run_prepare_sft.py.

Wired the local tool generator into SFT data preparation.
Added ToolSourceConfig to SFTDataConfig and configs/kestrel/sft_data.yaml.
Added prepare_tool() to write tool_local.jsonl plus tool_eval_seen, tool_eval_unseen, tool_eval_no_call, and tool_eval_missing_info JSONL files with manifest entries.
Extended scripts/run_prepare_sft.py with --source tool.
Added tests/data/test_sft_prepare_tool.py.
Ran the real local tool prep successfully:
- tool_local: 10000/10000 rows
- tool_eval_seen: 500/500 rows
- tool_eval_unseen: 500/500 rows
- tool_eval_no_call: 250/250 rows
- tool_eval_missing_info: 250/250 rows
Validation: make check passed with 253 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a seeded local rule-based tool SFT generator with sampled flat tool schemas, the locked 10k train breakdown, disjoint unseen-schema eval rows, and data-prep CLI wiring that writes train and eval JSONL files.
<!-- SECTION:FINAL_SUMMARY:END -->
