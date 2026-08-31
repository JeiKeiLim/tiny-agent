---
id: TASK-007.03.05
title: Add public tool normalizer for argilla/apigen-function-calling
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 07:57'
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
ordinal: 45000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the public tool-calling normalizer for the selected source.

Depends on:
- TASK-007.03.01

Files:
- src/kestrel/data/sft_public_tool.py
- tests/data/test_sft_public_tool.py
- tests/data/fixtures/sft_public_tool_fixture.jsonl

Source:
- argilla/apigen-function-calling
- primary target count: 5,000 rows for default M2 mixture
- fallback target count: 7,500 rows for no-internal-LLM mixture

Scope:
- Load or read source rows from a local cache/download path.
- Parse tools JSON strings.
- Handle both:
  - OpenAI-style JSON Schema tools
  - xLAM-style Python-type parameter maps
- Parse answers JSON strings.
- Keep only rows with exactly one expected tool call.
- Reject plain-text refusal answers.
- Reject nested or dict-valued arguments.
- Keep flat arguments only: string, int, float, bool, enum, short list.
- Cap tools per row at 5 or fewer.
- Enforce length caps suitable for 50M short-context SFT.
- Deduplicate by hash_id, id, and normalized query.
- Exclude function names that overlap the M2 tool eval set.
- Emit Kestrel logical rows with source=tool_public.
- Public rows are call-only: system, user, assistant tool_calls. No tool-role result or final answer is required.

Acceptance:
- Fixture tests cover both tool definition formats.
- Invalid rows are dropped and counted in the manifest.
- Output rows validate against the SFT schema.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Normalizer accepts both OpenAI-style and xLAM-style tool definitions
- [x] #2 Normalizer keeps only single-call flat-argument rows
- [x] #3 Normalizer emits source-tagged Kestrel logical rows
- [x] #4 Manifest records accepted/dropped counts and seed
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/data/sft_public_tool.py with a stateful PublicToolNormalizer for argilla/apigen-function-calling rows.
2. Parse tools JSON strings in both OpenAI-style function schemas and xLAM-style Python-type parameter maps.
3. Convert xLAM types str/int/float/bool and List[primitive] to flat JSON Schema; reject nested objects, dict arguments, unsupported types, and more than 5 tools.
4. Parse answers JSON strings, keep only exactly one expected tool call, reject plain-text refusals, unknown tools, schema-invalid arguments, and non-flat arguments.
5. Deduplicate accepted rows by id, hash_id, and normalized query; enforce query/tool length caps; exclude M2 local eval tool names.
6. Emit call-only Kestrel SFTRow values with source=tool_public: system, user, assistant tool_calls.
7. Add tests/data/fixtures/sft_public_tool_fixture.jsonl and tests/data/test_sft_public_tool.py covering both tool formats, invalid rows, dedup, exclusions, and schema validation.
8. Wire prepare_public_tool(), PublicToolSourceConfig, configs/kestrel/sft_data.yaml, and scripts/run_prepare_sft.py --source public_tool with manifest accepted/dropped counts.
9. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added src/kestrel/data/sft_public_tool.py with PublicToolNormalizer and load_public_tool_rows.
The normalizer accepts OpenAI-style function tools and xLAM-style Python-type parameter maps.
xLAM str/int/float/bool and List[primitive] parameters are converted to flat JSON Schema; nested objects, dict arguments, unsupported types, multi-call answers, plain-text refusals, unknown tools, schema-invalid arguments, and rows with more than 5 tools are dropped.
Deduplication is applied by id, hash_id, and normalized query after full row validation.
Added m2_eval_tool_names() to src/kestrel/tools/schema_sampler.py and excluded those local eval tool names from public training rows.
Public rows are call-only and use source=tool_public with system, user, and assistant tool_calls messages.
Added tests/data/fixtures/sft_public_tool_fixture.jsonl and tests/data/test_sft_public_tool.py.
Wired PublicToolSourceConfig, prepare_public_tool(), configs/kestrel/sft_data.yaml, and scripts/run_prepare_sft.py --source public_tool.
prepare_public_tool prefers origin=distilabel rows, fills remaining target rows from other origins, and records dropped_rows in the manifest.
Ran the real public tool prep successfully: tool_public wrote 5000/5000 rows with 82776 dropped rows.
Validation: make check passed with 268 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a public tool-calling normalizer for argilla/apigen-function-calling that accepts OpenAI-style and xLAM-style tool definitions, keeps only single-call flat-argument rows, excludes M2 local eval tool names, and writes source-tagged call-only tool_public.jsonl data with accepted/dropped manifest counts.
<!-- SECTION:FINAL_SUMMARY:END -->
