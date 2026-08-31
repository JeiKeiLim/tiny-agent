---
id: TASK-007.03.05
title: Add public tool normalizer for argilla/apigen-function-calling
status: To Do
assignee: []
created_date: '2026-08-31 01:20'
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
- [ ] #1 Normalizer accepts both OpenAI-style and xLAM-style tool definitions
- [ ] #2 Normalizer keeps only single-call flat-argument rows
- [ ] #3 Normalizer emits source-tagged Kestrel logical rows
- [ ] #4 Manifest records accepted/dropped counts and seed
- [ ] #5 make check passes
<!-- AC:END -->
