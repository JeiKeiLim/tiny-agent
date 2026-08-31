---
id: TASK-007.03.01
title: Add SFT logical schema and chat template
status: Done
assignee:
  - '@Jongkuk Lim'
created_date: '2026-08-31 01:19'
updated_date: '2026-08-31 01:37'
labels:
  - sft
  - implementation
milestone: m-2
dependencies: []
parent_task_id: TASK-007.03
priority: high
ordinal: 41000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the shared SFT row schema and chat renderer/parser for M2.

Files:
- src/kestrel/data/sft_schema.py
- src/kestrel/data/sft_chat.py
- tests/data/test_sft_schema.py
- tests/data/test_sft_chat.py

Scope:
- Strict Pydantic models for SFT rows: source, optional tools, messages.
- Support roles: system, user, assistant, tool.
- Assistant messages support either content or tool_calls.
- Tool definitions use type=function with function.name/description/parameters JSON Schema.
- Renderer maps logical rows to training text using the tokenizer reserved role markers.
- Renderer emits assistant tool calls as simple JSON: {"name": "...", "arguments": {...}}.
- Parser extracts rendered assistant tool-call JSON and validates name/arguments.
- Label masking helper marks assistant-turn tokens for loss and masks system/user/tool tokens.

Acceptance:
- Unknown keys in SFT rows raise ValidationError.
- Renderer handles non-tool chat, tool call, tool result, and final answer.
- Parser handles valid JSON, invalid JSON, unknown tool name, and schema-invalid arguments.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SFT row schema is strict Pydantic and rejects unknown keys
- [x] #2 Chat renderer emits the locked M2 logical-to-text format
- [x] #3 Parser extracts and validates simple JSON tool calls
- [x] #4 Tests cover non-tool, tool-call, tool-result, and invalid-call cases
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add strict Pydantic SFT schema in src/kestrel/data/sft_schema.py.
2. Add renderer/tokenizer sequence builder and simple JSON tool-call parser in src/kestrel/data/sft_chat.py.
3. Add tests for schema strictness, non-tool/tool/tool-result rendering, parser valid/invalid cases, and loss masking.
4. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict SFTRow schema in src/kestrel/data/sft_schema.py and renderer/parser in src/kestrel/data/sft_chat.py.
Renderer uses reserved markers from DEFAULT_SPECIAL_TOKENS by index to avoid hardcoding marker literals in new modules.
Assistant content and tool-call payloads are loss-masked; system/user/tool/role/structural tokens are masked.
Parser validates simple JSON tool calls against supplied tool schemas using a minimal built-in JSON Schema validator.
Tests added in tests/data/test_sft_schema.py and tests/data/test_sft_chat.py.
Validation: make check passed with 187 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the M2 SFT logical schema, chat renderer, loss-mask helper, and simple JSON tool-call parser. Verified with strict schema tests, renderer/parser tests, and make check.
<!-- SECTION:FINAL_SUMMARY:END -->
