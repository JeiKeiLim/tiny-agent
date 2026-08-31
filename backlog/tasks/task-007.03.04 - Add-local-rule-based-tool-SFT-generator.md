---
id: TASK-007.03.04
title: Add local rule-based tool SFT generator
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
- [ ] #1 Generator produces the locked 10k train breakdown
- [ ] #2 Generated rows use the standard logical messages/tools/tool_calls schema
- [ ] #3 Tool schemas are sampled and not hardcoded to one fixed registry
- [ ] #4 Unseen-schema eval rows use tool names/schemas absent from train rows
- [ ] #5 make check passes
<!-- AC:END -->
