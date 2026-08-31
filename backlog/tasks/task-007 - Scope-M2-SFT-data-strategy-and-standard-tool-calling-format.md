---
id: TASK-007
title: Scope M2 SFT data strategy and standard tool-calling format
status: Done
assignee: []
created_date: '2026-08-31 00:27'
updated_date: '2026-08-31 01:21'
labels:
  - sft
  - planning
  - research
dependencies: []
priority: high
ordinal: 37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Planning task for M2 SFT scope. M1 pretraining is complete. M2 builds the SFT stack and validates on 50M first, with 150M supported by config.

Locked direction:
- M2 is short-context SFT validation at 1k/2k context.
- Long-context extension is deferred to a later milestone.
- Use a hybrid 50k-example SFT mixture.
- Default mixture includes internal LLM-generated data.
- Use the standard logical messages/tools/tool_calls schema.
- Render M2 assistant tool calls as simple JSON with name and arguments.
- Use one public tool dataset and one bounded normalizer.
- Internal LLM endpoint/key/model are env-based and must not be committed.

Default 50k mixture:
- 22.5k public assistant/instruction
- 7.5k GSM8K train
- 10k local rule-based tool-use
- 5k public tool-calling
- 5k internal LLM-generated

Fallback no-internal-LLM mixture:
- 22.5k public assistant/instruction
- 7.5k GSM8K train
- 12.5k local rule-based tool-use
- 7.5k public tool-calling
- 0 internal LLM

Remaining open items:
- exact public assistant source confirmation, likely Smol-SmolTalk train
- exact public tool dataset selection after sample inspection
- final rendered chat template details once the locked format is documented
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child tasks capture the locked M2 SFT data strategy and tool-calling format
- [x] #2 Notes are standalone enough for a cold agent to continue without chat context
- [x] #3 No SFT implementation starts before the locked tool-calling format is documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Locked follow-up decisions:
- Public assistant source: Smol-SmolTalk train.
- Public tool source: argilla/apigen-function-calling, call-only, normalized to Kestrel logical schema.
- Local tool generator: 10k slice with 6k direct single-call, 1.5k no-tool, 1k distractor-heavy, 750 missing-info, 750 hard variation.
- Tool eval: unseen tool names/schemas are required.
- Eval philosophy: loss is sanity only; success is task-based improvement over pretrain baseline plus sane data-scaling trend.
- Remaining open items are now mainly implementation details: exact Smol-SmolTalk sampling/filtering, public tool normalizer edge cases, internal LLM prompt templates, and SFT hyperparameters.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M2 SFT scope is locked and documented in doc-006 and doc-007. Decision tasks TASK-007.01 and TASK-007.02 are Done. TASK-007.03 is In Progress as the M2 implementation umbrella, with milestone m-2 and concrete implementation tasks TASK-007.03.01 through TASK-007.03.10.
<!-- SECTION:FINAL_SUMMARY:END -->
