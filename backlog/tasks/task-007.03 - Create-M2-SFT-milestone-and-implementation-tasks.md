---
id: TASK-007.03
title: Create M2 SFT milestone and implementation tasks
status: In Progress
assignee: []
created_date: '2026-08-31 00:27'
updated_date: '2026-08-31 01:21'
labels:
  - sft
  - planning
dependencies:
  - TASK-007.01
  - TASK-007.02
parent_task_id: TASK-007
priority: high
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the locked SFT data strategy and tool-calling format are documented, create the M2 SFT milestone and implementation tasks in Backlog.

Scope to cover:
- SFT chat template implementation using the locked standard logical schema and simple JSON rendered tool-call payload.
- SFT dataset config and SFTDataset module for example-based JSONL training.
- Data prep for the locked default 50k mixture:
  - 22.5k public assistant/instruction
  - 7.5k GSM8K train
  - 10k local rule-based tool-use
  - 5k public tool-calling normalized from one selected public dataset
  - 5k internal LLM-generated data
- Fallback no-internal-LLM 50k mixture:
  - 22.5k public assistant/instruction
  - 7.5k GSM8K train
  - 12.5k local rule-based tool-use
  - 7.5k public tool-calling
  - 0 internal LLM
- Internal LLM generator with env-based endpoint/key/model configuration and secret-safe .env handling.
- Public tool dataset inspection and one bounded normalizer.
- Unified SFT JSONL + manifest output.
- SFT trainer entry point, checkpointing, and resume behavior.
- SFT evaluation comparing pretrain-only baseline against SFT checkpoints.
- 50M first validation runs; 150M config-supported but not the first execution target.

Constraints:
- Do not commit internal LLM endpoint, API key, or model name.
- Keep make check green.
- Update README.md and AGENTS.md if commands, layout, checkpoint format, or pipeline stages change.
- max_examples subset runs should preserve source ratios as closely as practical.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 M2 milestone exists with ordered SFT implementation tasks
- [x] #2 Implementation tasks cover chat template, SFTDataset, data prep, public tool normalizer, internal LLM generator, trainer, and eval
- [x] #3 Each implementation task is standalone with file paths, config fields, targets, tests, and verification gate
- [x] #4 Tasks include both default 50k mixture and no-internal-LLM fallback mixture
- [x] #5 Implementation tasks include the locked local tool generator breakdown and unseen-schema eval split
- [x] #6 Implementation tasks include the eval philosophy: loss sanity plus task-based success signals
- [x] #7 Public tool normalizer task targets argilla/apigen-function-calling with the locked filter rules
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Locked local tool generator design for the 10k local rule-based tool slice:
- tool domains: weather lookup, unit conversion, calculator, date/time math, simple lookup, document search, simple database/record lookup, inventory/record lookup
- 3-5 tool definitions per prompt, exactly one relevant tool, others are distractors
- flat arguments only: string/int/float/bool/enum/short list; no nested objects in M2
- deterministic mock JSON tool results; no side-effect tools
- 10k breakdown:
  - 6k direct single-call examples with tool result and final answer
  - 1.5k no-tool-needed examples
  - 1k distractor-heavy examples
  - 750 missing-information/clarification examples
  - 750 hard variation examples
- train/eval separation uses unseen tool names/schemas for eval
- generator outputs the locked logical messages/tools/tool_calls schema with source tag tool_local

Locked eval philosophy:
- SFT loss is a sanity signal, not the primary success metric
- track total and per-source validation loss plus train/val gap
- primary success signals are task-based:
  - GSM8K final-answer exact match improves over pretrain baseline
  - tool-call validity/schema-validity/tool-selection improve over pretrain baseline
  - unseen tool schema accuracy is non-trivial
  - no-call correctness remains sane
  - assistant sanity does not collapse
  - pretrain perplexity does not explode
- M2 success means measurable improvement over baseline plus sane 5k/10k/20k/50k scaling trend, not frontier capability

Public tool implementation target:
- normalizer reads argilla/apigen-function-calling
- filters single-call rows, flat args, <=5 tools, length caps, dedup, eval-name exclusion
- emits call-only Kestrel rows with assistant tool_calls

Created milestone m-2 "M2 SFT Validation" and implementation tasks:
- TASK-007.03.01 Add SFT logical schema and chat template
- TASK-007.03.02 Add SFTDataset and SFTDatasetConfig
- TASK-007.03.03 Add public assistant and GSM8K SFT data prep
- TASK-007.03.04 Add local rule-based tool SFT generator
- TASK-007.03.05 Add public tool normalizer for argilla/apigen-function-calling
- TASK-007.03.06 Add internal LLM SFT data generator
- TASK-007.03.07 Add SFT mixture builder and manifest
- TASK-007.03.08 Add SFT trainer phase and entry point
- TASK-007.03.09 Add SFT eval harness and baseline scorecard
- TASK-007.03.10 Run 50M SFT data-scaling validation

Decision docs:
- doc-006 M2 SFT Data Strategy Decision
- doc-007 M2 SFT Tool-Calling Format Decision
<!-- SECTION:NOTES:END -->
