---
id: TASK-007.03.06
title: Add internal LLM SFT data generator
status: To Do
assignee: []
created_date: '2026-08-31 01:20'
labels:
  - sft
  - data
  - internal-llm
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.01
parent_task_id: TASK-007.03
priority: high
ordinal: 46000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement optional internal LLM generation for the 5k internal LLM slice.

Depends on:
- TASK-007.03.01

Files:
- src/kestrel/data/sft_internal_llm.py
- tests/data/test_sft_internal_llm.py
- .gitignore update if .env is not already ignored

Locked slice:
- 2,000 assistant/instruction examples
- 2,000 math word problems with step-by-step solutions
- 1,000 tool-calling dialogues in the Kestrel schema

Scope:
- Config uses environment variable names only:
  - api_base_env
  - api_key_env
  - model_env
- Actual endpoint, API key, and model name must come from the environment.
- Generator runs offline during data prep, not during training.
- Generated rows must be validated against the Kestrel SFT schema.
- Tool rows must use sampled tool schemas and the standard tool_calls structure.
- Math rows must include final answer format compatible with GSM8K-style eval.
- Output source=internal_llm JSONL under data/sft/raw/.
- Manifest records model_env name, not model value, prompt version, seed, counts, and hash.

Constraints:
- Do not commit secrets.
- Unit tests must mock the LLM client and must not call a real endpoint.
- If the generated dataset is published later, it can be consumed as a local JSONL source.

Acceptance:
- Config rejects missing env var names when enabled.
- Mocked generation produces the locked 2k/2k/1k split.
- All generated rows validate against the SFT schema.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Internal LLM config is env-based and secret-safe
- [ ] #2 Generator produces assistant, math, and tool rows in the Kestrel schema
- [ ] #3 Unit tests mock the LLM client and do not require a real endpoint
- [ ] #4 Output manifest records counts, seed, prompt version, and hash
- [ ] #5 make check passes
<!-- AC:END -->
