---
id: TASK-007.03.06
title: Add internal LLM SFT data generator
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 12:00'
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
- [x] #1 Internal LLM config is env-based and secret-safe
- [x] #2 Generator produces assistant, math, and tool rows in the Kestrel schema
- [x] #3 Unit tests mock the LLM client and do not require a real endpoint
- [x] #4 Output manifest records counts, seed, prompt version, and hash
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add InternalLLMConfig to src/kestrel/data/sft_prepare.py with enabled, api_base_env, api_key_env, model_env, locked 2000/2000/1000 defaults, prompt_version, length caps, and strict validation.
2. Add src/kestrel/data/sft_internal_llm.py with an OpenAI-compatible urllib client, env resolution that never logs secret values, strict JSON prompt parsing, and converters for assistant, math, and tool rows.
3. Tool rows sample TRAIN_TOOL_FAMILIES locally, prompt the LLM for user_prompt/arguments/tool_result/final_answer, validate flat arguments against the sampled schema, and build system/user/assistant tool_call/tool/final assistant messages.
4. Math rows enforce a numeric final answer and render assistant content ending with "Final answer: <value>".
5. Add prepare_internal_llm() to src/kestrel/data/sft_prepare.py, extend SourceManifest with model_env, prompt_version, and generated_counts, add --source internal_llm to scripts/run_prepare_sft.py, and add an internal_llm section to configs/kestrel/sft_data.yaml.
6. Add tests/data/test_sft_internal_llm.py with a fake LLM client only; cover env-based config validation, missing env rejection, mocked 2k/2k/1k generation shape, schema validation, invalid response handling, manifest fields, and CLI source selection.
7. Add .env to .gitignore and update README/AGENTS data-prep documentation.
8. Run make format && make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added src/kestrel/data/sft_internal_llm.py with InternalLLMConfig, OpenAICompatibleClient, create_llm_client(), InternalLLMGenerator, and generate_internal_llm_rows().
Config stores only api_base_env, api_key_env, and model_env names. When enabled, it rejects empty env var names, and create_llm_client() rejects missing environment values without logging secret values.
The generator uses strict JSON prompts, drops invalid JSON, duplicate prompts, reserved tokenizer marker text, oversized fields, client errors, invalid math final answers, and tool rows with unknown/missing/non-flat/schema-invalid arguments.
Assistant rows are user/assistant rows. Math rows render assistant content ending with "Final answer: <numeric value>". Tool rows sample TRAIN_TOOL_FAMILIES, use the standard system/user/assistant tool_call/tool/final assistant shape, and validate arguments against the sampled JSON Schema.
Added prepare_internal_llm() to src/kestrel/data/sft_prepare.py, extended SourceManifest with model_env, prompt_version, and generated_counts, added --source internal_llm to scripts/run_prepare_sft.py, and added an internal_llm section to configs/kestrel/sft_data.yaml.
The committed config keeps internal_llm.enabled=false. prepare_all() skips the source when disabled; the explicit CLI source exits with an error when disabled.
Added tests/data/test_sft_internal_llm.py using fake LLM clients only. Tests cover locked 2000/2000/1000 mocked generation, schema validation, invalid/duplicate/reserved-token handling, client errors, env validation, manifest metadata, context filtering, and CLI behavior.
Added .env to .gitignore and updated README.md/AGENTS.md status for the M2 SFT data-prep stack.
Validation: make check passed with 282 tests.

Added .env.example documenting KESTREL_LLM_API_BASE, KESTREL_LLM_API_KEY, and KESTREL_LLM_MODEL with empty values only.
Updated README.md and AGENTS.md to reference .env.example and keep .env gitignored.
Validation: make check passed with 282 tests.

Added stderr progress reporting for internal LLM generation.
InternalLLMConfig.progress_every defaults to 50 accepted rows; 0 disables progress output.
generate_internal_llm_rows() accepts an optional progress_callback, and prepare_internal_llm() prints:
internal_llm: assistant <done>/<target>, math <done>/<target>, tool <done>/<target>
Progress is written to stderr so stdout remains manifest/output-only.
Added tests for callback updates, disabled progress, and stderr reporting.
Validation: make check passed with 285 tests.

Lowered default internal LLM progress frequency from every 50 accepted rows to every 10 accepted rows.
Updated configs/kestrel/sft_data.yaml to progress_every: 10.
For slow endpoint timing runs, use progress_every: 1 in a local timing config.
Validation: make check passed with 285 tests.

Added bounded concurrency to internal LLM generation.
InternalLLMConfig.max_workers defaults to 1 for backward compatibility; configs/kestrel/sft_data.yaml now sets max_workers: 8.
The generator pre-generates each batch of prompts sequentially from the seeded RNG, submits LLM completions to a ThreadPoolExecutor, then converts and deduplicates responses in original prompt order. This preserves deterministic prompt selection and output order while overlapping network latency.
Batch size is capped by max_workers and by the number of rows still needed, so generation stops without requesting a full extra batch after the target is reached.
Added a thread-safe BarrierFakeClient test verifying concurrent execution with max_workers=4.
Validation: make check passed with 286 tests.

Added opt-in drop debugging for internal LLM generation.
InternalLLMConfig.debug_drops and debug_drop_limit control per-category stderr drop messages.
generate_internal_llm_rows() accepts a debug_callback, and prepare_internal_llm() prints messages like:
internal_llm debug: math dropped reason=invalid_final_answer detail=1,000
Converters now return structured drop reasons for invalid JSON, missing fields, invalid final answer, oversized fields, reserved tokens, non-flat/schema-invalid tool arguments, invalid tool-result JSON, duplicate rows, and client errors.
Added tests for math drop reason, client-error reason, debug limit, and stderr reporting.
Validation: make check passed with 290 tests.

Fixed math final-answer normalization: the LLM often returns JSON numbers instead of strings, which the previous strict string-only regex rejected.
Added _normalize_math_final_answer() to accept JSON int/float values, plain numeric strings, and comma-separated numeric strings; bool/null/non-numeric values are still rejected.
Improved invalid_final_answer debug detail to show the actual received value, e.g. detail=\"got twelve\".
Added tests for numeric JSON final answers, comma normalization, and invalid debug detail.
Validation: make check passed with 292 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an optional, env-based internal LLM SFT generator that produces the locked 2k assistant / 2k math / 1k tool slice, validates all rows against the Kestrel SFT schema, writes source-tagged internal_llm.jsonl data, and records model_env, prompt version, seed, counts, and hash in the manifest.
<!-- SECTION:FINAL_SUMMARY:END -->
