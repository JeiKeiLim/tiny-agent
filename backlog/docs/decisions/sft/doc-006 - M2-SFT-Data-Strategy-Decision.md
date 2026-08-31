---
id: doc-006
title: M2 SFT Data Strategy Decision
type: specification
created_date: '2026-08-31 01:18'
updated_date: '2026-08-31 01:19'
---
# M2 SFT Data Strategy Decision

_Status: Accepted for M2 planning. This document supersedes the older ~10k synthetic-heavy SFT assumption in `doc-001` where they conflict._

## Decision

M2 is a short-context SFT validation milestone.

- First execution target: 50M
- 150M must remain config-supported, but it is not the first execution target
- SFT runs at short context first, likely 1024 tokens for smoke/validation runs, with 2048 supported by config
- Long-context extension is deferred to a later milestone
- The full M2 dataset is 50,000 examples, not 50,000 tokens

## Default 50k mixture

| Slice | Examples | Source / purpose |
|---|---:|---|
| Public assistant/instruction | 22,500 | Smol-SmolTalk train, general assistant behavior |
| Math / GSM8K | 7,500 | GSM8K train converted to chat/CoT form |
| Local rule-based tool use | 10,000 | Controlled tool format, edge cases, unseen-schema training |
| Public tool calling | 5,000 | `argilla/apigen-function-calling`, normalized call-only tool data |
| Internal LLM generated | 5,000 | Optional-to-regenerate but included in default mixture when available |

Total: 50,000 examples.

## Internal LLM slice

The default internal LLM 5k slice is split as:

- 2,000 assistant/instruction examples
- 2,000 math word problems with step-by-step solutions
- 1,000 tool-calling dialogues in the Kestrel schema

Internal LLM generation is an offline data-prep step. The SFT trainer does not call the internal LLM.

Config must reference environment variable names only, for example:

```yaml
internal_llm:
  enabled: true
  api_base_env: KESTREL_LLM_API_BASE
  api_key_env: KESTREL_LLM_API_KEY
  model_env: KESTREL_LLM_MODEL
```

Actual endpoint, API key, and model name must not be committed. Add `.env` to `.gitignore` if used.

## No-internal-LLM fallback

If a user cannot generate or download the internal LLM slice, use this fallback 50k mixture:

| Slice | Examples |
|---|---:|
| Public assistant/instruction | 22,500 |
| Math / GSM8K | 7,500 |
| Local rule-based tool use | 12,500 |
| Public tool calling | 7,500 |
| Internal LLM generated | 0 |

If the public tool source cannot supply 7,500 clean normalized examples, shift the deficit to public assistant or local tool data and record the actual mixture in the manifest.

If the generated internal LLM dataset is published, other users can use it as a local JSONL source without needing an endpoint.

## Public assistant source

Use Smol-SmolTalk train for the public assistant/instruction slice.

Implementation should:

- sample 22,500 rows deterministically by seed
- filter by max rendered token length
- convert conversations to the Kestrel logical message schema
- hold out Smol-SmolTalk test or a separate slice for assistant sanity eval

## Public tool source

Use `argilla/apigen-function-calling` as the primary public tool-calling source.

Reasons:

- non-gated
- CC-BY-4.0
- 109,402 rows
- structured `query`, `tools`, and `answers` fields
- superset of the gated `Salesforce/xlam-function-calling-60k` data
- suitable for a 5k normalized M2 slice

The public tool slice is call-only:

- query
- tools
- expected assistant tool call

It does not require tool-result or final-answer messages. The local tool slice covers tool-result/final-answer behavior.

Filter rules:

- prefer `origin=distilabel` rows because tools are already JSON Schema
- keep only rows with exactly one expected tool call
- reject nested or dict-valued arguments
- keep flat arguments only: string, int, float, bool, enum, short list
- cap tools per row at 5 or fewer
- enforce length caps suitable for 50M short-context SFT
- deduplicate by `hash_id`, `id`, and normalized query
- exclude function names that overlap the M2 tool eval set

Fallback order:

1. `argilla/apigen-function-calling`
2. `Team-ACE/ToolACE` if full public trajectories become required
3. `Salesforce/xlam-function-calling-60k` only if gated access is acceptable
4. BFCL is eval-only, not a training source

## Local tool generator

The 10k local rule-based tool slice uses deterministic mock tools and sampled schemas.

Tool domains:

- weather lookup
- unit conversion
- calculator
- date/time math
- simple lookup
- document search
- simple database/record lookup
- inventory/record lookup

Constraints:

- 3-5 tool definitions per prompt
- exactly one relevant tool
- other tools are distractors
- flat arguments only
- no nested objects in M2
- no side-effect tools
- deterministic mock JSON results
- train and eval use disjoint tool schema families where eval requires unseen schemas

10k breakdown:

| Type | Examples |
|---|---:|
| Direct single-call with tool result and final answer | 6,000 |
| No tool needed | 1,500 |
| Distractor-heavy tool selection | 1,000 |
| Missing information / clarification | 750 |
| Hard variation / near-miss | 750 |

## Dataset implementation shape

Data prep produces per-source normalized JSONL files:

```text
data/sft/raw/assistant_public.jsonl
data/sft/raw/gsm8k_math.jsonl
data/sft/raw/tool_local.jsonl
data/sft/raw/tool_public.jsonl
data/sft/raw/internal_llm.jsonl
```

A mixer combines them into:

```text
data/sft/mixture/sft-50k.jsonl
data/sft/mixture/manifest.json
```

Each row includes a source tag. The SFT trainer reads only the unified mixture.

`max_examples` subset runs should preserve source ratios as closely as practical.

## Eval philosophy

SFT loss is a sanity signal, not the primary success metric.

Track:

- total SFT validation loss
- per-source validation loss
- train/validation loss gap
- held-out pretrain perplexity

Primary success signals:

- GSM8K final-answer accuracy improves over pretrain baseline
- tool-call validity improves over pretrain baseline
- schema-valid argument rate improves
- correct tool selection improves
- unseen tool/schema accuracy is non-trivial
- no-call correctness remains sane
- assistant sanity does not collapse
- pretrain perplexity does not explode

M2 success means measurable improvement over the pretrain baseline plus a sane 5k / 10k / 20k / 50k scaling trend, not frontier capability.
