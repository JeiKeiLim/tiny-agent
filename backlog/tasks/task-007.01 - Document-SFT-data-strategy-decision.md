---
id: TASK-007.01
title: Document locked M2 SFT data strategy
status: Done
assignee: []
created_date: '2026-08-31 00:27'
updated_date: '2026-08-31 01:19'
labels:
  - sft
  - research
  - decision
dependencies: []
references:
  - 'https://huggingface.co/rajofearth/Chinchilla-1-73M'
  - 'https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk'
  - 'https://huggingface.co/datasets/HuggingFaceTB/smoltalk'
parent_task_id: TASK-007
priority: high
ordinal: 38000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Locked M2 SFT data strategy to record in a Backlog research or decision document.

Decision:
- M2 is short-context SFT validation at 1k/2k context.
- Long-context extension is deferred to a later milestone.
- The full M2 dataset is 50,000 examples, not 50,000 tokens.
- The default M2 mixture includes internal LLM-generated data.

Default 50k mixture:
- 22,500 public assistant/instruction examples
- 7,500 GSM8K train examples converted to chat/CoT form
- 10,000 locally generated rule-based tool-use examples
- 5,000 public tool-calling examples normalized to the Kestrel schema
- 5,000 internal LLM-generated examples

Internal LLM 5k target split:
- 2,000 assistant/instruction examples
- 2,000 math word problems with step-by-step solutions
- 1,000 tool-calling dialogues in the standard Kestrel schema

Reproducibility without an internal LLM endpoint:
- The internal LLM slice is generated offline during data prep.
- If the generated dataset is published, other users can download it and use it as a local JSONL source without needing an endpoint.
- If no internal LLM data is available, use the fallback 50k mixture:
  - 22,500 public assistant/instruction
  - 7,500 GSM8K train
  - 12,500 local rule-based tool-use
  - 7,500 public tool-calling
  - 0 internal LLM
- If the public tool source cannot supply 7,500 clean normalized examples, shift the deficit to public assistant or local tool data and record the actual mixture in the manifest.

Implementation implications:
- Data prep should produce per-source normalized JSONL files.
- A mixer combines them into one unified SFT JSONL plus manifest.
- The SFT trainer reads only the unified JSONL.
- max_examples subset runs should preserve source ratios as closely as practical.
- Internal LLM endpoint, API key, and model name must come from environment variable names in config; actual secrets must not be committed.
- Add .env to .gitignore if used.

Research context:
- Google Chinchilla is a pretraining paper, not the SFT recipe source.
- Chinchilla-1-73M, SmolLM2/Smol-SmolTalk, TinyLlama chat, and Qwen3 small-model SFT examples support using public assistant/math/tool data plus controlled task-specific data.
- The masterplan is a north star; this locked strategy supersedes the older ~10k synthetic-heavy SFT assumption where they conflict.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document records the locked default 50k mixture and the no-internal-LLM fallback mixture
- [x] #2 Document explains that internal LLM data is generated offline and can be published or replaced by other slices
- [x] #3 Document defines per-source JSONL, unified mixture JSONL, manifest, and max_examples ratio behavior
- [x] #4 Document states that M2 is short-context SFT validation and long-context extension is later
- [x] #5 Document records the selected public tool source, call-only scope, filter rules, and fallback order
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Public assistant source is likely Smol-SmolTalk train. Public tool source should be selected after inspecting one or two clean candidates such as xLAM function-calling, ToolACE-style, or APIGen-style data. GSM8K train is about 7,473 rows, so 7,500 is the natural full slice.

Locked public tool source decision: use argilla/apigen-function-calling as the primary public tool-calling source for M2. It is non-gated, CC-BY-4.0, has 109,402 rows, and is a superset of the gated Salesforce/xlam-function-calling-60k data. The public tool slice is call-only: query + tools + expected tool call. It does not include tool results or final answers. The local 10k tool slice covers tool-result/final-answer behavior, no-call cases, missing-info cases, and unseen-schema behavior.

Public tool filter rules:
- prefer origin=distilabel rows because tools are already JSON Schema
- keep only rows with exactly one expected tool call
- reject nested/dict arguments
- keep flat arguments only: string/int/float/bool/short list
- cap tools per row at 5 or fewer
- enforce length caps suitable for 50M short-context SFT
- deduplicate by hash_id/id/query
- exclude function names that overlap the M2 tool eval set

Fallback order:
1. argilla/apigen-function-calling
2. Team-ACE/ToolACE if full public trajectories with tool results/final answers become required
3. Salesforce/xlam-function-calling-60k only if gated access is acceptable
4. BFCL is eval-only, not a training source
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and filled doc-006 with the accepted M2 SFT data strategy, default 50k mixture, no-internal-LLM fallback, public tool source decision, local generator design, and eval philosophy.
<!-- SECTION:FINAL_SUMMARY:END -->
