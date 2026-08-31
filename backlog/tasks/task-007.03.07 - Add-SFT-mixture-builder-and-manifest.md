---
id: TASK-007.03.07
title: Add SFT mixture builder and manifest
status: To Do
assignee: []
created_date: '2026-08-31 01:20'
labels:
  - sft
  - data
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.03
  - TASK-007.03.04
  - TASK-007.03.05
  - TASK-007.03.06
parent_task_id: TASK-007.03
priority: high
ordinal: 47000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the mixer that combines per-source SFT JSONL files into the final M2 mixture.

Depends on:
- TASK-007.03.03
- TASK-007.03.04
- TASK-007.03.05
- TASK-007.03.06

Files:
- src/kestrel/data/sft_mixture.py
- scripts/run_build_sft_mixture.py if a CLI entry point is needed
- tests/data/test_sft_mixture.py

Scope:
- Read per-source JSONL files:
  - assistant_public.jsonl
  - gsm8k_math.jsonl
  - tool_local.jsonl
  - tool_public.jsonl
  - internal_llm.jsonl
- Support default 50k mixture:
  - 22,500 assistant_public
  - 7,500 gsm8k_math
  - 10,000 tool_local
  - 5,000 tool_public
  - 5,000 internal_llm
- Support no-internal-LLM fallback 50k mixture:
  - 22,500 assistant_public
  - 7,500 gsm8k_math
  - 12,500 tool_local
  - 7,500 tool_public
  - 0 internal_llm
- If a source has fewer rows than requested, record the deficit and optionally redistribute according to config policy.
- Shuffle deterministically by seed.
- Write:
  - data/sft/mixture/sft-50k.jsonl
  - data/sft/mixture/manifest.json
- Manifest records requested counts, actual counts, source hashes, seed, config hash, and total rows.

Acceptance:
- Default mixture total is 50,000 when all sources have enough rows.
- Fallback mixture total is 50,000 without internal_llm.
- Manifest records actual source counts and hashes.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mixer builds the default 50k mixture from per-source JSONL files
- [ ] #2 Mixer builds the no-internal-LLM fallback mixture
- [ ] #3 Manifest records requested counts, actual counts, seeds, and hashes
- [ ] #4 Tests cover missing-source deficit handling
- [ ] #5 make check passes
<!-- AC:END -->
