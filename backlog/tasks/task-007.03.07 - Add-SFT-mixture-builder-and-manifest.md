---
id: TASK-007.03.07
title: Add SFT mixture builder and manifest
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 23:25'
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
modified_files:
  - src/kestrel/data/sft_mixture.py
  - scripts/run_build_sft_mixture.py
  - tests/data/test_sft_mixture.py
  - src/kestrel/data/sft_prepare.py
  - configs/kestrel/sft_data.yaml
  - README.md
  - AGENTS.md
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
- [x] #1 Mixer builds the default 50k mixture from per-source JSONL files
- [x] #2 Mixer builds the no-internal-LLM fallback mixture
- [x] #3 Manifest records requested counts, actual counts, seeds, and hashes
- [x] #4 Tests cover missing-source deficit handling
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add MixtureRecipe, MixtureConfig, and manifest models in src/kestrel/data/sft_mixture.py.
2. Implement build_mixture() to read per-source JSONL files from mixture.input_dir, select the default recipe, automatically use fallback_recipe when internal_llm is missing/empty, sample rows deterministically, shuffle the combined rows by seed, and write data/sft/mixture/sft-50k.jsonl plus manifest.json.
3. Support deficit_policy allow/fail/redistribute, record requested/actual/deficit/source hashes/config hash/total rows in the manifest.
4. Add scripts/run_build_sft_mixture.py as a thin CLI over SFTDataConfig.mixture.
5. Add mixture config to configs/kestrel/sft_data.yaml with the default 50k recipe and no-internal-LLM fallback recipe.
6. Add tests/data/test_sft_mixture.py covering default mixture, fallback, deficit handling, redistribution, fail policy, determinism, and manifest hashes.
7. Update README.md and AGENTS.md to reflect the new mixture builder command and layout.
8. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented MixtureRecipe/MixtureConfig/MixtureManifest in src/kestrel/data/sft_mixture.py, added build_mixture() with deterministic per-source shuffling, final shuffle, default/no-internal fallback, and allow/fail/redistribute deficit policies. Added scripts/run_build_sft_mixture.py, mixture config in configs/kestrel/sft_data.yaml, tests/data/test_sft_mixture.py, and README/AGENTS docs. make check passes with 302 tests.

Treated recipe count 0 as a hard opt-out: sources with requested_rows=0 are not used as redistribution surplus. make check still passes with 302 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the M2 SFT mixture builder: src/kestrel/data/sft_mixture.py combines per-source raw JSONL files into data/sft/mixture/sft-50k.jsonl with a deterministic shuffle and manifest.json. Added scripts/run_build_sft_mixture.py, default/fallback 50k mixture config in configs/kestrel/sft_data.yaml, deficit allow/fail/redistribute policies, tests/data/test_sft_mixture.py, and updated README/AGENTS. Verified with make check: 302 tests passed.
<!-- SECTION:FINAL_SUMMARY:END -->
