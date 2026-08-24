---
id: TASK-005.06
title: Validation run - ~50M tokens on Kestrel-50M (loss down + coherent text)
status: To Do
assignee: []
created_date: '2026-08-24 01:57'
labels: []
milestone: m-1
dependencies:
  - TASK-005.05
  - TASK-005.04
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
parent_task_id: TASK-005
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the ~50M-token validation pretrain on Kestrel-50M using the built stack (TASK-005.05 entry point) and verify it worked: forward-pass loss decreases and generate() (TASK-005.04) yields coherent text. This is the M1 acceptance gate. No new code unless a bug surfaces (then fix it in the relevant subtask).

Steps:
1. Run the validation pretrain: uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain.yaml  (~50M tokens; expected ~30 min at ~30k tok/s, unbenchmarked - could be longer).
2. Load the final checkpoint + tokenizer; run generate() on a few prompts; eyeball coherence.

Quantitative targets (the M1 gate):
- forward-pass loss decreases monotonically (post-warmup) from ~ln(16384) ~= 9.7 to a clearly lower value (target < ~8.5; the authoritative signal is coherent text).
- generate() on a simple prompt (e.g. 'The capital of France is') produces coherent, English-like text (human-verified), not gibberish.
- final checkpoint saved at checkpoints/pretrain/50m/ and reloads via kestrel.model.io.load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 the ~50M-token run completes and the logged loss decreases monotonically (post-warmup) from ~9.7 to below ~8.5
- [ ] #2 generate() from the final checkpoint produces coherent English-like text on a few prompts (human-verified; capture 2-3 sample outputs as evidence in the task notes)
- [ ] #3 the final checkpoint is saved and reloads via kestrel.model.io.load
- [ ] #4 record final loss, tokens/sec, and wall-clock time in the task notes (benchmarks the ~30k tok/s estimate from doc-001 §2)
<!-- AC:END -->
