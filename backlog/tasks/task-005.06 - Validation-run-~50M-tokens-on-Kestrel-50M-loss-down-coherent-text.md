---
id: TASK-005.06
title: >-
  Validation run - full ~275M-token single pass on Kestrel-50M (loss down +
  English-like text)
status: To Do
assignee: []
created_date: '2026-08-24 01:57'
updated_date: '2026-08-24 08:11'
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
Run the FULL ~275M-token single-pass validation pretrain on Kestrel-50M using the built stack (TASK-005.05 entry point) and verify it worked: forward-pass loss decreases and generate() (TASK-005.04) yields English-like / simple coherent-ish text. This is the M1 acceptance gate. No new code unless a bug surfaces (then fix it in the relevant subtask).

PLAN UPDATE (2026-08-24): the validation run is a FULL single pass over the ~275M-token corpus we have (supersedes the earlier ~50M target). Rationale: modern LLM pretraining is single-pass (Chinchilla ~20 tokens/param; LLaMA "each token used only once"); ~50M tokens = 1 token/param is too undertrained to show real text. Even the full ~275M = 5.5 tokens/param is still undertrained (Chinchilla wants 20), so the text bar is "English-like / simple coherent-ish, not gibberish" - NOT fully coherent. Fully coherent text needs ~1B tokens (deferred data-gathering, separate milestone).

Steps:
1. Run the validation pretrain: uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain.yaml  (full ~275M tokens, single pass; expected ~2.5h at ~30k tok/s, unbenchmarked - could be longer).
2. Load the final checkpoint + tokenizer; run generate() on a few prompts; eyeball coherence.

Quantitative targets (the M1 gate):
- forward-pass loss decreases monotonically (post-warmup) from ~ln(16384) ~= 9.7 to a clearly lower value (well below ~8; the authoritative signal is the generated text).
- generate() on a simple prompt (e.g. 'The capital of France is') produces English-like / simple coherent-ish text (human-verified), not gibberish. (Fully coherent is out of reach at 5.5 tokens/param.)
- final checkpoint saved at checkpoints/pretrain/50m/ and reloads via kestrel.model.io.load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 the full ~275M-token single-pass run completes and the logged loss decreases monotonically (post-warmup) from ~9.7 to well below ~8
- [ ] #2 generate() from the final checkpoint produces English-like / simple coherent-ish text (not gibberish) on a few prompts (human-verified; capture 2-3 sample outputs as evidence in the task notes)
- [ ] #3 the final checkpoint is saved and reloads via kestrel.model.io.load
- [ ] #4 record final loss, tokens/sec, and wall-clock time in the task notes (benchmarks the ~30k tok/s estimate from doc-001 section 2)
<!-- AC:END -->
