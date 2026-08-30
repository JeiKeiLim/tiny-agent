---
id: TASK-005.06
title: >-
  Validation run - full ~275M-token single pass on Kestrel-50M (loss down +
  English-like text)
status: Done
assignee: []
created_date: '2026-08-24 01:57'
updated_date: '2026-08-30 23:14'
labels: []
milestone: m-1
dependencies:
  - TASK-005.05
  - TASK-005.04
  - TASK-005.07
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
- [x] #1 the full ~275M-token single-pass run completes and the logged loss decreases monotonically (post-warmup) from ~9.7 to well below ~8
- [x] #2 generate() from the final checkpoint produces English-like / simple coherent-ish text (not gibberish) on a few prompts (human-verified; capture 2-3 sample outputs as evidence in the task notes)
- [x] #3 the final checkpoint is saved and reloads via kestrel.model.io.load
- [x] #4 record final loss, tokens/sec, and wall-clock time in the task notes (benchmarks the ~30k tok/s estimate from doc-001 section 2)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-25 attention spike + fix (TASK-005.07): the model was missing a causal mask; Attention now uses causal query-chunked SDPA. Benchmarks (15-layer 50M shape, B=8, float32, one fwd+bwd+AdamW step): T=1024 1.28s/step, peak 12.79GB, ~6.4k tok/s; T=2048 chunk=1024 11.44s/step, peak 33.34GB, ~1.4k tok/s. Full 275M-token single-pass estimate: ~12h at T=1024, ~54h at T=2048. M1 should use seq_len=1024 unless 2048 context is explicitly required.

2026-08-25 user selected seq_len=1024 for M1. Updated configs/kestrel/50m/pretrain.yaml to seq_len=1024 and num_steps=40000; make check green.

2026-08-25 identified in-loop validation bias: PretrainDataset reads corpus domain files sequentially, so the current M1 val loss is not a representative web/code/jsonl mix. Created TASK-005.02.01 to add deterministic weighted multi-file mixing. Do not interpret current/previous M1 val loss as mixed-domain until that task lands.

M1 validation loss caveat: runs using the pre-TASK-005.02.01 PretrainDataset consumed val domain files sequentially, so the first eval_iters batches were biased toward the first val file rather than the intended web/code/jsonl mix. After TASK-005.02.01 lands, validation batches use the same weighted multi-file scheduler and are more representative.

Investigation after 50M full run and 150M partial run: 50M completed 28810 steps with final train loss 1.048 but best val 3.944 at step 25000 and final val 4.588. The late train-loss drop is explained by the current byte-weighted line scheduler: jsonl exhausts early, web exhausts near step 25.8k, and the final ~3k steps are effectively code-only. Generation from 50M step_024000/final and 150M step_022000 is still word salad or repetitive loops. 150M uses batch_size 4, so 40000 steps is only ~164M tokens and stops before corpus exhaustion; it is not a same-token-budget comparison with the 50M run. Conclusion: current outputs are expected for this data budget; the main follow-ups are token-aware domain mixing, a fair 50M/150M token budget, and more pretraining data if coherent generation is the goal.

Root cause update: the weak 50M/150M outputs are not only from undertraining or byte-weighted line mixing. The corpus pipeline flattened HF documents into physical lines. prepare_tokenizer_data.py wrote text + newline, and corpus/builder.py treated physical lines as documents. Current web/code corpus is lossy. Fix is tracked under TASK-005.08: document-level JSONL corpus, manifest, token-aware mixing, doc_ids, document-aware attention, position reset, and auto num_steps.

2026-08-28 M1 validation completed using the 50M 1B-token run on data/corpus-12g. Final checkpoint checkpoints/pretrain/50m/final completed 123,719 steps / 1,013,506,048 tokens. LR schedule reached 0.0. Initial train loss 10.1537, last-100 train mean 3.1463, best/final in-loop val 3.1583. Per-step train loss is noisy rather than strictly monotonic, but the post-warmup trend is clearly downward. External eval on 302,808 held-out val tokens: mixed loss 3.1443, ppl 23.20, bpt 4.536; code loss 1.9249/ppl 6.85, synthetic 2.6949/ppl 14.80, web 3.2953/ppl 26.99. Checkpoint reloads successfully via scripts/check_model.py and scripts/eval_pretrain.py. Approx wall-clock from filesystem metadata: run.jsonl created Aug 26 14:25, final checkpoint Aug 28 13:36 (~47.2h span), average ~5.97k tokens/s including pauses; later checkpoint intervals imply ~7.4k tokens/s. Generation evidence is recorded in TASK-005.14: English-like and code-pattern outputs, with repetition and weak factual reliability expected at 50M/1B.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-08-26 01:36
---
2026-08-26 dataset volume review: the current ~275M-token data/corpus run is best treated as a pipeline smoke run, not as the final 50M/150M pretraining budget. The follow-up expanded corpus plan (~12GiB raw, ~2.97B estimated train tokens, SmolLM-style FineWeb-Edu/code/synthetic mix) is tracked under TASK-005.10.
---

created: 2026-08-26 04:33
---
2026-08-26 status: Keep In Progress. The old ~275M data/corpus run should remain classified as a pipeline smoke run, not the final M1 validation result. The current 50M 12GiB Chinchilla-capped run is the active validation attempt, but it cannot truly resume from current weights-only checkpoints if killed. Close this task only after a run completes or after resume support lands and a resumed run completes, with recorded final loss, validation loss, 2-3 generated samples, tokens/sec, wall-clock time, and checkpoint reload evidence.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M1 pretraining validation is complete. The 50M model completed a 1.0135B-token run, reached LR 0, reduced loss from 10.15 to ~3.15 train / 3.158 val, reloads from checkpoints/pretrain/50m/final, and generates English-like/code-pattern text. Quality is adequate for pipeline validation and SFT-base experiments, not general assistant capability.
<!-- SECTION:FINAL_SUMMARY:END -->
