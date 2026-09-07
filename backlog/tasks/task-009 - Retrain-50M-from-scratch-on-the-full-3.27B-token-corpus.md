---
id: TASK-009
title: Retrain 50M from scratch on the full 3.27B-token corpus
status: In Progress
assignee: []
created_date: '2026-09-01 05:25'
updated_date: '2026-09-07 08:02'
labels:
  - pretraining
  - 50m
  - experiment
dependencies: []
references:
  - 'https://arxiv.org/abs/2502.02737'
documentation:
  - >-
    backlog/docs/research/pretrain-token-budget/doc-005 -
    50M-Pretrain-Token-Budget-and-3B-Continuation-Research.md
  - >-
    backlog/docs/research/pretrain-benchmarks/doc-012 -
    50M-1B-vs-3B-Pretrain-Benchmark-Result.md
modified_files:
  - >-
    backlog/docs/research/pretrain-benchmarks/doc-012 -
    50M-1B-vs-3B-Pretrain-Benchmark-Result.md
priority: high
ordinal: 58000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retrain Kestrel-50M from random init on the full 12GiB corpus (~3.27B train tokens, ~64.5 tokens/param), replacing the Chinchilla-capped 1B-token run as the 50M pretrain reference.

Why: the 1B run's train and val losses were still decreasing at the end (checkpoints/pretrain/50m/run.jsonl); doc-005 estimates 0.1-0.3 nats val-loss improvement from the full corpus. SmolLM2 (arXiv:2502.02737) shows overtraining small models on high-quality data pays off (their 135M used ~14,800 tokens/param vs our 20). Training the 50M on the full corpus also aligns the 50M vs 150M scaling comparison at a matched token budget (configs/kestrel/150m/pretrain.yaml already uses total_tokens: null on the same corpus).

Why from scratch instead of continuation: the corpus shuffle is deterministic from the seed (fixed at 0 in configs/kestrel/corpus.yaml), so a continuation run from the 1B checkpoint with a fresh iterator would re-process the same first ~1B tokens before reaching new data - same wall time (~124h) with no time saving, plus it would need a new weights-only-init feature and a two-phase LR heuristic. From scratch is one config with one continuous warmup->cosine schedule over the full horizon.

Files:
- New: configs/kestrel/50m/pretrain_3b.yaml - copy of configs/kestrel/50m/pretrain.yaml with total_tokens: null and trainer.output_dir: checkpoints/pretrain/50m-3b. All other fields identical (lr 3e-4, warmup 500, batch 8, seq 1024, num_steps 0) so the data budget is the only variable. The LR horizon auto-derives from the corpus estimated step count (~399k steps).
- No source code changes.

Steps:
1. Baseline: uv run python scripts/eval_pretrain.py --pretrain-config configs/kestrel/50m/pretrain.yaml --checkpoint checkpoints/pretrain/50m/final - record the val loss (comparison target).
2. Launch: nohup uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain_3b.yaml > logs/pretrain-50m-3b.log 2>&1 &
3. Mid-run check at ~1.5B tokens: in-loop val loss should already be below the 1B baseline final val loss; if clearly worse, investigate before continuing.
4. On completion: uv run python scripts/eval_pretrain.py --pretrain-config configs/kestrel/50m/pretrain_3b.yaml --checkpoint checkpoints/pretrain/50m-3b/final - record the delta vs the 1B baseline.

Targets:
- Full val loss at 3.27B tokens at least 0.1 nats lower than the 1B baseline (doc-005 planning range 0.1-0.3; >0.3 bonus, <0.05 falsifies the hypothesis).
- Reproducibility sanity: in-loop val loss at ~1B tokens within +/-0.05 nats of the old 1B run final val loss (same data/config/seed).

Gotchas:
- ETA ~124h (~5 days) at the previous run pace (~26M tokens/h; the 1B run took ~38-40h).
- Do NOT modify configs/kestrel/50m/pretrain.yaml or existing checkpoints/pretrain/50m artifacts - they are the 1B comparison baseline.
- The corpus is already built (data/corpus-12g); build_corpus() is a no-op. Do not perform the planned data/corpus-12g -> data/corpus rename (corpus.yaml comment) while this run is active.
- Checkpoint retention: keep_latest_checkpoints: 5 + best; final written at the end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 configs/kestrel/50m/pretrain_3b.yaml exists, loads via PretrainConfig, and has total_tokens: null
- [ ] #2 Baseline eval of checkpoints/pretrain/50m/final recorded (val loss)
- [ ] #3 Run completes to corpus exhaustion: checkpoints/pretrain/50m-3b/final exists with weights.npz + state.json + run.jsonl showing ~3.27B tokens
- [ ] #4 Final eval recorded with delta vs the 1B baseline (target: >=0.1 nats improvement per doc-005)
- [ ] #5 make check stays green (no code changes expected)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline recorded 2026-09-01 (AC #2 done): eval_pretrain.py on checkpoints/pretrain/50m/final (1B run final) -> val mixed loss 3.183922, ppl 24.14 (106,392 tokens). Per-domain: web 3.260366, code 1.980641, synthetic 2.724424. Target for the 3.27B final: mixed val loss <= 3.0839 (>=0.1 nats improvement).

External benchmark scorecards analyzed 2026-09-07:
- 1B scorecard: data/pretrain_eval/scorecard_50m_1b.json, checkpoint checkpoints/pretrain/50m/final
- 3B scorecard: data/pretrain_eval/scorecard_50m_3b.json, checkpoint checkpoints/pretrain/50m-3b/best

LM deltas, 3B best vs 1B final:
- Pile test BPB 1.3863 -> 1.3302, delta -0.0561
- C4 validation BPB 1.2775 -> 1.2367, delta -0.0408
- WikiText BPB 1.3652 -> 1.3175, delta -0.0477
- LAMBADA BPB 1.5035 -> 1.4706, delta -0.0329
- token-weighted unique LM loss 3.3784 -> 3.2520, delta -0.1264
- token-weighted unique LM BPB 1.3457 -> 1.2953, delta -0.0504

MC deltas, 3B best vs 1B final:
- unweighted acc 33.52 -> 34.07, delta +0.55
- unweighted acc_norm 35.77 -> 35.76, delta -0.01
- PIQA +1.85, HellaSwag +0.44, ARC-Easy +0.76, MMLU +0.38
- WinoGrande -0.95 and OpenBookQA acc_norm -2.6 are within small-set noise

In-loop validation comparison:
- 1B final state best_val_loss 3.1583
- 3B best best_val_loss 3.0213
- delta 0.1370 nats
- versus recorded 1B eval_pretrain baseline 3.1839, delta 0.1626
- this is inside the doc-005 planned 0.1-0.3 nat improvement range

Public anchor interpretation:
- Kestrel-50M at 3.13B tokens is not in the same token budget as public anchors in doc-004, which mostly used 300B-6T tokens.
- The 3B 50M result is broadly in the Pythia-70M neighborhood on PIQA and ARC-Challenge, but still below Pythia-70M on ARC-Easy and WinoGrande.
- GPT-2 124M Pile BPB anchor is 1.2253; Kestrel 3B token-weighted unique LM BPB is 1.2953, about 0.070 BPB higher, with GPT-2 using more params and a much larger token budget.

Caveats:
- wikitext103 is identical to wikitext2 in these scorecards, so treat WikiText as one result.
- LAMBADA is reported as BPB, not final-token accuracy.
- scorecards contain a local data_dir value and should not be committed without redaction.
- the 3B scorecard uses best, not final, because final was degraded by the late code-only training suffix.

Benchmark result documented in doc-012: 50M 1B vs 3B Pretrain Benchmark Result. Interpretation: positive learning-pipeline result, 3B best is the new 50M pretrain reference, public-model comparison is favorable once token budget is considered.
<!-- SECTION:NOTES:END -->
