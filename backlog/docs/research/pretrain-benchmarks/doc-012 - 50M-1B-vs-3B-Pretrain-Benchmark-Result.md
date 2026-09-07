---
id: doc-012
title: 50M 1B vs 3B Pretrain Benchmark Result
type: guide
created_date: '2026-09-07 08:01'
updated_date: '2026-09-07 08:02'
tags:
  - pretraining
  - evaluation
  - benchmarks
  - 50m
  - results
  - public-comparison
---
# 50M 1B vs 3B Pretrain Benchmark Result

Result note for the Kestrel-50M 1B-token baseline and full-corpus 3B-token run. This documents the external benchmark scorecards, the comparison with public small-model anchors from doc-004, and the project-level interpretation.

## Status

- 1B scorecard: `data/pretrain_eval/scorecard_50m_1b.json`
- 3B scorecard: `data/pretrain_eval/scorecard_50m_3b.json`
- 1B checkpoint: `checkpoints/pretrain/50m/final`
- 3B checkpoint: `checkpoints/pretrain/50m-3b/best`
- Related task: TASK-009
- Related run analysis: TASK-013
- Related research: doc-004 and doc-005

## Headline interpretation

The 3B run is a positive result. Compared with the 1B baseline, the 3B best checkpoint shows:

- clear improvement on external language-modeling BPB
- clear improvement in held-out validation loss
- small but mostly positive movement on zero-shot multiple-choice benchmarks
- behavior that is broadly consistent with public small-model anchors once token budget is considered

The result is meaningful because Kestrel used a small, transparent, locally built corpus and a simple single-pass pretrain recipe. The public anchors usually used far more tokens, more parameters, stronger curation, distillation, or multi-stage training.

For the project goal, this is a good sign: the 50M pipeline is producing a usable pretrain base, and the 1B to 3B scaling behavior is sane.

## Checkpoint selection

The 3B scorecard uses `best`, not `final`.

This is intentional. TASK-013 found that the 3B run entered a code-only training suffix after the web source exhausted. The final checkpoint therefore collapsed toward Python generation. The best checkpoint is step 368000 and is the correct 3B reference.

Use:

```text
checkpoints/pretrain/50m-3b/best
```

Do not use:

```text
checkpoints/pretrain/50m-3b/final
```

for downstream SFT or evaluation unless studying the failure mode.

## Language-modeling benchmarks

```text
Benchmark        1B BPB   3B BPB   Delta BPB   1B PPL   3B PPL
WikiText         1.3652   1.3175   -0.0477     46.30    40.50
C4 validation    1.2775   1.2367   -0.0408     37.88    33.73
Pile test        1.3863   1.3302   -0.0561     25.81    22.63
LAMBADA          1.5035   1.4706   -0.0329     45.81    42.13
```

Token-weighted over the unique LM sets:

```text
loss:   3.3784 -> 3.2520   delta -0.1264 nats
BPB:    1.3457 -> 1.2953   delta -0.0504
```

The LM result is the strongest evidence. The 3B model compresses all external LM sets better than the 1B model.

Note: `wikitext103` is identical to `wikitext2` in these scorecards, so WikiText is counted once.

## Multiple-choice benchmarks

```text
Benchmark        1B acc   3B acc   Delta acc   Delta acc_norm
HellaSwag        26.57    27.01    +0.44       +0.76
PIQA             58.00    59.85    +1.85       +0.98
ARC-Easy         36.49    37.25    +0.76       +0.59
ARC-Challenge    19.80    19.97    +0.17       +0.26
WinoGrande       51.62    50.67    -0.95       -0.32
OpenBookQA       13.80    16.40    +2.60       -2.60
BoolQ            37.98    38.20    +0.21       +0.21
SciQ             32.90    32.40    -0.50       -0.30
MMLU             24.51    24.88    +0.38       +0.36
```

Unweighted averages:

```text
acc:       33.52 -> 34.07   delta +0.55
acc_norm:  35.77 -> 35.76   delta -0.01
```

Interpretation:

- PIQA improved the most.
- HellaSwag, ARC-Easy, and MMLU improved slightly.
- WinoGrande and OpenBookQA moved in opposite directions between `acc` and `acc_norm`, and those sets are small enough that this is mostly noise.
- Overall, MC ability improved only modestly. That is expected at 50M scale: LM compression improves more cleanly than zero-shot task accuracy.

## In-loop validation comparison

```text
1B final state best val loss:    3.1583
3B best val loss:                3.0213
delta:                           0.1370 nats
```

Versus the recorded `eval_pretrain.py` 1B baseline:

```text
1B baseline:   3.1839
3B best:       3.0213
delta:         0.1626 nats
```

This is inside the doc-005 planned improvement range of 0.1 to 0.3 nats.

## Public model comparison

The public anchors below come from doc-004. They are not a clean same-budget leaderboard. Tokenizers, harnesses, prompt formats, data recipes, and token budgets differ.

```text
Model              Tokens    PIQA   ARC-E   ARC-C   Wino   HellaSwag   MMLU
Kestrel-50M        ~1B      58.0   36.5    19.8    51.6   26.6        24.5
Kestrel-50M        ~3B      59.8   37.2    20.0    50.7   27.0        24.9
Pythia-70M         ~300B    59.5   38.1    18.0    52.8   -           -
Pythia-160M        ~300B    62.7   44.9    18.6    53.1   -           -
OPT-125M           n/a      63.0   43.5    18.9    50.3   29.2        26.0
GPT-Neo-125M       ~300B    -      -       23.0    51.8   30.3        26.0
SmolLM2-135M       ~2T      68.4   43.9    -       51.3   42.1        31.5
MobileLLM-125M     ~1T      65.3   43.9    27.1    53.1   38.9        -
Gemma-3-270M       ~6T      67.7   57.7    29.0    52.0   40.9        -
```

Reading:

- Kestrel-50M at ~3B tokens is broadly in the Pythia-70M neighborhood on PIQA and ARC-Challenge.
- It is still below Pythia-70M on ARC-Easy and WinoGrande.
- The gap to 125M to 270M public models is large, but those models use more parameters, much more training data, and often stronger data curation or distillation.
- For BPB, the useful anchor is GPT-2 124M Pile BPB 1.2253. Kestrel-50M at ~3B is 1.2953, about 0.070 BPB higher. That is reasonable given the much smaller token budget.

## Why this is a positive learning result

- Kestrel-50M used only ~3B tokens, while many public anchors used 300B to 6T tokens.
- The corpus is a simple transparent mix: web 0.85, code 0.10, synthetic 0.05.
- The run used a simple single-pass recipe without distillation, multi-stage mixing, or task-specific pretraining data.
- The 3B model improved LM compression by the expected amount.
- The 3B model did not regress on the broad MC trend.
- The result validates the current pretrain pipeline as a learning platform.
- The 50M model is now a reasonable base for SFT, tool-calling, and agent experiments.

The strongest framing is not raw SOTA at 50M. The strongest framing is that Kestrel now has a reproducible, transparent, end-to-end pretrain result with sane scaling behavior and public-anchored interpretation.

## Caveats

- The scorecards contain a local `data_dir` value. They should not be committed without redacting that path.
- `wikitext103` duplicated `wikitext2` in these scorecards.
- LAMBADA is reported as BPB, not final-token accuracy.
- Public numbers are anchors, not a controlled same-tokenizer or same-harness comparison.
- The 3B best checkpoint saw ~3.01B emitted tokens, while the full 3B run final reached ~3.128B emitted tokens before the late code-only suffix.
- The 1B scorecard uses the 1B `final` checkpoint, not a separately selected 1B `best` checkpoint.

## Decision

Adopt:

```text
checkpoints/pretrain/50m-3b/best
```

as the current Kestrel-50M pretrain reference for downstream SFT and agent work.

Keep the 3B `final` checkpoint only as a failure artifact for the late code-only training suffix. Track the dataset scheduler / source-exhaustion fix as separate follow-up work.
