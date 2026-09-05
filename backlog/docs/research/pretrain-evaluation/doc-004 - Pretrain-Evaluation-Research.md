---
id: doc-004
title: Pretrain Evaluation Research
type: guide
created_date: '2026-08-27 04:29'
updated_date: '2026-09-05 00:59'
tags:
  - research
  - pretraining
  - evaluation
  - benchmarks
  - datasets
  - bpb
  - external-evaluation
  - kestrel
---
# Pretrain Evaluation Research

Research note for Kestrel pretrain evaluation and the planned `eval_pretrain.py` tool.

## Question

How should Kestrel measure the result of a pretraining run?

- Is held-out validation loss enough?
- Should we use external benchmarks?
- What should the M1 50M/150M pretrain scorecard contain?
- Why is the current in-loop validation loss only a monitoring signal?

## Summary

Held-out next-token loss is the primary scientific metric for pretraining. For Kestrel, the current in-loop validation loss is useful for training health and checkpoint selection, but it is not a final evaluation because it evaluates only a small sample of the validation split.

A proper Kestrel pretrain result should report:

1. token-weighted held-out loss over a larger fixed validation set
2. perplexity and bits/token
3. domain-wise loss for web / code / synthetic
4. training diagnostics: tokens seen, steps, wall-clock time, tokens/sec, best/final checkpoint
5. generation samples from fixed prompts
6. optional directional common-sense benchmarks for base models

Public task benchmarks are secondary for 50M/150M base models. They are useful for relative trends, but they should not be the main M1 gate.

## Current Kestrel behavior

The trainer evaluates validation loss in-loop:

- config: `TrainerConfig.eval_every` and `TrainerConfig.eval_iters`
- current 50M/150M pretrain configs: `eval_every: 1000`, `eval_iters: 10`
- with `batch_size: 8` and `seq_len: 1024`, one validation estimate uses roughly:

```text
10 batches * 8 sequences * 1024 tokens ≈ 81,920 tokens
```

The corpus validation split is 1% of the built corpus. For the 12GiB corpus, that is still much larger than 80k tokens. Therefore the current in-loop val loss is a small random-ish sample, not a full validation-set measurement.

Current behavior is good for:

- detecting divergence
- tracking training progress
- selecting the `best` checkpoint

It is not ideal for:

- final run reporting
- precise 50M vs 150M comparison
- domain-wise diagnosis
- reproducible checkpoint evaluation after the run

## Metric definitions

The trainer loss is next-token cross-entropy in nats/token.

Useful derived metrics:

```text
perplexity  = exp(loss)
bits/token  = loss / ln(2)
BPB         = bits per byte, used when comparing different tokenizers
```

For internal Kestrel comparisons using the same tokenizer, perplexity is fine. For comparison against external models or external datasets, BPB is safer because raw perplexity depends on tokenizer vocabulary and tokenization.

## Standard practice

### Primary: held-out language-modeling loss

Most pretraining work uses held-out next-token loss as the main objective-level metric.

Examples:

- Chinchilla / compute-optimal training: https://arxiv.org/abs/2203.15556
- OLMo: https://arxiv.org/abs/2402.00838
- Dolma: https://arxiv.org/abs/2402.00159
- SmolLM pretraining repo: https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md

The important properties are:

- fixed eval data
- no overlap with training data
- same tokenizer
- same batching/document semantics
- enough tokens to make the estimate stable
- token-weighted averaging

### Domain-specific perplexity

Aggregate loss can hide domain problems. A model can improve overall loss while getting worse on code or structured JSONL.

Paloma is a strong reference for this:

- https://arxiv.org/abs/2312.10523
- https://paloma.allen.ai/

Its main lesson: evaluate language-model fit on fine-grained domains, not only one monolithic held-out set.

For Kestrel, the relevant domains are the corpus components:

- `web`
- `code`
- `synthetic`

The corpus builder already writes per-domain validation files:

```text
data/corpus-12g/val/web.jsonl
data/corpus-12g/val/code.jsonl
data/corpus-12g/val/synthetic.jsonl
```

This makes domain-wise evaluation practical.

### Generation samples

People also inspect generated text. For small base models, generation is subjective but still useful.

For Kestrel M1, generation should be judged as:

- English-like vs gibberish
- repetitive vs varied
- plausible local syntax
- 150M vs 50M relative improvement at matched token budgets

Fully coherent long-form text should not be expected at this scale.

### Downstream task benchmarks

Common LLM benchmarks include:

- HellaSwag
- PIQA
- ARC-Easy / ARC-Challenge
- WinoGrande
- BoolQ
- LAMBADA
- MMLU
- GSM8K
- HumanEval / MBPP
- tool-calling / agent benchmarks

The LM Evaluation Harness is the standard tool for many of these:

- https://github.com/EleutherAI/lm-evaluation-harness

For Kestrel-50M/150M base models, task benchmarks should be treated carefully:

- hard benchmarks such as MMLU/GSM8K/HumanEval are likely near chance or very noisy
- many task benchmarks measure abilities that emerge more after SFT
- base-model loglikelihood tasks can still be useful as directional signals
- they are not a substitute for held-out LM loss during pretraining

If added later, start with simple common-sense tasks:

- HellaSwag
- PIQA
- ARC-Easy
- WinoGrande

## External perplexity datasets

Common external held-out sets include:

- WikiText / WikiText-103
- C4 validation
- The Pile validation slices
- PG19
- ArXiv
- GitHub / code text
- LAMBADA
- Penn Treebank

These are useful for external comparison, but they are secondary for Kestrel because:

- they may use different tokenizers
- they may overlap with pretraining sources
- they may not match Kestrel's web/code/synthetic mix
- raw perplexity is not directly comparable across tokenizers

If used, report BPB and note contamination/tokenizer caveats.

## External benchmark dataset sizes and download protocol

Research update: 2026-09-03.

This section records the verified evaluation-split sizes for the external benchmarks under consideration, so that Kestrel can reuse a consistent public evaluation protocol later.

### Recommended external benchmark tracks

For a pretrained base model, use two tracks:

1. **Pretrain compression track**
   - WikiText-2 BPB
   - WikiText-103 BPB
   - C4 validation BPB, full or fixed subsample
   - Pile test BPB, full or fixed subsample
   - LAMBADA final-token accuracy

2. **Public comparability track**
   - HellaSwag
   - PIQA
   - ARC-Easy
   - ARC-Challenge
   - WinoGrande
   - OpenBookQA
   - optionally BoolQ, SciQ, and MMLU

For Kestrel-50M, the public comparability track is useful for comparison with published small-model cards, but it should not replace held-out LM loss as the primary pretraining metric.

### Verified evaluation split sizes

The sizes below are for the standard public evaluation splits, not training splits.

| Benchmark | Eval split | Examples | Format | Approx. size |
|---|---|---:|---|---:|
| HellaSwag | `validation` | 10,042 | 4-choice | ~11 MB |
| PIQA | `validation` | 1,838 | 2-choice | ~0.3 MB |
| ARC-Easy | `test` | 2,376 | 4-choice | ~0.7 MB |
| ARC-Challenge | `test` | 1,172 | 4-choice | ~0.4 MB |
| WinoGrande | `winogrande_xl validation` | 1,267 | 2-choice | ~0.2 MB |
| OpenBookQA | `main test` | 500 | 4-choice | ~0.1 MB |
| MMLU | `test` | 14,042 | 4-choice | ~3.5–7 MB |
| BoolQ | `validation` | 3,270 | yes/no | ~2.1 MB |
| SciQ | `test` | 1,000 | 4-choice | ~0.6 MB |
| LAMBADA | `test` | 5,153 | final-word | ~1.7 MB |
| WikiText-2 | `test` | 4,358 lines / ~62 docs / ~245k words | LM text | ~1.3 MB |
| WikiText-103 | `test` | 4,358 lines / ~62 docs / ~245k words | LM text | ~1.3 MB |
| C4 `en` | `validation` | 364,724 docs | LM text | large, use subsample |
| The Pile | `test` | 214,584 docs | LM text | ~1.3 GB |

Notes:

- HellaSwag, PIQA, WinoGrande, and BoolQ use `validation` as the standard public evaluation split in common harnesses. This does not mean training data is being downloaded; it means the publicly scored eval split is named `validation`.
- C4 has no official test split; `validation` is the standard held-out split.
- The Pile test set is much larger than the multiple-choice sets.
- The cheap benchmark suite, excluding C4 and Pile, is small: roughly **<50 MB total**.
- Adding full Pile test brings the total to roughly **~1.4 GB**.
- Adding full C4 validation and Pile test is roughly **~2–3 GB total**, depending on the exact C4 shard format.

### Download helper

Kestrel has a helper script for downloading these evaluation splits:

```bash
uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path/to/datasets
```

Useful variants:

```bash
uv run python scripts/download_pretrain_eval_datasets.py --list
uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path/to/datasets --dry-run
uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path/to/datasets --skip-large
uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path/to/datasets --only hellaswag,piqa
```

The script downloads the raw data files for the requested evaluation split only. It does not download training splits and does not build the full Hugging Face dataset cache. Each dataset directory contains the raw eval files plus a `manifest.json` with the repo, config, split, files, and split size metadata. The destination path is always provided by the caller and is not hard-coded in the repository.

A matching local evaluator reads those raw files without network access:

```bash
uv run python scripts/run_eval_pretrain_benchmarks.py \
  --pretrain-config configs/kestrel/50m/pretrain.yaml \
  --checkpoint checkpoints/pretrain/50m/best \
  --data-dir /path/to/datasets \
  --max-tokens 100000 \
  --max-examples 200
```

It reports BPB for language-modeling sets and zero-shot multiple-choice accuracy for task sets, and writes a JSON scorecard.

### Evaluation reporting protocol

For any external benchmark result, Kestrel should record:

- benchmark name
- dataset repo / config
- split name
- number of examples evaluated
- number of tokens or bytes evaluated
- whether the full split or a fixed subsample was used
- subsample seed, if applicable
- prompt format / harness version
- metric name:
  - `acc` or `acc_norm` for multiple-choice tasks
  - `bits_per_byte` for language-modeling tasks
  - final-token accuracy for LAMBADA
- decontamination statement:
  - whether the eval set was excluded from training
  - whether n-gram decontamination was performed
  - known overlap caveats

For language-modeling BPB:

```text
BPB = total_nats / (ln(2) * total_bytes)
```

BPB is the preferred cross-tokenizer metric. Token perplexity is useful for internal Kestrel comparisons, but it should not be the headline number when comparing against external models with different tokenizers.

### Subsampling guidance

Full splits are preferred when cheap:

```text
HellaSwag
PIQA
ARC-Easy
ARC-Challenge
WinoGrande
OpenBookQA
BoolQ
SciQ
LAMBADA
WikiText-2
WikiText-103
```

For larger sets, use a documented fixed subsample:

```text
C4 validation: 10k–100k docs or 1M–10M tokens
Pile test:     1/10 of test, or 1M–10M tokens
MMLU:          full test if convenient, otherwise fixed subject subsample
```

The subsample must be seeded and reported. Do not silently evaluate only the first N examples.

### Published reference anchors

The numbers below are useful anchors, not a clean leaderboard. Shot settings, harnesses, tokenizers, and token budgets differ across sources.

| Model | Params | Tokens | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | OBQA | MMLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 124M | 124M | ~40B | — | — | — | — | — | — | — |
| Pythia-70M | 70M | ~300B | — | 59.5 | 38.1 | 18.0 | 52.8 | — | — |
| Pythia-160M | 160M | ~300B | — | 62.7 | 44.9 | 18.6 | 53.1 | — | — |
| OPT-125M | 125M | not cleanly stated | 29.2 | 63.0 | 43.5 | 18.9 | 50.3 | — | 26.0 |
| GPT-Neo-125M | 125M | ~300B | 30.3 | — | — | 23.0 | 51.8 | — | 26.0 |
| SmolLM-135M | 135M | 600B | 41.2 | 68.4 | 42.4 avg | — | 51.3 | 34.0 | 30.2 |
| SmolLM2-135M | 135M | 2T | 42.1 | 68.4 | 43.9 avg | — | 51.3 | 34.6 | 31.5 |
| MobileLLM-125M | 125M | ~1T | 38.9 | 65.3 | 43.9 | 27.1 | 53.1 | 39.5 | — |
| Gemma 3 270M | 270M | ~6T | 40.9 | 67.7 | 57.7 | 29.0 | 52.0 | — | — |
| OLMo-1B | 1B | ~2–3T | 62.5 | 73.7 | 58.1 | 34.5 | 58.9 | 46.4 | — |

Useful GPT-2 / Pile anchors:

- GPT-2 124M WikiText-103 PPL: **37.50**
- GPT-2 124M Pile test BPB: **1.2253**
- GPT-3 175B Pile test BPB: **0.7177**

Interpretation for Kestrel:

- Kestrel-50M at 3.27B tokens is not in the same token budget as most published anchors.
- The published anchors are useful for sanity-checking scale, not for a same-budget comparison.
- A credible Kestrel public table should compare at similar tokens/FLOPs or clearly label the token-budget difference.
- The strongest public framing is a reproducible 50M pretrain → SFT → tool-calling pipeline with transparent eval protocol, not a claim of raw SOTA at 50M.

## Recommended Kestrel pretrain scorecard

For each completed or resumed run, report:

### Primary

- checkpoint path: `best` and/or `final`
- tokens seen
- global step count
- best validation loss
- final validation loss
- perplexity = `exp(val_loss)`
- bits/token
- wall-clock time
- tokens/sec

### Secondary

- mixed validation loss over a larger fixed eval sample
- domain-wise loss:
  - web
  - code
  - synthetic
- train loss vs val loss gap
- LR schedule behavior
- generation samples from fixed prompts

### Optional

- zero-shot loglikelihood common-sense benchmarks:
  - HellaSwag
  - PIQA
  - ARC-Easy
  - WinoGrande
- external LM perplexity sets with BPB normalization

## Recommended eval_pretrain design

A dedicated `eval_pretrain.py` script should evaluate a saved checkpoint after training, without depending on the small in-loop `eval_iters` sample.

Suggested CLI:

```bash
uv run python scripts/eval_pretrain.py \
  --pretrain-config configs/kestrel/50m/pretrain.yaml \
  --checkpoint checkpoints/pretrain/50m/best \
  --max-tokens 1000000
```

Behavior:

1. Load model config, tokenizer path, and corpus config from the pretrain config.
2. Load model weights from the checkpoint directory.
3. Evaluate the validation split using the same seq_len and batch size as training.
4. Compute token-weighted next-token loss.
5. Report:
   - total eval tokens
   - mixed val loss
   - perplexity
   - bits/token
6. Evaluate per-domain validation files when present:
   - `val/web.jsonl`
   - `val/code.jsonl`
   - `val/synthetic.jsonl`
7. Optionally generate fixed samples with `--generate`.

Important properties:

- read-only: never modifies checkpoints or training state
- supports both weights-only and full resumable checkpoint directories
- uses `--max-tokens` to avoid accidental full-corpus evaluation
- uses document-aware batches from `PretrainDataset`
- is deterministic for a given checkpoint, config, seed, and max-token cap

## M1 gate recommendation

For M1, the pretrain result should be accepted based on:

1. training completed or cleanly resumed
2. held-out validation loss decreased from the random-init baseline
3. domain-wise val loss is sane for web/code/synthetic
4. checkpoint reloads successfully
5. generated samples are English-like / plausible, not pure gibberish
6. 50M vs 150M comparison uses the same token budget where possible

Public task benchmarks should not block M1. They can be added later as a separate evaluation track.

## References

- Chinchilla / Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OLMo: https://arxiv.org/abs/2402.00838
- Dolma: https://arxiv.org/abs/2402.00159
- Paloma: https://arxiv.org/abs/2312.10523
- Paloma project page: https://paloma.allen.ai/
- SmolLM pretraining repo: https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md
- LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
- The Pile: https://arxiv.org/abs/2101.00027
- The Pile leaderboard / BPB reference: https://pile.eleuther.ai/
- GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Pythia: https://arxiv.org/abs/2304.01373
- SmolLM2: https://arxiv.org/abs/2502.02737
- MobileLLM: https://arxiv.org/abs/2402.14905
- Gemma 3: https://arxiv.org/abs/2503.19786
- HellaSwag dataset: https://huggingface.co/datasets/Rowan/hellaswag
- PIQA dataset: https://huggingface.co/datasets/baber/piqa
- ARC dataset: https://huggingface.co/datasets/allenai/ai2_arc
- WinoGrande dataset: https://huggingface.co/datasets/allenai/winogrande
- OpenBookQA dataset: https://huggingface.co/datasets/allenai/openbookqa
- MMLU dataset: https://huggingface.co/datasets/cais/mmlu
- BoolQ / SuperGLUE dataset: https://huggingface.co/datasets/aps/super_glue
- SciQ dataset: https://huggingface.co/datasets/allenai/sciq
- LAMBADA OpenAI dataset: https://huggingface.co/datasets/EleutherAI/lambada_openai
- WikiText document-level dataset: https://huggingface.co/datasets/EleutherAI/wikitext_document_level
- C4 dataset: https://huggingface.co/datasets/allenai/c4
- Pile validation/test splits: https://huggingface.co/datasets/EleutherAI/pile_val_test
