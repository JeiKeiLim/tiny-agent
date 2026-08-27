---
id: doc-004
title: Pretrain Evaluation Research
type: guide
created_date: '2026-08-27 04:29'
updated_date: '2026-08-27 04:30'
tags:
  - research
  - pretraining
  - evaluation
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
