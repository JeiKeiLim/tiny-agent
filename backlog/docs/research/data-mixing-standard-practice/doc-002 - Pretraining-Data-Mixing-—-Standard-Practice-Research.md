---
id: doc-002
title: Pretraining Data Mixing — Standard Practice Research
type: guide
created_date: '2026-08-25 01:43'
updated_date: '2026-08-25 01:44'
tags:
  - research
  - data
  - pretraining
---

# Pretraining Data Mixing — Standard Practice Research

Research note supporting `TASK-005.02.01`.

## Question

Is it standard LLM pretraining practice to mix corpus domains by target proportion, rather than training on one domain file completely before moving to the next?

## Summary

Yes. Public LLM recipes commonly use an explicit domain mixture and sample or shuffle data so that the training stream approximates the intended proportions. Domain-block training such as:

```text
code -> code -> code -> web -> web -> web -> jsonl -> jsonl
```

is not the usual default. It can be valid as a deliberate curriculum, but the current Kestrel domain-block order is an implementation artifact, not an intended curriculum.

The planned Kestrel fix, a deterministic weighted multi-file scheduler, is much closer to standard practice.

## Public evidence

### LLaMA

LLaMA used explicit sampling proportions across multiple sources:

| source | target proportion |
|---|---:|
| CommonCrawl | 67% |
| C4 | 15% |
| GitHub | 4.5% |
| Wikipedia | 4.5% |
| Books | 4.5% |
| ArXiv | 2.5% |
| StackExchange | 2% |

Most large sources were used approximately once, while smaller high-value sources such as Wikipedia and Books were used multiple times. This shows two standard ideas:

1. choose target domain proportions,
2. upsample smaller high-value domains when appropriate.

Sources:

- https://arxiv.org/abs/2302.13971
- https://chanys.github.io/llama1/
- https://www.together.xyz/blog/redpajama

### SmolLM and SmolLM3

SmolLM training configs expose explicit `dataset_weights` across multiple tokenized datasets, including web, educational synthetic data, math, Python, and StackOverflow.

SmolLM3 uses staged domain mixtures:

| stage | web | code | math |
|---|---:|---:|---:|
| stage 1 | 85% | 12% | 3% |
| stage 2 | 75% | 15% | 10% |
| stage 3 | 63% | 24% | 13% |

This shows that changing the mixture over training is also standard when it is a deliberate staged recipe.

Sources:

- https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm1/config_smollm1_1B.yaml
- https://huggingface.co/blog/smollm3

### SlimPajama-DC

SlimPajama-DC empirically studies how domain combinations and proportions affect LLM pretraining. It finds that domain diversity and mixture composition matter, especially after deduplication.

Source:

- https://arxiv.org/abs/2309.10818

### SampleMix

SampleMix describes the common baseline as domain-wise mixing: choose domain weights first, then sample data according to those weights. It then proposes a more refined sample-wise strategy.

Source:

- https://aclanthology.org/2025.findings-emnlp.741/

## Standard pattern

A typical production pattern is:

```text
choose target domain proportions
-> sample documents or token shards according to those proportions
-> shuffle or interleave the sampled data
-> train
```

Conceptually, the training stream looks like:

```text
[web + code + web + web + jsonl + web + code]
[web + web + web + code + web + web + web]
[web + web + jsonl + web + web + code + web]
...
```

The exact implementation can vary:

- weighted document sampling,
- weighted token-shard sampling,
- global shuffling of tokenized datasets,
- staged mixtures,
- upsampling small domains,
- periodic reshuffling between epochs.

The important property is that the long-run token distribution matches the intended mixture.

## Random sampling versus exact per-step quotas

Seeded random weighted sampling is common. Exact per-step domain quotas are less common and usually not required at large scale because batches contain millions of tokens.

Kestrel M1 uses a much smaller step:

```text
8 sequences x 1024 tokens = 8192 tokens per step
```

Therefore random weighted sampling will have more visible per-step variance. A quota-based scheduler would give tighter control, but weighted sampling is still a reasonable M1 approximation.

## Domain exhaustion

Public recipes handle exhausted domains in different ways:

1. Renormalize the remaining weights.
2. Upsample smaller high-value domains by repeating them.
3. Change the mixture in a deliberate training stage.
4. Stop when all data is exhausted.

For Kestrel M1, renormalization after a domain exhausts is reasonable. Upsampling small domains may be worth revisiting for later full runs.

## Kestrel implications

Current Kestrel behavior:

```text
code.txt -> web.txt -> jsonl.txt
```

This is acceptable for validating that the training pipeline works, but it is not ideal for model quality or representative validation loss.

Planned `TASK-005.02.01` behavior:

```text
weighted interleaved sampling across web, code, and jsonl
```

This is much closer to standard practice.

Follow-up considerations:

- use token-count weights instead of byte-size weights when exact domain proportions matter,
- consider a corpus manifest for authoritative component fractions,
- consider larger effective batch size to reduce mixture variance,
- consider document-aware packing or attention masking so sequences do not blend unrelated documents,
- consider staged mixtures or upsampling for later full-scale runs.

## Bottom line

The planned weighted mixing design is aligned with standard LLM pretraining practice. The current domain-block run should be treated as a pipeline validation artifact, not as the target data strategy.
