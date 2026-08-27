---
id: doc-005
title: 50M Pretrain Token Budget and 3B Continuation Research
type: guide
created_date: '2026-08-27 23:53'
updated_date: '2026-08-27 23:54'
tags:
  - research
  - pretraining
  - scaling-laws
  - token-budget
  - 50m
---
# 50M Pretrain Token Budget and 3B Continuation Research

Research note recording whether Kestrel-50M should keep training beyond the current ~1B Chinchilla-capped pretrain budget toward the full built ~3.27B-token corpus.

## Question

- Will training Kestrel-50M on the full ~3.27B-token corpus meaningfully reduce validation loss compared with stopping near 1B tokens?
- Is the extra compute worthwhile given the observed training time?
- Should the next step be a full 3B continuation, a smaller pilot, or stopping at 1B and moving to the next pipeline stage?

## Short answer

Training beyond 1B tokens will probably lower Kestrel-50M validation loss a bit more, but with strongly diminishing returns. Based on the current `run.jsonl` curve and published scaling-law evidence, a reasonable planning range for full-corpus training is roughly **0.1–0.3 nats lower validation loss**, with better results possible but uncertain.

The exact loss cannot be known without training. The research supports the direction and the diminishing-return expectation, but not a precise Kestrel-specific loss value.

Because the current 1B run takes about **40 hours** on the current hardware, full 3B training is a large compute commitment: roughly **120 hours from scratch**, or roughly **80–90 additional hours** if continuing from a completed 1B checkpoint. A smaller continuation pilot is the preferred way to turn this from an estimate into project-specific evidence.

## Current run context

Recorded while the 50M validation run was still in progress.

```text
model params:                  50,675,200
config:                        configs/kestrel/50m/pretrain.yaml
configured total_tokens:       1,013,504,000
tokens per step:               8,192  (batch_size=8, seq_len=1024)
observed step:                 108,210
observed tokens seen:          ~886,456,320
progress toward 1B target:     ~87.5%
latest in-loop val loss:       3.1695 at step 108,000
latest observed LR:            ~1.16e-05
```

The current 50M config is intentionally Chinchilla-capped at about 20 tokens/param:

```text
current 1B target:   1,013,504,000 / 50,675,200 ≈ 20.0 tokens/param
full train corpus:   3,269,394,373 / 50,675,200 ≈ 64.5 tokens/param
```

The built train corpus manifest reports:

```text
data/corpus-12g/train/manifest.json
total_estimated_token_count: 3,269,394,373

web:        2,778,852,428
code:          327,112,074
synthetic:     163,429,870
```

This matters because the extra tokens are not simply the same 1B tokens repeated three times. The corpus builder already produced a larger document-level train corpus. Training to 3.27B would mostly consume additional unique documents, assuming the dataset iterator does not start recycling documents early.

## Observed validation-loss trend

From `checkpoints/pretrain/50m/run.jsonl`:

```text
step      tokens      val_loss
10,000    ~81.9M      3.963
20,000    ~163.8M     3.701
30,000    ~245.8M     3.580
40,000    ~327.7M     3.513
50,000    ~409.6M     3.444
60,000    ~491.5M     3.401
70,000    ~573.4M     3.340
80,000    ~655.4M     3.294
90,000    ~737.3M     3.245
100,000   ~819.2M     3.197
108,000   ~884.7M     3.170
```

The loss is still decreasing, but the improvement per 10k steps has slowed substantially. This is the expected shape of a language-modeling loss curve: large early gains, then smaller refinements.

## Empirical extrapolation from the current run

A simple scaling-style fit was applied to the in-loop validation points after step 10,000:

```text
loss(T) ≈ L_inf + B * T^-beta
```

Extrapolating to the full train corpus size:

```text
beta    predicted loss @3.27B    delta from current
0.05                  2.80          -0.37
0.10                  2.84          -0.33
0.20                  2.91          -0.26
0.30                  2.97          -0.21
0.50                  3.06          -0.11
```

This is not a research-paper prediction. It is an empirical extrapolation from Kestrel's own in-loop validation curve. The in-loop validation uses only `eval_iters=10` batches, so it is noisier than a full `eval_pretrain.py` measurement.

Planning interpretation:

```text
conservative:   0.1–0.2 nats lower
reasonable:     0.2–0.3 nats lower
optimistic:     0.3+ nats lower
```

A full-corpus 50M validation loss around **2.9–3.1** is plausible, but should be treated as an estimate, not an expectation.

## Research evidence

### Chinchilla: compute-optimal training

Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.

- Trained over 400 models from 70M to 16B+ parameters on 5–500B tokens.
- Found that compute-optimal training scales model size and token count roughly equally.
- Common summary: about **20 tokens/param**.

Relevance:

- Kestrel-50M at 1B tokens is already near the Chinchilla-optimal ratio.
- Training to 3.27B is about **64.5 tokens/param**, well beyond the compute-optimal point.
- This does **not** mean extra tokens cannot reduce loss. It means the marginal return per token is expected to be lower, and some of that compute might have produced a better result if used for a larger model.

### Data-constrained scaling

Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.

- Studied scaling when data is limited, with runs up to 9B params and 900B training tokens.
- Found that repeated data can be useful for a while, but the value of extra compute eventually decays as repetition and excess parameters increase.
- Proposed scaling laws that account for finite data and decreasing value of repeated tokens.

Relevance:

- Kestrel's extra tokens are mostly unique built-corpus documents, not simple repetition.
- Therefore the extra data should still have value.
- However, the model is small and the corpus is finite, so the curve should flatten as the model approaches what it can usefully compress from this data.

### SmolLM2: overtrained small model with high-quality data

Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, arXiv:2502.02737.

- 1.7B model trained on about 11T tokens using multi-stage, data-centric training.
- Outperformed comparable small models.

Relevance:

- Strong evidence that overtraining a small model on high-quality data can be beneficial.
- Not directly transferable because SmolLM2 is 1.7B, uses a large curated multi-stage mixture, and is optimized for downstream quality, not just 50M pretrain validation loss.

### LLaMA: smaller model trained on much more data

Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, arXiv:2302.13971.

- 7B and 13B models trained on ~1T tokens.
- These are far above Chinchilla-optimal token counts by later standards, yet produced strong models.

Relevance:

- Supports the idea that more-than-20 tokens/param can be useful when the data is good and the goal is the strongest possible model at that parameter size.

### Small-model token/parameter study

Balaji, *An Empirical Study of Compute-Efficient Token–Parameter Scaling in Small Language Models (1M–20M)*, Research Square preprint.

- Studied 1.6M–18.4M models on ~93M-token datasets.
- Found around **10 tokens/param** to be the most compute-efficient point in that specific setup.
- Higher ratios showed stronger capacity-bottleneck behavior.

Relevance:

- Cautionary evidence that very small models can hit data/capacity limits earlier.
- Not directly transferable to 50M, but it warns against assuming that extra tokens will always produce Chinchilla-like gains.

## What is known vs unknown

Known or well supported:

- More unique training tokens will probably reduce validation loss further.
- The improvement will be sublinear and much smaller per token than early training.
- 3.27B tokens is a valid “strongest 50M base” experiment because the corpus is already built and mostly unique.
- Overtraining beyond 20 tokens/param is not automatically wasted, especially with good data.
- The current LR schedule is tied to the 1B horizon, so a continuation run needs a fresh schedule.

Unknown without training:

- Exact Kestrel-50M validation loss at 3.27B tokens.
- Whether the loss improvement translates into visibly better generation.
- Whether the model will start repeating or memorizing as it approaches the end of the corpus.
- The best continuation peak LR and schedule.
- Whether the compute is better spent on 150M, SFT, RL, or data quality.

## Compute tradeoff

Rough operator estimate from the current run:

```text
1B tokens:                 ~40 hours
full 3.27B from scratch:   ~120–130 hours
continuation from 1B:      ~80–90 additional hours
+500M token pilot:         ~20 hours
```

These are approximate and depend on hardware, checkpointing, evaluation overhead, and data loading.

Decision framing:

1. **Stop at 1B**
   - Keeps the 50M run Chinchilla-aligned.
   - Saves ~80–90 hours.
   - Better if the goal is pipeline validation and 50M vs 150M comparison.

2. **Full 3B continuation**
   - Likely produces the strongest 50M pretrain base.
   - Plausible validation-loss improvement around 0.1–0.3 nats, uncertain.
   - Large time cost and risk of diminishing returns.

3. **+500M continuation pilot**
   - Cheapest way to measure the actual marginal return.
   - Turns the scaling-law estimate into Kestrel-specific evidence.
   - Preferred next step if the goal is to decide rationally.

## Recommended pilot

If we want to test whether full 3B training is worth it:

1. Let the current 1B run finish.
2. Run `scripts/eval_pretrain.py` on the final checkpoint for a stable full-validation measurement.
3. Start a continuation run from the final 1B weights for **+500M tokens**.
4. Use a fresh LR schedule, not the original cosine horizon.
   - Suggested starting range: peak LR `3e-5` to `1e-4`, short warmup, cosine decay to zero.
   - The exact LR should be chosen conservatively because the model is already trained.
5. Monitor:
   - in-loop val loss
   - full `eval_pretrain.py` loss at the end
   - generation quality at `temp=0.0` and `temp=0.1–0.2`
   - repetition / memorization behavior
6. Go/no-go heuristic:
   - `>=0.05` val-loss improvement over 500M tokens: full 3B continuation is likely worthwhile.
   - `0.02–0.05`: borderline; decide based on generation quality and project goals.
   - `<0.02`: probably not worth full 3B; stop and move to the next stage.

## Implementation caveat

The current full-checkpoint resume flow validates checkpoint state against the current config. Therefore “just increase `total_tokens` and resume” is not necessarily a valid continuation path, because the LR horizon and config fingerprint would change. A proper continuation experiment may require either:

- weights-only initialization from the 1B final checkpoint, or
- explicit continuation support in the pretrain entry point/trainer.

This should be scoped as a separate task before running the pilot.

## Conclusion

The intuition that “more data may meaningfully lower loss” is research-plausible. The current run has not plateaued, and the extra corpus tokens are mostly unique. However, the 50M model is already at the Chinchilla-style 20 tokens/param point, so the extra gain should be treated as a continuation/overtraining experiment, not as a continuation of the early fast loss drop.

The best historical record position is:

> Full 3B training is likely to improve Kestrel-50M pretrain loss by a meaningful but diminishing amount, plausibly around 0.1–0.3 nats based on the current curve. Because the compute cost is large, the rational next step is a +500M-token continuation pilot before committing to the full corpus.

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, https://arxiv.org/abs/2203.15556
- Muennighoff et al., *Scaling Data-Constrained Language Models*, https://arxiv.org/abs/2305.16264
- Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, https://arxiv.org/abs/2502.02737
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, https://arxiv.org/abs/2302.13971
- Balaji, *An Empirical Study of Compute-Efficient Token–Parameter Scaling in Small Language Models (1M–20M)*, https://doi.org/10.21203/rs.3.rs-8948933/v1
- Local run data: `checkpoints/pretrain/50m/run.jsonl`
- Local corpus manifest: `data/corpus-12g/train/manifest.json`
- Local config: `configs/kestrel/50m/pretrain.yaml`
