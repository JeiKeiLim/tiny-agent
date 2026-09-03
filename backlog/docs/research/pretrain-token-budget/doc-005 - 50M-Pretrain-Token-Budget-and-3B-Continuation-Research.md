---
id: doc-005
title: 50M Pretrain Token Budget and 3B Continuation Research
type: guide
created_date: '2026-08-27 23:53'
updated_date: '2026-09-02 08:33'
tags:
  - research
  - pretraining
  - scaling-laws
  - token-budget
  - 50m
  - chinchilla
  - modern-small-models
  - inference-optimal
  - test-time-scaling
  - data-constrained
---
# 50M Pretrain Token Budget and 3B Continuation Research

Research note recording whether Kestrel-50M should keep training beyond the current ~1B Chinchilla-capped pretrain budget toward the full built ~3.27B-token corpus, and how modern small-model evidence changes that decision.

## Question

- Will training Kestrel-50M on the full ~3.27B-token corpus meaningfully reduce validation loss compared with stopping near 1B tokens?
- Is the extra compute worthwhile given the observed training time?
- Should the next step be a full 3B continuation, a smaller pilot, or stopping at 1B and moving to the next pipeline stage?
- Updated 2026-09-02: how should Kestrel interpret Chinchilla after reviewing modern small-model training ratios, inference-optimal scaling, test-time scaling, and data-constrained scaling?

## Short answer

Training beyond 1B tokens will probably lower Kestrel-50M validation loss a bit more, but with strongly diminishing returns. Based on the current `run.jsonl` curve and published scaling-law evidence, a reasonable planning range for full-corpus training is roughly **0.1–0.3 nats lower validation loss**, with better results possible but uncertain.

The exact loss cannot be known without training. The research supports the direction and the diminishing-return expectation, but not a precise Kestrel-specific loss value.

Updated 2026-09-02: Chinchilla’s ~20 tokens/param should be treated as a **historical lower-bound reference**, not as the target token budget for Kestrel. Modern sub-1B models are commonly trained at thousands to tens of thousands of tokens/param. Kestrel-50M at 3.27B tokens is only ~64.5 tokens/param, so it is still far below the regime used by many modern small models. The 3.27B full-corpus run is therefore best interpreted as the **first serious full-corpus epoch**, not as the final answer to the token-budget question.

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

## 2026-09-02 comprehensive research update

This section records a broader research pass triggered by the question: “If Chinchilla is outdated or architecture-specific, how do we infer the proper pretrain token size for Kestrel-50M?”

### Chinchilla is a historical baseline, not a Kestrel design constraint

Chinchilla used the Gopher architecture. The Chinchilla paper states that it uses the same model architecture and training setup as Gopher, with only limited differences such as AdamW, tokenizer normalization, and data re-weighting.

Important architecture comparison:

| Feature | Chinchilla / Gopher | Kestrel / modern LLaMA-style |
|---|---|---|
| Positional encoding | Relative positional encoding from Dai et al. 2019, not RoPE | RoPE |
| Normalization | RMSNorm | RMSNorm |
| FFN | 4× `d_model`; activation not explicitly stated in Gopher paper, GELU implied by GPT-2 lineage | SwiGLU |
| Attention | Standard multi-head attention; no GQA/MQA documented | GQA |
| Tokenizer | SentencePiece BPE, 32k vocab, byte backoff | byte-level BPE |
| Objective | training-compute-optimal pretrain loss | training + downstream SFT/tool/agent behavior |
| Scale | 70M–16B+ fit; 50M is below the fitted range | 50M/150M small-model regime |

This matters because the familiar “~20 tokens/param” number is not a universal constant. It is an empirical fit for a particular 2022 dense-Transformer setup, tokenizer, corpus, and objective.

Corrected interpretation:

```text
Chinchilla is still useful as a historical sanity check.
It is not the right primary anchor for Kestrel-50M.
The modern direction for small deployed models is usually more tokens per parameter, not fewer.
```

### Modern small-model token budgets

Verified public small-model reference points:

| Model | Params | Pretrain tokens | tokens/param | Notes |
|---|---:|---:|---:|---|
| SmolLM 135M | 135M | 600B | ~4,400 | GQA, RoPE, SwiGLU; SmolLM-Corpus ~252B, so multi-epoch |
| SmolLM 360M | 360M | 600B | ~1,700 | same corpus |
| SmolLM 1.7B | 1.7B | 1T | ~590 | same corpus |
| SmolLM2 135M | 135M | 2T | ~14,800 | data-centric small model |
| SmolLM2 360M | 360M | 4T | ~11,100 | data-centric small model |
| SmolLM2 1.7B | 1.7B | 11T | ~6,500 | data-centric small model |
| MobileLLM 125M | 125M | 1T | ~8,000 | GQA, RoPE, SwiGLU |
| MobileLLM 350M | 345M | 1T | ~2,900 | GQA, RoPE, SwiGLU |
| MobileLLM 1.5B | 1.5B | 1T | ~660 | GQA, RoPE, SwiGLU |
| Qwen2 0.5B | 494M | 12T | ~24,300 | GQA, RoPE, SwiGLU |
| Qwen3 0.6B | 596M | 36T | ~60,000 | GQA, RoPE, SwiGLU |
| Qwen3 1.7B | 1.7B | 36T | ~21,200 | GQA, RoPE, SwiGLU |
| Llama 3.2 1B | 1.23B | up to 9T | ~7,300 | GQA, RoPE, SwiGLU; uses logit distillation |
| Llama 3.2 3B | 3.21B | up to 9T | ~2,800 | GQA, RoPE, SwiGLU; uses logit distillation |
| Gemma 3 1B | ~1B | ~2T | ~2,000 | GQA, RoPE, SwiGLU |
| Gemma 3 270M | 270M | ~6T | ~22,000 | closest public size above 50M |
| Phi-1.5 | 1.3B | 150B | ~115 | multi-epoch over small high-quality set; not directly comparable |
| Kestrel-50M current full corpus | 50.7M | 3.27B | ~64.5 | first full-corpus epoch |

Caveats:

- Token counts are tokenizer-dependent. Kestrel uses byte-level BPE, while many public models use 32k/128k BPE tokenizers.
- Some public models use distillation, synthetic data, multi-stage mixing, or repeated epochs.
- “Corpus size” and “training tokens” are not always the same thing.
- The exact token ratio is less important than the pattern: modern sub-1B models are routinely trained far beyond Chinchilla-style ratios.

### Inference-optimal scaling

Sardana & Frankle, *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448.

Key finding:

- When inference cost is included, the optimum shifts toward **smaller models trained on more tokens**.
- Their experiments found quality continuing to improve at high tokens/param, but with slower returns than naive Chinchilla extrapolation predicts.

Relevance to Kestrel:

- Kestrel is not trying to minimize pretraining FLOPs only.
- The end goal is a small model that can be served, SFT-tuned, and used in agent/tool loops.
- That objective favors a more overtrained small base model, not a strictly Chinchilla-capped one.

### Test-time scaling

Roberts et al., *Test-Time Scaling Makes Overtraining Compute-Optimal*, arXiv:2604.01411.

Key finding:

- When the total budget includes training plus test-time sampling/inference, the compute-optimal point moves strongly toward smaller, more overtrained models.
- The study covers small models including the 5M–901M range, which is directly relevant to 50M.

Relevance to Kestrel:

- Agent/tool use may involve retries, repeated sampling, verification, or multi-step inference.
- A cheaper, more overtrained 50M model can be more useful than a Chinchilla-capped model if the pipeline spends meaningful inference compute.

### Overtraining is predictable and downstream-relevant

Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, arXiv:2403.08540.

Key findings:

- Overtrained models follow predictable power-law trends.
- Downstream task error tracks perplexity reasonably well in aggregate.
- Modern LLaMA models were already far beyond Chinchilla ratios.

Relevance to Kestrel:

- We should not decide token budget from pretrain validation loss alone.
- The SFT/tool-call eval bundle is the right tie-breaker.

### Data-constrained and repeated-data caveats

Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.

Key findings:

- Repeated data can remain useful for a while, but its value decays.
- Held-out validation loss is much more reliable than training loss in multi-epoch settings.
- Very heavy repetition can cause degradation or double-descent behavior.

Olsson et al., *Scaling Laws and Interpretability of Learning from Repeated Data*, arXiv:2207.07251.

Key finding:

- Heavy repetition of small data subsets can disproportionately damage generalization through memorization.

Relevance to Kestrel:

- Going beyond one pass over the 3.27B corpus is plausible, but should be done with:
  - deduplication,
  - disjoint eval/SFT data,
  - held-out validation monitoring,
  - memorization/repetition spot checks.

## Updated Kestrel interpretation

### 1B tokens

```text
1B tokens ≈ 20 tokens/param
```

This is the Chinchilla-style reference point. It is useful as a sanity baseline, but modern small-model evidence suggests it is likely undertrained for a strong 50M base.

### 3.27B tokens

```text
3.27B tokens ≈ 64.5 tokens/param
```

This is the full current corpus, one epoch. It is still far below modern sub-1B token budgets, but it is a meaningful step up from 1B because the extra tokens are mostly unique documents.

TASK-009 is therefore the right next experiment: it converts the question from theory into Kestrel-specific evidence.

### Beyond 3.27B tokens

If the 3.27B run improves validation loss and downstream behavior, larger training is likely beneficial. The modern small-model prior suggests plausible next ranges:

| Total tokens | tokens/param | Interpretation |
|---:|---:|---|
| 3.27B | ~65 | first full-corpus epoch |
| 6.5B | ~130 | second epoch over current corpus |
| 9.8B | ~195 | third epoch over current corpus |
| 50B | ~1,000 | requires much more unique data or many epochs |
| 200B | ~4,000 | SmolLM-135M-like regime |
| 1T | ~20,000 | Gemma-3-270M / Qwen2-0.5B-like regime |

For a fixed 3.27B-token corpus, the immediate practical question is whether 2–3 epochs over the same deduplicated corpus still produce useful held-out and downstream gains.

## Updated decision rule

Do not choose the token budget from Chinchilla alone. Use this decision process:

1. **Run TASK-009 first.**
   - Compare 1B vs 3.27B full-corpus validation loss.
   - Confirm reproducibility around the 1B point.
   - Record downstream SFT/tool behavior if feasible.

2. **Use held-out validation loss as the primary curve.**
   - Track marginal improvement per additional 1B tokens.
   - Stop expanding when the marginal gain is small relative to compute cost.

3. **Use SFT/tool eval as the tie-breaker.**
   - Pretrain loss is necessary but not sufficient.
   - The actual Kestrel objective is downstream agentic behavior.

4. **Treat 2–3 epochs as the next evidence-backed range.**
   - If 3.27B is still improving, 6.5B–9.8B total tokens is a reasonable next experimental band.
   - Do not jump directly to very high epoch counts without deduplication and memorization checks.

5. **Check data quality before scaling epochs.**
   - Deduplicate.
   - Keep eval and SFT data disjoint.
   - Monitor repetition/memorization.
   - Prefer adding high-quality unique data over blindly repeating low-quality data.

6. **Re-evaluate after TASK-009.**
   - If 3.27B gives `>=0.1` nat improvement and downstream behavior improves, continue toward 2–3 epochs or a larger corpus.
   - If improvement is `<0.05` nat and downstream behavior does not improve, stop and move to the next pipeline stage.
   - If validation improves but downstream degrades, investigate data distribution, SFT mismatch, or overfitting before adding more tokens.

## Implementation caveat

The current full-checkpoint resume flow validates checkpoint state against the current config. Therefore “just increase `total_tokens` and resume” is not necessarily a valid continuation path, because the LR horizon and config fingerprint would change. A proper continuation experiment may require either:

- weights-only initialization from the 1B final checkpoint, or
- explicit continuation support in the pretrain entry point/trainer.

This should be scoped as a separate task before running the pilot.

## Conclusion

The original question was whether 3.27B tokens is worth it compared with 1B. The broader 2026-09-02 research update changes the framing:

```text
1B tokens is a Chinchilla-style lower bound.
3.27B tokens is the first serious full-corpus epoch.
Modern small models suggest 3.27B is not overtrained for 50M.
Larger training is likely beneficial if data quality and downstream eval support it.
```

The best current position is:

> Keep TASK-009 as the immediate experiment. Do not treat Chinchilla as the target. Use the 1B vs 3.27B result to decide whether Kestrel-50M should move into a 2–3 epoch regime, expand the corpus, or stop and proceed to SFT/agent work.

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, https://arxiv.org/abs/2203.15556
- Rae et al., *Scaling Language Models: On Methods, Data, and Compute*, https://arxiv.org/abs/2112.11446
- Muennighoff et al., *Scaling Data-Constrained Language Models*, https://arxiv.org/abs/2305.16264
- Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, https://arxiv.org/abs/2502.02737
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, https://arxiv.org/abs/2302.13971
- Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, https://arxiv.org/abs/2307.09288
- Dubey et al., *The Llama 3 Herd of Models*, https://arxiv.org/abs/2407.21783
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Query Checkpoints*, https://arxiv.org/abs/2305.13245
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, https://arxiv.org/abs/2104.09864
- Shazeer, *GLU Variants Improve Transformer*, https://arxiv.org/abs/2002.05202
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, https://arxiv.org/abs/1910.07467
- Sardana & Frankle, *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, https://arxiv.org/abs/2401.00448
- Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, https://arxiv.org/abs/2403.08540
- Roberts et al., *Test-Time Scaling Makes Overtraining Compute-Optimal*, https://arxiv.org/abs/2604.01411
- Wu et al., *Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference*, https://arxiv.org/abs/2408.00724
- Olsson et al., *Scaling Laws and Interpretability of Learning from Repeated Data*, https://arxiv.org/abs/2207.07251
- Balaji, *An Empirical Study of Compute-Efficient Token–Parameter Scaling in Small Language Models (1M–20M)*, https://doi.org/10.21203/rs.3.rs-8948933/v1
- SmolLM blog, https://huggingface.co/blog/smollm
- SmolLM2 model cards, https://huggingface.co/HuggingFaceTB/SmolLM2-135M, https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
- MobileLLM paper, https://arxiv.org/abs/2402.14905
- Qwen2 technical report, https://arxiv.org/abs/2407.10671
- Qwen3 technical report, https://arxiv.org/abs/2505.09388
- Llama 3.2 model card, https://huggingface.co/meta-llama/Llama-3.2-1B
- Gemma 3 model card, https://ai.google.dev/gemma/docs/core/model_card_3
- Local run data: `checkpoints/pretrain/50m/run.jsonl`
- Local corpus manifest: `data/corpus-12g/train/manifest.json`
- Local config: `configs/kestrel/50m/pretrain.yaml`
- Local full-corpus config: `configs/kestrel/50m/pretrain_3b.yaml`
