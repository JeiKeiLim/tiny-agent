---
id: doc-008
title: 'SFT Drift, Pretrain Perplexity, and Tool-Call Failure Research'
type: other
created_date: '2026-09-01 22:07'
updated_date: '2026-09-01 22:10'
tags:
  - research
  - sft
  - eval
  - forgetting
  - tool-calling
---
# SFT Drift, Pretrain Perplexity, and Tool-Call Failure Research

## Status

Research notes from the Kestrel 50M M2 SFT investigation.

- Date: 2026-09-02
- Related task: `TASK-007.03.14 - Fix SFT inference prompt and special-token decoding`
- Related validation task: `TASK-007.03.10 - Run 50M SFT data-scaling validation`
- Related decision: `doc-007 - M2 SFT Tool-Calling Format Decision`
- Primary artifact: `data/sft/eval/scorecard.json`
- Checkpoint analyzed: `checkpoints/sft/50m/final`

## Executive summary

After fixing the SFT eval/inference prompt and special-token decoding bugs, the 50M SFT checkpoint shows clear improvement on assistant-style behavior, but strict tool-call JSON generation still fails.

The main observations are:

1. SFT improved assistant non-empty behavior from `0.825` to `1.000` in the corrected scorecard.
2. SFT improved no-call and missing-info behavior from roughly `0.24` / `0.172` to `1.000`.
3. Tool seen/unseen `valid_json_rate` remains `0.0`.
4. Manual probing shows the model now emits the `tool_call` marker in direct tool cases, but the JSON payload is usually malformed.
5. Pretrain-validation perplexity increases from `24.14` to `50.08` after SFT.
6. The perplexity increase is consistent with documented SFT drift / catastrophic forgetting on prior pretraining data.
7. The specific hypothesis that removing JSON-heavy pretraining data caused the tool-call failure is plausible but not yet proven.

## 1. Corrected SFT scorecard

The corrected scorecard was generated after fixing:

- `generate()` now preserves tokenizer special tokens by default.
- SFT eval prompts append the assistant completion prefix:
  `im_start` + newline + `im_assistant` + newline.
- Interactive chat prompts use the same completion prefix.

### Pretrain checkpoint

| Metric | Value |
|---|---:|
| assistant non-empty | 0.630 |
| assistant no-tool-call | 1.000 |
| assistant no-repetition | 0.510 |
| GSM8K exact match | 0.002 |
| tool seen valid JSON | 0.000 |
| tool unseen valid JSON | 0.000 |
| no-call correct | 0.140 |
| missing-info correct | 0.000 |
| pretrain-val loss | 3.1839 |
| pretrain-val perplexity | 24.141 |

### SFT 50k checkpoint

| Metric | Corrected value | Previous buggy-eval value |
|---|---:|---:|
| assistant non-empty | 1.000 | 0.825 |
| assistant no-tool-call | 1.000 | 1.000 |
| assistant no-repetition | 0.520 | 0.555 |
| GSM8K exact match | 0.012 | 0.020 |
| tool seen valid JSON | 0.000 | 0.000 |
| tool unseen valid JSON | 0.000 | 0.000 |
| no-call correct | 1.000 | 0.240 |
| missing-info correct | 1.000 | 0.172 |
| pretrain-val loss | 3.9137 | 3.9137 |
| pretrain-val perplexity | 50.084 | 50.084 |

The `sft_5k` and `sft_20k` checkpoints are missing because only the full 50k SFT run was executed.

## 2. Manual tool-call probe

A direct probe over the first 10 `tool_seen` and first 10 `tool_unseen` eval rows showed:

| Probe | Result |
|---|---:|
| prompt ends with assistant prefix | 20/20 |
| raw output contains `tool_call` | 20/20 |
| valid JSON | 0/20 |
| schema valid | 0/20 |
| correct tool name | 0/20 |
| exact arguments | 0/20 |

The dominant failure mode is missing JSON object structure. Typical output:

```text
tool_call name":"lookup_current_weather","arguments":{"city":"Austin","unit":"fahrenheit"}}}}
```

Expected format:

```text
tool_call
{"name":"lookup_current_weather","arguments":{"city":"Austin","unit":"fahrenheit"}}
tool_call_end
```

Common failures:

- missing opening `{`
- missing newline after `tool_call`
- repeated `}`
- duplicate argument keys
- missing `tool_call_end`
- wrong tool name or arguments in some unseen cases

Repetition penalty improved termination but did not fix JSON validity:

| Repetition penalty | `tool_call` emitted | `tool_call_end` emitted | valid JSON |
|---:|---:|---:|---:|
| 1.0 | 20/20 | 8/20 | 0/20 |
| 1.1 | 20/20 | 16/20 | 0/20 |
| 1.2 | 20/20 | 18/20 | 0/20 |
| 1.3 | 20/20 | 19/20 | 0/20 |

This indicates the remaining tool-call problem is primarily a model/format-learning problem, not the eval harness bug.

## 3. Scorecard metric definitions

The SFT scorecard metrics are mostly rates, i.e. fractions of eval rows where a behavior occurred. They are not all accuracy metrics.

### Assistant metrics

| Metric | Meaning |
|---|---|
| `non_empty_rate` | Fraction of assistant rows where generated visible text is non-empty. |
| `no_tool_call_rate` | Fraction where the model did not emit the `tool_call` special token. |
| `no_repetition_rate` | Fraction where a simple heuristic did not flag obvious degenerate repetition. |

The repetition heuristic checks word uniqueness and repeated 5-word shingles. It is a sanity metric, not a quality score.

### Math metric

| Metric | Meaning |
|---|---|
| `exact_match_rate` | Extracts the final number from expected and generated text and checks numeric equality. This is accuracy-like. |

### Tool seen/unseen metrics

| Metric | Meaning |
|---|---|
| `valid_json_rate` | Fraction where the generated tool payload parses as a JSON object. |
| `schema_valid_rate` | Fraction where parsed arguments validate against the expected tool schema. |
| `tool_selection_rate` | Fraction where JSON is valid and the tool name matches the expected tool. |
| `argument_exact_rate` | Fraction where tool name and arguments exactly match the expected call. |
| `argument_partial_accuracy` | Average fraction of expected argument keys whose values match exactly. |

### No-call / missing-info metrics

| Metric | Meaning |
|---|---|
| `no_tool_call_rate` | Fraction where the model did not emit `tool_call`. |
| `non_empty_rate` | Fraction where the model produced non-empty text. |
| `correct_rate` | Fraction where the model did not call a tool and produced non-empty text. |

Important: `correct_rate` for no-call and missing-info is not semantic answer correctness. It only measures that the model behaved like a text answer rather than a tool call.

### Perplexity metric

| Metric | Meaning |
|---|---|
| `loss` | Token-averaged cross-entropy on held-out pretrain validation text. |
| `perplexity` | `exp(loss)`. |
| `bits_per_token` | `loss / ln(2)`. |

This perplexity is measured on the pretrain validation corpus, not on SFT-style assistant/math/tool rows.

## 4. How pretrain perplexity is computed

The perplexity metric is teacher-forced next-token cross-entropy.

In `src/kestrel/eval/pretrain.py`:

```python
for x, target, doc_ids in iterator:
    logits = model(x, doc_ids)
    loss_sum = cross_entropy(logits[:, :-1], target[:, :-1], reduction="sum")
    tokens = x.shape[0] * max(x.shape[1] - 1, 0)
    acc.add(loss_sum.item(), tokens)
```

For each position:

1. The model reads the token sequence up to that position.
2. It outputs logits over the vocabulary.
3. The logits are converted into a probability distribution.
4. The loss is `-log P(actual next token | previous tokens)`.
5. Loss is averaged over evaluated tokens.
6. Perplexity is `exp(average_loss)`.

It does not generate text. It measures how surprised the model is by the held-out pretrain text.

Therefore, this metric answers:

> How well does this checkpoint still model the original pretraining distribution?

It does not directly answer:

> How well does this checkpoint perform SFT tasks?

For that, we need held-out SFT cross-entropy and/or task metrics.

## 5. Interpretation: SFT drift is expected

SFT trains the model on a narrower distribution:

- ChatML role structure
- assistant-style answers
- GSM8K-style math
- tool-call JSON
- clarification/refusal behavior

After SFT, the model is biased toward completing assistant turns rather than continuing arbitrary web/code text. As a result, pretrain-validation perplexity can increase even while SFT task behavior improves.

This is especially plausible for Kestrel because:

- the model is only 50M parameters
- SFT is full fine-tuning, not LoRA
- the SFT data is narrow and task-specific
- the model has high plasticity relative to its capacity
- the current tool-call format requires exact serialization

The observed result is therefore a classic task-specialization tradeoff:

```text
SFT task behavior: improved
general pretrain distribution: worse
strict tool serialization: still underfit
```

## 6. Research backing

The following sources support the general claim that SFT/instruction tuning can drift a model away from its pretraining distribution and that this drift is commonly measured as increased loss or log-perplexity on prior data.

### 6.1 Jin & Ren — Demystifying Forgetting in Language Model Fine-Tuning

Source: [arXiv:2406.14026](https://ar5iv.labs.arxiv.org/html/2406.14026)

Key findings:

- Forgetting is defined as degradation, i.e. increase, in log-perplexity on upstream examples after fine-tuning.
- Experiments fine-tune OLMo-7B on instruction tasks and evaluate forgetting on Dolma pretraining examples.
- Dolma upstream log-perplexity before fine-tuning: `2.283`.
- Average forgetting across instruction tasks: `+0.035` log-perplexity.
- Some tasks cause negative forgetting, some cause positive forgetting.

Relevance:

- Directly supports using pretrain-data log-perplexity increase as a forgetting/drift metric.
- Shows instruction tuning can make a model worse on prior pretraining text.
- Shows the effect is task/data dependent, not universal in magnitude.

### 6.2 Kalajdzievski — Scaling Laws for Forgetting When Fine-Tuning LLMs

Source: [arXiv:2401.05605](https://arxiv.org/abs/2401.05605)

Key findings:

- PEFT strategies such as LoRA still suffer catastrophic forgetting.
- There is a strong inverse linear relationship between fine-tuning performance and forgetting.
- Forgetting increases as a shifted power law in:
  - number of parameters fine-tuned
  - number of update steps
- Early stopping does not avoid forgetting.

Relevance:

- Supports the tradeoff between task adaptation and retention.
- Suggests full fine-tuning and more update steps should generally produce more drift than smaller or more constrained updates.

### 6.3 Biderman et al. — LoRA Learns Less and Forgets Less

Source: [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)

Key findings:

- Full fine-tuning usually learns the target domain better than LoRA.
- LoRA better maintains base-model performance outside the target domain.
- LoRA mitigates forgetting more than common regularization such as weight decay and dropout.
- Full fine-tuning learns high-rank perturbations, while typical LoRA learns lower-rank perturbations.

Relevance:

- Kestrel’s current SFT is full fine-tuning.
- Higher drift on pretrain text is expected compared with a LoRA/PEFT variant.

### 6.4 Luo et al. — Catastrophic Forgetting in LLMs During Continual Fine-tuning

Source: [arXiv:2308.08747](https://arxiv.org/abs/2308.08747)

Key findings:

- Catastrophic forgetting is observed in LLMs from 1B to 7B during continual instruction tuning.
- Forgetting is evaluated across domain knowledge, reasoning, and reading comprehension.
- General instruction tuning can help alleviate forgetting during subsequent fine-tuning.

Relevance:

- Shows the phenomenon is not limited to very large models.
- Supports monitoring general ability after task-specific instruction tuning.

### 6.5 Jin & Ren — What Will My Model Forget?

Source: [arXiv:2402.01865](https://arxiv.org/abs/2402.01865)

Key findings:

- Model updates can cause catastrophic forgetting of upstream pretraining or instruction-tuning examples.
- Forecasting which upstream examples will be forgotten can improve replay-based mitigation.

Relevance:

- Supports the idea that SFT updates can selectively damage prior capabilities.
- Motivates keeping a general retention metric alongside task metrics.

## 7. What is proven versus what is still hypothesis

| Claim | Status | Evidence |
|---|---|---|
| SFT improved assistant/no-call behavior | Supported | Corrected scorecard |
| Tool-call eval harness bug affected previous tool metrics | Supported | Prompt/decode fix + manual probe |
| Model now emits `tool_call` marker in direct tool cases | Supported | 20/20 manual probe |
| Model still fails strict tool JSON | Supported | 0/20 valid JSON probe |
| SFT increased pretrain-val perplexity | Supported | Scorecard: 24.14 -> 50.08 |
| SFT can drift models away from pretrain distribution | Research-backed | Papers above |
| Full fine-tuning drifts more than LoRA | Research-backed | Biderman et al. |
| Removing JSON-heavy pretrain data caused the tool-call failure | Hypothesis | Not yet isolated by ablation |
| 50M size is the main cause | Hypothesis | Plausible, but not isolated |
| More tool SFT data will fix JSON serialization | Hypothesis | Plausible, needs experiment |

## 8. Hypotheses for the tool-call JSON failure

Ranked by current evidence:

### 8.1 SFT underfitting of the exact tool-call format

Evidence:

- The model emits the `tool_call` marker.
- It often emits plausible tool names and argument values.
- It fails the boundary structure: newline, opening `{`, closing `}`, `tool_call_end`.

This is the most direct explanation.

Possible fixes:

- more tool rows
- more epochs on tool data
- lower LR
- stronger tool-format oversampling
- simpler tool-call format
- explicit format warm-up data

### 8.2 Full SFT drift / forgetting

Evidence:

- Pretrain-val perplexity increased.
- Full fine-tuning is known to drift more than LoRA.
- The model is small.

This may reduce general JSON/code fluency, but it does not by itself explain the very specific missing `{` failure.

Possible fixes:

- LoRA/PEFT SFT
- pretrain replay mixed into SFT
- lower LR
- fewer update steps
- stronger regularization

### 8.3 Reduced JSON/code readiness from pretrain data mix

Current pretrain corpus:

```yaml
web: 0.85
code: 0.10
synthetic: 0.05
```

The older 1G tokenizer/pretrain experiment included a dedicated JSONL/Alpaca component. The current 12G corpus does not have an explicit JSON/Alpaca component, though code likely contains some JSON.

This is plausible but not proven. The SFT data itself contains 15k tool rows with JSON payloads, so the model did see JSON during SFT.

Possible test:

- continue pretraining on JSON/code/tool-format data
- rebuild a pretrain corpus with explicit JSON component
- compare tool-call SFT results

### 8.4 Learning rate / schedule / step count

Current SFT config:

```yaml
epochs: 1
lr: 0.00005
batch_size: 8
seq_len: 1024
```

This is modest, but the model is small and the task format is exact. The optimum may be lower LR, fewer steps, or a different schedule.

### 8.5 Tokenizer/format boundary difficulty

The required sequence is:

```text
im_start
im_assistant
tool_call
{"name":...,"arguments":{...}}
tool_call_end
im_end
```

The model appears to learn the semantic payload but not the exact structural boundary. This could be a capacity, data, or tokenizer interaction issue.

## 9. Recommended follow-up experiments

### 9.1 Add better tool diagnostics to the scorecard

The current scorecard only reports final tool metrics. Add at least:

```text
attempted_tool_call_rate
tool_call_end_rate
payload_starts_with_brace_rate
first_token_is_tool_call_rate
```

This would separate:

- model did not attempt tool call
- model attempted but failed termination
- model attempted and terminated but JSON invalid
- JSON valid but schema/tool/args wrong

### 9.2 Add held-out SFT cross-entropy

Add a metric that computes masked cross-entropy on held-out SFT rows.

This would measure:

```text
fit to SFT task distribution
```

while the existing perplexity continues to measure:

```text
retention of pretrain distribution
```

### 9.3 Tool-call ablations

Priority experiments:

1. Current pretrain + current SFT, but with tool rows oversampled.
2. Current pretrain + lower SFT LR, e.g. `2e-5` or `1e-5`.
3. Current pretrain + shorter SFT run / earlier checkpoint.
4. Current pretrain + LoRA SFT, if Track B is available.
5. Current pretrain + small pretrain replay mixed into SFT.
6. JSON-rich continued pretrain, then same SFT.
7. Pretrain with explicit JSON/Alpaca component, then same SFT.

The cleanest first experiment is likely tool oversampling or lower LR, because it is cheaper than rebuilding pretraining data.

### 9.4 Repetition penalty policy

Repetition penalty improves `tool_call_end` emission but does not fix JSON validity. Decide whether scorecard should report:

- raw greedy, penalty `1.0`
- decoding-assisted, penalty `1.1`–`1.3`

These should be separate metrics, not silently mixed.

## 10. Decision guidance

For `TASK-007.03.14`:

- The eval/inference bug fix is successful.
- The corrected scorecard is now measuring the intended prompt contract.
- The remaining tool-call failure should be tracked as separate model-quality work.

For `TASK-007.03.10`:

- The 50k SFT run should not be considered a successful tool-call validation yet.
- Assistant/no-call behavior improved.
- Math remains weak.
- Strict tool-call JSON remains failing.

For future SFT runs:

- Keep pretrain perplexity as a forgetting monitor.
- Add held-out SFT loss as a task-fit monitor.
- Add tool attempt/termination diagnostics.
- Do not attribute the tool-call failure to pretrain data mix until an ablation supports it.

## References

- Jin & Ren, “Demystifying Forgetting in Language Model Fine-Tuning with Statistical Analysis of Example Associations”: https://ar5iv.labs.arxiv.org/html/2406.14026
- Kalajdzievski, “Scaling Laws for Forgetting When Fine-Tuning Large Language Models”: https://arxiv.org/abs/2401.05605
- Biderman et al., “LoRA Learns Less and Forgets Less”: https://arxiv.org/abs/2405.09673
- Luo et al., “An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning”: https://arxiv.org/abs/2308.08747
- Jin & Ren, “What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”: https://arxiv.org/abs/2402.01865
