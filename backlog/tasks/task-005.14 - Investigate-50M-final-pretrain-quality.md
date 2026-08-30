---
id: TASK-005.14
title: Investigate 50M final pretrain quality
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-28 07:48'
updated_date: '2026-08-28 07:53'
labels:
  - eval
  - pretrain
  - 50m
milestone: m-1
dependencies: []
parent_task_id: TASK-005
priority: high
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 50M 1B-token pretrain run has completed. Investigate whether the final 50M pretrained model is sufficiently trained to serve as the SFT validation base.

This is a read-only evaluation/investigation task. It should not modify model code, training code, checkpoints, or configs.

Goal:
- Produce an evidence-based judgment of whether Kestrel-50M pretraining succeeded for the project's M1 validation purpose.
- Compare quantitative validation metrics with qualitative generation samples.
- Decide whether the 50M checkpoint is adequate for starting SFT experiments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run eval_pretrain.py on the 50M final checkpoint and record loss/perplexity/bits-per-token metrics
- [x] #2 Compare final and best checkpoints when their validation metrics differ
- [x] #3 Summarize the full run.jsonl training curve and confirm the LR schedule completed
- [x] #4 Collect check_model.py generation samples from at least three prompts
- [x] #5 Collect generation samples under at least two decoding settings
- [x] #6 Record qualitative observations on coherence, repetition, syntax, and semantic limitation
- [x] #7 Produce an explicit judgment on whether the 50M model is adequate as an SFT validation base
- [x] #8 Do not modify source code, configs, checkpoints, or training state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run read-only pretrain evaluation on checkpoints/pretrain/50m/final using configs/kestrel/50m/pretrain.yaml and the validation split.
2. Run the same evaluation on checkpoints/pretrain/50m/best if its metrics or checkpoint differ meaningfully.
3. Summarize checkpoints/pretrain/50m/run.jsonl: step count, token count, initial loss, best in-loop val loss, final val loss, and LR completion.
4. Run scripts/check_model.py against the final checkpoint with multiple prompts and decoding settings, including temp=0.0, temp=0.1, and repetition_penalty=1.2 where useful.
5. Record generation samples and qualitative observations: coherence, repetition, syntax, vocabulary, semantic limits, and prompt sensitivity.
6. Write the final judgment in the task notes/final summary: adequate for SFT validation, adequate with caveats, or not adequate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Context:
- 50M pretrain final checkpoint is at checkpoints/pretrain/50m/final.
- 150M pretrain is currently running or being considered; keep this task read-only and avoid heavy concurrent training workloads.
- The project goal is pipeline validation and 50M/150M learning, not frontier capability. Judge the model against that goal, not against general assistant quality.

Evaluated 302,808 held-out validation tokens. Final mixed loss=3.144315, perplexity=23.2038, bits/token=4.5363. Best mixed loss=3.144318, perplexity=23.2038; final and best are effectively identical. Domain losses (final): web=3.2953/ppl 26.99, code=1.9249/ppl 6.85, synthetic=2.6949/ppl 14.80.

Run summary: 123,719 steps, 1,013,506,048 tokens, LR completed to 0.0. Initial train loss=10.1537, last-100 train mean=3.1463, best in-loop val=3.1583 at step 123000, last in-loop val=3.1583. No obvious train/val divergence.

Generation samples:
- temp=0.0, 'The quick brown fox...': produced 'The dog is a good source of protein, but it is also a good source of protein.' with repetition.
- temp=0.0, 'Once upon a time': produced repetitive 'young man' sentence fragments.
- temp=0.0, 'def add(a, b):': produced syntactically plausible but semantically wrong Python: 'a = a.add(b)', 'b = a.add(b)', 'return a'.
- temp=0.1 + repetition_penalty=1.2, default prompt: less exact repetition, coherent generic clause about protein/vitamins/supplement.
- temp=0.1 + repetition_penalty=1.2, 'The capital of France is': produced repetitive and factually unreliable text: 'the capital of France. The capital of France is located in the north-east corner... major port city'.

Qualitative judgment: the model has learned surface language, register, and code patterns, but shows repetition, weak factual reliability, and limited semantic reasoning. It is adequate as a 50M SFT validation base for pipeline mechanics, not a capable general assistant.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
50M final pretrain is judged adequately trained for M1 validation and SFT-pipeline experiments, with caveats. It completed 1.0135B tokens, LR reached 0, final held-out mixed loss was 3.1443 (ppl 23.20), code loss was 1.9249 (ppl 6.85), and final/best checkpoints were effectively identical. Generation shows coherent local language and code surface patterns, but also repetition, weak factual reliability, and limited semantic reasoning. Use checkpoints/pretrain/50m/final as the 50M SFT validation base.
<!-- SECTION:FINAL_SUMMARY:END -->
