---
id: TASK-013
title: Analyze 50M 3B pretrain run and late-training degradation
status: Done
assignee:
  - '@opencode'
created_date: '2026-09-06 23:33'
updated_date: '2026-09-07 07:37'
labels:
  - analysis
  - pretrain
dependencies: []
ordinal: 62000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Analyze checkpoints/pretrain/50m-3b after the 3B+ token single-pass pretrain run. The user observed a late drop in train loss, an increase in validation loss, and final model generations that collapse to Python code. Determine the training duration, interruption/restart behavior, best checkpoint, loss trajectory, data-order/corpus-exhaustion hypothesis, and which checkpoint should be used for downstream work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify best and final checkpoint step/token/loss values from state.json and run.jsonl
- [x] #2 Analyze train/validation loss trajectory and identify when degradation began
- [x] #3 Inspect pretrain dataset/corpus order and evaluate whether late training overfit a Python/code-heavy suffix
- [x] #4 Recommend the checkpoint to use and any follow-up data/ordering fixes
- [x] #5 Summarize total steps, tokens, calendar span, resume resets, and explain why exact active wall-clock/gap durations are not recoverable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Parse run.jsonl for steps, evals, resets, and loss windows. 2. Inspect best/final/step checkpoint state.json for step, tokens, val loss, and source_states. 3. Inspect corpus manifest and pretrain_dataset scheduler. 4. Decode late dataset buffer to confirm code dominance. 5. Report duration, degradation onset, root cause, and checkpoint recommendation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
run.jsonl has no timestamps. Calendar span from file birth Sep 1 14:29:11 to final Sep 7 05:56:31 is 5d 15h27m. Unique final step=381833, emitted_total=3127981056. Two resume resets: 222012->222001 and 265999->264001, duplicate work=2011 steps=16474112 tokens. Best step=368000 val=3.021303820610046. Final last eval step=381000 val=3.265814423561096. Train avg drops from 2.987 at 360k-370k to 1.601 at 370k-380k. Web source exhausted by step_372000; synthetic already exhausted at best. Code-only tokens from step_372000 to final=80557977. Decoded step_380000/final dataset buffers are Python code.

Exact active wall-clock time and restart gap durations are not recoverable from current run.jsonl because entries do not include timestamps. We can report calendar span, resume step resets, duplicate work, and final-segment checkpoint cadence, but not exact downtime durations.

Detailed analysis report for checkpoints/pretrain/50m-3b:

Run totals:
- final step 381833
- emitted tokens 3127981056
- tokens per step 8192
- calendar span Sep 1 14:29:11 to Sep 7 05:56:31, 5d 15h 27m
- exact active wall-clock and shutdown gap durations are not recoverable because run.jsonl has no timestamps
- final-segment checkpoint cadence from step 368000 to final was about 2000 steps per 49-50 min, roughly 5.5k tokens/sec
- resume resets observed at 222012 -> 222001 and 265999 -> 264001, duplicate work 2011 steps or 16474112 tokens

Checkpoint result:
- best checkpoint: checkpoints/pretrain/50m-3b/best, step 368000, val loss 3.021303820610046, val ppl about 20.5
- final checkpoint: checkpoints/pretrain/50m-3b/final, step 381833, last eval step 381000, val loss 3.265814423561096, val ppl about 26.2
- planned schedule_steps 399735, actual stop 381833 because corpus exhausted

Loss trajectory:
- val loss improved until step 368000
- step 368000 val 3.0213, best
- step 370000 val 3.026, degradation begins
- step 372000 val 3.065
- step 374000 val 3.123
- step 376000 val 3.176
- step 378000 val 3.223
- step 380000 val 3.256
- step 381000 val 3.2658
- average train loss falls from about 2.99 in 360k-370k to about 1.60 in 370k-380k, indicating the training distribution became easier/narrower rather than healthy generalization

Data-order root cause:
- corpus train mix was web 0.85, code 0.10, synthetic 0.05
- at best step 368000, synthetic was already exhausted, web had 10157 docs left, and code had 30412 docs left
- by step 372000, web was exhausted
- from step 372000 to final, web emitted tokens stayed constant while code emitted +80557977 tokens
- decoded dataset buffers in step_380000 and final are Python code
- therefore the final model trained on a long code-only suffix and collapsed to Python generation
- this confirms the user hypothesis, with the refinement that web exhausted near the end and synthetic had already exhausted earlier

Recommendation:
- use checkpoints/pretrain/50m-3b/best for SFT, eval, and downstream experiments
- do not use checkpoints/pretrain/50m-3b/final except as a failure artifact

Follow-up fixes to consider:
- add timestamps to run.jsonl
- log per-source emitted/remaining docs and tokens
- add source-exhaustion lookahead to the pretrain dataset scheduler
- stop single-pass runs before a major source exhausts or use a safe token cap
- consider early stopping on validation loss
- consider interleaving or capping code so the final suffix cannot become code-only
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Analyzed checkpoints/pretrain/50m-3b. The run reached step 381833 / 3.128B emitted tokens over a Sep 1 14:29 to Sep 7 05:56 calendar span, with two resume resets. Best checkpoint is step 368000 with val loss 3.0213. Degradation began immediately after step 368000: val rose to 3.2658 by step 381000 while train loss fell to ~1.6. Root cause was corpus-source exhaustion: synthetic was already exhausted, web exhausted by step 372000, and the final ~80.56M tokens were code-only, causing the final model to collapse to Python generation. Use checkpoints/pretrain/50m-3b/best for downstream work; do not use final. Future runs need source-exhaustion lookahead/safe token cap and per-source/timestamp telemetry.
<!-- SECTION:FINAL_SUMMARY:END -->
