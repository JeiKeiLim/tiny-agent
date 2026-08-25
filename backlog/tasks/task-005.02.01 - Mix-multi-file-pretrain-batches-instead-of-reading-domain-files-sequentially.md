---
id: TASK-005.02.01
title: Mix multi-file pretrain batches instead of reading domain files sequentially
status: To Do
assignee: []
created_date: '2026-08-25 00:13'
labels:
  - data
  - pretrain
  - validation
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/data/pretrain_dataset.py
  - tests/test_pretrain_dataset.py
parent_task_id: TASK-005.02
priority: high
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PretrainDataset currently reads all .txt files in a split directory sequentially. It shuffles file order once, but then consumes an entire file before moving to the next file. With corpus builder output data/corpus/{train,val}/{web,code,jsonl}.txt, this causes domain-block training and unrepresentative validation.

Current 50M eval impact: eval_iters=10, batch_size=8, seq_len=1024 means ~80 validation sequences. With seed 0, the val file order is code.txt, web.txt, jsonl.txt, so the in-loop val loss is effectively code-only, not the intended ~85/10/5 web/code/jsonl mix.

Goal: make batches from a directory of .txt files draw lines from multiple files in a deterministic, weighted, interleaved order so both training and sampled validation are representative of the split's domain mix.

Proposed implementation:
- Modify src/kestrel/data/pretrain_dataset.py.
- _resolve_files should return file paths plus weights, initially weighted by file size in bytes as an approximation of token/domain share.
- Replace the sequential for path in self._files loop with a weighted line scheduler.
- Use random.Random(config.seed) for deterministic scheduling.
- Stop scheduling a file once it is exhausted.
- Preserve single-file behavior.
- Preserve existing token-packing logic; packed sequences may still span lines/files.
- Do not change corpus builder output in this task.

Gotchas:
- Byte-size weighting is approximate; acceptable for M1. If exact component fractions are required later, add a corpus manifest instead.
- Do not shuffle lines inside each large file; only schedule across files.
- __iter__ must restart deterministically so periodic validation uses the same representative sample.
- The current/previous M1 run used the old block-order behavior; its val loss should not be interpreted as mixed-domain.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A unit test with multiple tiny .txt files proves yielded batches contain tokens from more than one file
- [ ] #2 A unit test with skewed file sizes proves the scheduled file share is within 5 percentage points of the byte-size weights over at least 10k scheduled line draws
- [ ] #3 Single-file dataset behavior remains unchanged
- [ ] #4 make check is green
- [ ] #5 TASK-005.06 notes record that prior/current M1 val loss was domain-block biased until this task lands
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect tests/test_pretrain_dataset.py and current PretrainDataset behavior.
2. Implement a deterministic weighted multi-file line scheduler in src/kestrel/data/pretrain_dataset.py.
3. Add unit tests for multi-file mixing, weighted distribution, determinism, and single-file behavior.
4. Run make check.
5. Append a note to TASK-005.06 explaining the old val-loss bias.
<!-- SECTION:PLAN:END -->
