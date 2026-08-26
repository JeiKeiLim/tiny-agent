---
id: TASK-005.02.01
title: Mix multi-file pretrain batches instead of reading domain files sequentially
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-25 00:13'
updated_date: '2026-08-26 00:28'
labels:
  - data
  - pretrain
  - validation
milestone: m-1
dependencies: []
documentation:
  - doc-002
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
- [x] #1 A unit test with multiple tiny .txt files proves yielded batches contain tokens from more than one file
- [x] #2 A unit test with skewed file sizes proves the scheduled file share is within 5 percentage points of the byte-size weights over at least 10k scheduled line draws
- [x] #3 Single-file dataset behavior remains unchanged
- [x] #4 make check is green
- [x] #5 TASK-005.06 notes record that prior/current M1 val loss was domain-block biased until this task lands
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a WeightedLineScheduler helper in src/kestrel/data/pretrain_dataset.py that yields (path, line) pairs using seeded weighted sampling across active files.
2. Change PretrainDataset._resolve_files to return (path, weight) pairs, using byte size as the weight for directory inputs and a fixed weight for single-file inputs.
3. Replace the sequential file loop in PretrainDataset.__iter__ with the weighted line scheduler while preserving the existing token packing and total_tokens behavior.
4. Add tests for multi-file mixing, weighted share tolerance, determinism, and single-file behavior.
5. Run make check.
6. Append a note to TASK-005.06 explaining that prior M1 validation loss was domain-block biased until this task lands.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Research findings on standard pretraining data mixing recorded in doc-002: weighted domain mixing and interleaving are standard in public LLM recipes; current Kestrel domain-block order is an implementation artifact; the planned weighted multi-file scheduler aligns the design with standard practice.

Implemented WeightedLineScheduler and switched PretrainDataset directory inputs to seeded weighted line sampling. Added tests for multi-file mixing, weighted share tolerance, determinism, exhaustion, and single-file line order. make check is green with 83 tests.

Post-run correction: byte-weighted line sampling reduced early code dominance, but it is not the full fix. The corpus itself was flattened from document-level HF rows into physical lines. Document-aware corpus/dataset/model work is tracked under TASK-005.08.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @opencode
created: 2026-08-25 23:04
---
Post-run investigation: byte-weighted line sampling does not produce the intended token-domain mix because line lengths differ across web/code/jsonl. In the 50M run, jsonl exhausted early and the final ~3k steps were code-only. Follow-up should use token-aware weights, a corpus manifest, or a deficit-based token scheduler.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented deterministic weighted multi-file line scheduling for PretrainDataset. Directory inputs now sample lines from web/code/jsonl files according to byte-size weights instead of consuming files sequentially, making both training and sampled validation more representative. Verified with new unit tests and make check (83 tests green).
<!-- SECTION:FINAL_SUMMARY:END -->
