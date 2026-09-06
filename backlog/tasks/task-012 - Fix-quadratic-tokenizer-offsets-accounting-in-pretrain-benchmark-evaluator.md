---
id: TASK-012
title: Fix quadratic tokenizer offsets accounting in pretrain benchmark evaluator
status: Done
assignee: []
created_date: '2026-09-06 08:20'
updated_date: '2026-09-06 08:25'
labels:
  - bug
  - performance
  - eval
dependencies: []
modified_files:
  - src/kestrel/eval/pretrain_benchmarks.py
  - tests/eval/test_pretrain_benchmarks.py
priority: high
ordinal: 61000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The pretrain benchmark evaluator is pathologically slow on long PILE documents.

In `src/kestrel/eval/pretrain_benchmarks.py`, `evaluate_language_modeling` computes per-chunk byte counts with:

```python
chunk_bytes = sum(
    int(encoding.offsets[index][1]) - int(encoding.offsets[index][0])
    for index in range(start, end)
)
```

Each `encoding.offsets` access materializes the full offset sequence from the tokenizer. Because this happens repeatedly inside a per-token generator, long rows cause roughly quadratic CPU cost. A live process sample showed the main thread stuck in:

```text
builtin_sum
  PyEncoding::get_offsets
  owned_sequence_into_pyobject
```

This made full `pile_test` evaluation estimate to many days/weeks instead of hours.

Fix the evaluator so tokenizer offsets are materialized once per encoded row and byte spans are computed without repeated full-offset conversion. Preserve the existing BPB byte accounting semantics exactly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LM benchmark byte accounting no longer accesses encoding.offsets repeatedly per token
- [x] #2 BPB, loss, perplexity, bits/token, tokens, and bytes outputs remain unchanged for existing benchmark tests
- [x] #3 MCQ continuation logprob path caches encoding.offsets instead of accessing it inside the per-token loop
- [x] #4 Add or update tests covering the byte-length helper and existing LM/MCQ evaluator behavior
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a small helper to convert encoding.offsets to per-token byte lengths once. 2. Use prefix sums or cached token byte lengths in evaluate_language_modeling. 3. Cache offsets in _continuation_logprob. 4. Add focused tests. 5. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Do not change metric semantics while fixing performance. The current byte sum includes every token in the chunk, including the first token, even though evaluated_tokens is len(chunk)-1. Preserve that behavior unless a separate metric task decides otherwise.

Implemented _offset_byte_prefix helper. LM evaluation now builds a per-row byte prefix once and uses O(1) chunk byte lookups. MCQ continuation logprob caches encoding.offsets once. Added tests for helper access count and text-length prefix. make check passed: ruff, ruff format, mypy, 382 tests.

Microbenchmarked old vs new byte accounting on real PILE rows using the project tokenizer: 6,398-token row old=2.738s new=0.001s (~2,286x); 19,692-token row old=28.129s new=0.002s (~12,804x). Byte totals matched. This isolates the offsets hot path, not model-forward time.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed quadratic tokenizer offset accounting in the pretrain benchmark evaluator. LM byte accounting now materializes offsets once per row via a prefix sum, and MCQ caches offsets instead of re-accessing them per token. Verified with make check: 382 tests passed.
<!-- SECTION:FINAL_SUMMARY:END -->
