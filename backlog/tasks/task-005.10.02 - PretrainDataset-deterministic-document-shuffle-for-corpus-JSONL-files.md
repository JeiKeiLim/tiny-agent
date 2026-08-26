---
id: TASK-005.10.02
title: 'PretrainDataset: deterministic document shuffle for corpus JSONL files'
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-26 01:35'
updated_date: '2026-08-26 02:31'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/data/pretrain_dataset.py
  - tests/test_pretrain_dataset.py
parent_task_id: TASK-005.10
priority: high
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current PretrainDataset consumes each corpus JSONL file sequentially. For a 12GiB corpus streamed from HF sources, sequential file order can introduce source or dump order bias. Add deterministic per-file document shuffle while preserving the existing token-deficit domain mixing and doc_ids behavior.

Files:
- modify src/kestrel/data/pretrain_dataset.py
- modify tests/test_pretrain_dataset.py

Design:
- Add a helper that collects physical line byte offsets for a .jsonl or legacy .txt corpus file.
- Add _iter_documents_shuffled(path, seed) that shuffles those offsets deterministically and reads each physical line by seeking to its offset.
- Use the shuffled iterator in PretrainDataset.__iter__ instead of the sequential _iter_documents iterator.
- Derive a deterministic per-source seed from config.seed and source.domain or path so the same corpus + seed is reproducible.
- Keep memory usage bounded: store line offsets, not full document text. An array of 64-bit offsets is preferred over a list of Python ints for large files.

Behavior:
- Every document is emitted exactly once.
- The same seed produces the same document order.
- Different seeds may produce different orders.
- Single-file inputs and directory inputs both work.
- doc_ids are still assigned in consumption order, so document-aware attention and position reset remain unchanged.

Tests:
- A tiny JSONL file with multiple docs is not consumed in raw file order for a fixed seed.
- The same seed yields the same shuffled document order.
- All docs are emitted exactly once.
- Directory input with multiple component files still yields batches and respects total_tokens.
- make check green.

Gate: make check green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PretrainDataset no longer consumes a multi-doc JSONL file in raw file order
- [x] #2 The same dataset seed produces the same document order across runs
- [x] #3 All documents are emitted exactly once
- [x] #4 Single-file and directory corpus inputs still work
- [x] #5 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add line-offset scanning for .jsonl/.txt corpus files using 64-bit offsets. 2. Add deterministic per-source shuffle seed derived from config.seed + source.domain. 3. Add _iter_documents_shuffled() that shuffles offsets and reads lines by seeking while preserving existing JSONL/txt parsing behavior. 4. Use the shuffled iterator in PretrainDataset.__iter__. 5. Add tests for non-raw order, same-seed reproducibility, exactly-once document emission, and directory input. 6. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented deterministic per-file document shuffle using 64-bit physical line offsets and numpy RNG. Per-source seed is derived from config.seed + source.domain. PretrainDataset.__iter__ now uses _iter_documents_shuffled. Added tests for non-raw order, same-seed reproducibility, exactly-once emission, and directory total_tokens behavior. make check passed with 116 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added deterministic per-file document shuffle to PretrainDataset while preserving token-deficit mixing and doc_id assignment. Verified with shuffle regression tests and make check (116 tests passed).
<!-- SECTION:FINAL_SUMMARY:END -->
