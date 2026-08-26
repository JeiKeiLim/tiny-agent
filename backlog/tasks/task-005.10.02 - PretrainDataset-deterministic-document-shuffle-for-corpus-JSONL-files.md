---
id: TASK-005.10.02
title: 'PretrainDataset: deterministic document shuffle for corpus JSONL files'
status: To Do
assignee: []
created_date: '2026-08-26 01:35'
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
- [ ] #1 PretrainDataset no longer consumes a multi-doc JSONL file in raw file order
- [ ] #2 The same dataset seed produces the same document order across runs
- [ ] #3 All documents are emitted exactly once
- [ ] #4 Single-file and directory corpus inputs still work
- [ ] #5 make check is green
<!-- AC:END -->
