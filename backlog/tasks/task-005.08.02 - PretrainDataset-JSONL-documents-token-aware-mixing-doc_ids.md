---
id: TASK-005.08.02
title: 'PretrainDataset: JSONL documents, token-aware mixing, doc_ids'
status: To Do
assignee: []
created_date: '2026-08-26 00:25'
updated_date: '2026-08-26 00:25'
labels:
  - data
  - pretraining
milestone: m-1
dependencies:
  - TASK-005.08.01
documentation:
  - doc-003
modified_files:
  - src/kestrel/data/pretrain_dataset.py
  - tests/test_pretrain_dataset.py
parent_task_id: TASK-005.08
priority: high
ordinal: 20000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the pretraining dataset consume document-level JSONL and emit document metadata.

Depends on TASK-005.08.01.

Current problem:
- PretrainDataset reads physical lines from .txt files.
- It mixes by byte-weighted line sampling.
- It does not emit document boundaries.
- It cannot give the model doc_id or position-reset information.

Outcome:
- PretrainDataset reads data/corpus/{split}/*.jsonl plus manifest.json.
- Each JSONL row is one document.
- The tokenizer encodes document.text with internal newlines preserved.
- The dataset emits im_start and im_end around each document.
- The dataset yields batches of shape (B, T) for input, target, and doc_ids.
- doc_ids increases at every document boundary.
- Domain mixing uses token counts from the manifest, not byte size.
- The scheduler samples the active domain that is currently behind its target token share.
- The dataset exposes estimated_steps() for auto LR schedule horizon.

Files to modify:
- src/kestrel/data/pretrain_dataset.py
- tests/test_pretrain_dataset.py
- configs/kestrel/50m/pretrain.yaml if dataset input format changes

Design decisions:
- Use manifest target fractions when present.
- If target fractions are absent, use measured token_count fractions.
- No multi-epoch / no rewind by default.
- If a domain quota or file is exhausted, remove it and continue with remaining domains.
- Keep context_length packing, but track doc_id across packed sequences.

Acceptance targets:
- A multi-line document produces one doc_id, not multiple doc_ids.
- Batches from a multi-domain directory contain more than one doc_id over a short run.
- Token-share test with skewed manifest token counts stays within 5 percentage points over at least 10k scheduled documents.
- estimated_steps() matches total manifest tokens divided by batch_size * seq_len, rounded down.
- make check is green.

Reference: doc-003.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Multi-line document produces one doc_id, not multiple doc_ids
- [ ] #2 Batches from a multi-domain directory contain more than one doc_id over a short run
- [ ] #3 Token-share test with skewed manifest token counts stays within 5 percentage points over at least 10k scheduled documents
- [ ] #4 estimated_steps matches total manifest tokens divided by batch_size * seq_len, rounded down
- [ ] #5 make check is green
<!-- AC:END -->
