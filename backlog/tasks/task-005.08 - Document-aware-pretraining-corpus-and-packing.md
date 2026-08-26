---
id: TASK-005.08
title: Document-aware pretraining corpus and packing
status: To Do
assignee: []
created_date: '2026-08-26 00:24'
updated_date: '2026-08-26 00:27'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies: []
documentation:
  - doc-003
parent_task_id: TASK-005
priority: high
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parent task for fixing the corpus/packing root cause found after the 50M/150M M1 runs.

Root cause: the current corpus pipeline writes HF document text as physical text lines. Internal newlines in web pages and code files become physical line breaks, so the training corpus loses document structure. The current web/code corpus is therefore line-fragment data, not document data.

Goal: make the pretraining pipeline use document-level data end to end:
1. corpus builder emits document-level JSONL plus manifest
2. pretrain dataset reads documents, mixes by tokens, and emits doc_ids
3. model/trainer use document-aware attention, position resets, and auto step horizon

Reference: doc-003.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-005.08.01 is Done
- [ ] #2 TASK-005.08.02 is Done
- [ ] #3 TASK-005.08.03 is Done
- [ ] #4 data/corpus contains document-level JSONL plus manifest.json
- [ ] #5 make check is green
<!-- AC:END -->
