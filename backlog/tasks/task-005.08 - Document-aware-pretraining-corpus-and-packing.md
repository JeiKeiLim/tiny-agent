---
id: TASK-005.08
title: Document-aware pretraining corpus and packing
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-26 00:24'
updated_date: '2026-08-26 00:55'
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
- [x] #1 TASK-005.08.01 is Done
- [x] #2 TASK-005.08.02 is Done
- [x] #3 TASK-005.08.03 is Done
- [x] #4 data/corpus contains document-level JSONL plus manifest.json
- [x] #5 make check is green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Regenerated data/corpus from HF sources with document-level JSONL + manifest. Build results: web 912697307 bytes, code 107383862 bytes, jsonl 52075371 bytes. Train: 225716 docs, 963284380 bytes, estimated 240821095 tokens. Val: 25102 docs, 108872160 bytes, estimated 27218040 tokens. Web/code samples preserve internal newlines. make check green: 101 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Document-aware pretraining pipeline is complete: corpus builder emits JSONL + manifest, PretrainDataset emits doc_ids with token-aware mixing, model/trainer use same-document attention and position resets, and pretrain configs use auto dataset-exhaustion step horizons. The real data/corpus was regenerated from HF sources.
<!-- SECTION:FINAL_SUMMARY:END -->
