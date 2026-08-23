---
id: TASK-003.02
title: Train the BPE tokenizer
status: To Do
assignee: []
created_date: '2026-08-21 07:15'
labels: []
milestone: m-0
dependencies:
  - TASK-003.01
parent_task_id: TASK-003
ordinal: 2200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Train a byte-level BPE tokenizer (16k vocab, configurable) using HuggingFace tokenizers on the prepared sample. Provide the training script + the saved tokenizer artifact, shared by both model sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tokenizer trains on the prepared sample and saves the artifact to disk
- [ ] #2 Vocab size is configurable (default 16384)
<!-- AC:END -->
