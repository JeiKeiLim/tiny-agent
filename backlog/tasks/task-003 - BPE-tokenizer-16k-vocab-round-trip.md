---
id: TASK-003
title: BPE tokenizer (16k vocab) + round-trip
status: To Do
assignee: []
created_date: '2026-08-21 06:44'
labels: []
milestone: m-0
dependencies:
  - TASK-001
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Train a byte-level BPE tokenizer per §7 using HuggingFace tokenizers: 16k vocab (configurable), shared by both model sizes. Provide a train script + the saved tokenizer artifact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tokenizer trains on a text sample and saves to disk
- [ ] #2 encode -> decode round-trips arbitrary text losslessly (byte-level)
- [ ] #3 Vocab size is configurable (default 16384)
<!-- AC:END -->
