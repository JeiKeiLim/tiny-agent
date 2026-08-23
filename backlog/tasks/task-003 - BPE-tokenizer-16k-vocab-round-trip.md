---
id: TASK-003
title: BPE tokenizer (16k vocab)
status: To Do
assignee: []
created_date: '2026-08-21 06:44'
updated_date: '2026-08-21 07:15'
labels: []
milestone: m-0
dependencies:
  - TASK-001
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A working byte-level BPE tokenizer (16k vocab, configurable) shared by both model sizes, trained on a representative sample of the target domain. Steps (sub-tasks): prepare training data -> train -> verify.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tokenizer trains on a text sample and saves to disk
- [ ] #2 encode -> decode round-trips arbitrary text losslessly (byte-level)
- [ ] #3 Vocab size is configurable (default 16384)
<!-- AC:END -->
