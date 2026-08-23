---
id: TASK-003
title: BPE tokenizer (16k vocab)
status: Done
assignee: []
created_date: '2026-08-21 06:44'
updated_date: '2026-08-23 22:48'
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
- [x] #1 Tokenizer trains on a text sample and saves to disk
- [x] #2 encode -> decode round-trips arbitrary text losslessly (byte-level)
- [x] #3 Vocab size is configurable (default 16384)
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All three subtasks done: 003.01 prepared the training sample, 003.02 trained the 16k byte-level BPE (artifact at checkpoints/tokenizer/tokenizer.json, vocab exactly 16384, 9 specials), 003.03 verified lossless round-trip on all valid UTF-8 text + 235/256 raw-byte coverage (21 unobserved C0 controls documented). make check green.
<!-- SECTION:FINAL_SUMMARY:END -->
