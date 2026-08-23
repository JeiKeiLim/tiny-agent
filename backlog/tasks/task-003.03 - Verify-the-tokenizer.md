---
id: TASK-003.03
title: Verify the tokenizer
status: To Do
assignee: []
created_date: '2026-08-21 07:15'
labels: []
milestone: m-0
dependencies:
  - TASK-003.02
parent_task_id: TASK-003
ordinal: 2300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the trained tokenizer: lossless round-trip on arbitrary text, correct vocab size, byte-coverage, and sanity on sample text. Add a test to the suite (test the code, not data values).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 encode -> decode round-trips arbitrary text losslessly (byte-level)
- [ ] #2 Vocab size matches the configured value (16k)
- [ ] #3 A test in the suite covers round-trip + vocab size
<!-- AC:END -->
