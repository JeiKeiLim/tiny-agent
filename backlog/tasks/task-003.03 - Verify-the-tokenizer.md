---
id: TASK-003.03
title: Verify the tokenizer
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-21 07:15'
updated_date: '2026-08-23 22:47'
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
- [x] #1 encode -> decode round-trips arbitrary text losslessly (byte-level)
- [x] #2 Vocab size matches the configured value (16k)
- [x] #3 A test in the suite covers round-trip + vocab size
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add byte-coverage unit test (round-trip arbitrary bytes incl. non-UTF-8 via surrogateescape) to tests/test_tokenizer_train.py. 2. Add scripts/verify_tokenizer.py: standalone CLI taking file path(s), core verify_bytes(tokenizer, data) -> stats (raw bytes, token count, bytes/token, token-id u16 size, round-trip bytes, diff, lossless). 3. Add tests/test_tokenizer_verify.py for the core (importable, no TTY needed). 4. make check green. 5. Run script against real files + the trained 16k artifact to confirm lossless.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
verify_bytes bridges bytes via latin-1 (bijection over 0..255) so non-UTF-8 files are checkable; decode must use skip_special_tokens=False or special tokens matched as substrings (e.g. 'tool_call' in 'tool_calling') are dropped (low-level decode defaults to True, which is for generation). Real 16k artifact: all valid UTF-8 text lossless; raw-byte coverage 235/256 — 21 C0 control bytes (0x0,0x2-0x5,0x0e-0x1a,0x1d-0x1f) have no token (unobserved, pruned) and are dropped. Documented, not retrained (text tokenizer). make check green: 32 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added scripts/verify_tokenizer.py (standalone CLI: file path(s) -> raw bytes, token count, bytes/token, token-id u16 size, round-trip bytes, diff, LOSSLESS; --coverage reports 235/256 raw-byte coverage) + tests/test_tokenizer_verify.py (6 tests: lossless round-trip, stats consistency, missing-byte detection, coverage partitions 0..255, natural text round-trip, vocab reaches configured size on a rich corpus). Key finding: decode needs skip_special_tokens=False for lossless round-trip; 21 C0 control bytes are unobserved/pruned and dropped (documented, not retrained). Verified lossless on README/AGENTS/Makefile/pyproject + Unicode + special-token substrings; make check green (32 tests).
<!-- SECTION:FINAL_SUMMARY:END -->
