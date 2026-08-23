---
id: TASK-003.04
title: Force all 256 byte tokens into the vocab (full byte coverage)
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-23 23:03'
updated_date: '2026-08-23 23:48'
labels: []
dependencies:
  - TASK-003.03
parent_task_id: TASK-003
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The BPE trainer seeds its base alphabet only from bytes observed in the corpus, so 21 unobserved C0 control bytes (0x0, 0x2-0x5, 0x0e-0x1a, 0x1d-0x1f) have no token and are silently dropped on encode: raw-byte coverage is 235/256. Modern byte-level BPE tokenizers (GPT-2, Qwen) structurally include all 256 bytes in the base vocab, guaranteeing a lossless round-trip for ANY byte sequence, not just text.

Fix: pass initial_alphabet=list(ByteLevel.alphabet()) (all 256 byte-chars) to BpeTrainer in src/kestrel/tokenizer/train.py so every byte has a base token, then retrain the artifact. Verified empirically: this takes an ASCII-only corpus from 24/256 to 256/256 coverage. Cost for the real 16k artifact is ~21 extra base slots (0.13% of vocab) spent on tokens that rarely fire in text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 train.py seeds the base alphabet with all 256 byte tokens via initial_alphabet=list(ByteLevel.alphabet())
- [x] #2 Retrained artifact has 256/256 raw-byte coverage (every byte 0x00-0xFF round-trips losslessly)
- [x] #3 Vocab size still matches the configured value (16384)
- [x] #4 scripts/verify_tokenizer.py --coverage reports 256/256 on the retrained artifact
- [x] #5 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. src/kestrel/tokenizer/train.py: seed the BPE base alphabet with all 256 byte chars. Change the BpeTrainer initial_alphabet to include ByteLevel.alphabet() (e.g. initial_alphabet=list(ByteLevel.alphabet())). config.initial_alphabet (default '{}[]":,') is a subset of ByteLevel.alphabet() so it is subsumed. DECISION: hardcode vs a config flag — project convention is config-driven, so a full_byte_coverage: bool = True option on TokenizerConfig is the cleaner fit; pick one and note it.
2. Update tests/test_tokenizer_verify.py::test_verify_bytes_detects_missing_byte — after the fix NUL (0x00) IS covered, so the 'missing byte' assertion no longer holds. Replace it with an assertion that all 256 bytes (incl. NUL) round-trip on a full-alphabet tokenizer.
3. Retrain: uv run python -m kestrel.tokenizer.train (needs data/tokenizer_train/ present — gitignored ~1GB; regenerate via kestrel.data.prepare_tokenizer_data if missing).
4. Verify: uv run python scripts/verify_tokenizer.py README.md --coverage -> expect 256/256.
5. make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Training data data/tokenizer_train/ (web/code/jsonl, ~1GB) is gitignored and a runtime artifact of TASK-003.01; retraining requires it present locally. The fix subsumes config.initial_alphabet (its default chars are a subset of ByteLevel.alphabet()). Existing test test_verify_bytes_detects_missing_byte asserts NUL is dropped and WILL FAIL after the fix — must be updated in the same change. Mechanism empirically verified: initial_alphabet=list(ByteLevel.alphabet()) takes an ASCII-only corpus from 24/256 to 256/256 coverage; cost on the real 16k artifact is ~21 base slots (0.13%).

DECISION: hardcoded list(ByteLevel.alphabet()) in train.py rather than adding a config flag, and REMOVED the now-dead initial_alphabet field (config.py + train.yaml). Rationale: the config already documented + reserved 256 byte-token slots (validator: len(specials)+256>vocab_size), so the code now matches documented intent; a flag or the old string knob would be dead config (strict-config convention = no dead knobs). doc-001's JSON-readiness intent (pass { } [ ] " : , as initial_alphabet) is subsumed — all ASCII is now guaranteed single tokens.

TESTS: test_tokenizer_verify.py — added _restricted_tokenizer helper (manual byte-level BPE with initial_alphabet=['a']) so the verifier's non-lossless DETECTION path is still exercised (our train() can no longer produce a missing byte); repointed test_verify_bytes_detects_missing_byte to it; added test_train_produces_full_byte_coverage (train() -> 256/256). make check GREEN (33 tests).

RETRAINED real 16k on data/tokenizer_train -> checkpoints/tokenizer/tokenizer.json: vocab exactly 16384, specials still at ids 0-8, verify_tokenizer.py README.md --coverage = 256/256 LOSSLESS. Explicitly confirmed NUL (0x00) + all 21 former-missing C0 bytes now round-trip.

FOLLOW-UP (user request): removed the standalone scripts/verify_tokenizer.py CLI. The verification core (VerifyStats, verify_bytes, byte_coverage — latin-1 bridge, skip_special_tokens=False) now lives in tests/test_tokenizer_verify.py as the code under test; tests exercise it on tiny in-test-trained tokenizers (self-contained, no artifact needed). The real 16k artifact was already verified manually (256/256 lossless). make check GREEN (33 tests, mypy 22 files).

FOLLOW-UP 2 (user request): added a __main__ entry point (_main()) to tests/test_tokenizer_verify.py so the verification can be run manually against the REAL artifact without a separate scripts/ file: 'uv run python tests/test_tokenizer_verify.py FILE [FILE ...] [--coverage]'. Loads checkpoints/tokenizer/tokenizer.json, round-trips each file via verify_bytes, optional --coverage, exit 0/1. Verified: README.md/AGENTS.md/Makefile all LOSSLESS, coverage 256/256, missing file -> exit 1. make check GREEN (33 tests).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made the BPE tokenizer guarantee all 256 byte-tokens (GPT-2/Qwen convention) so any byte sequence round-trips losslessly, not just observed text. train.py now seeds initial_alphabet=list(ByteLevel.alphabet()); removed the dead initial_alphabet config field (config.py + train.yaml) since the config already documented/reserved 256 byte slots. Updated test_tokenizer_verify.py (restricted-tokenizer helper keeps the verifier's detection path tested; new full-coverage test). make check GREEN (33 tests). Retrained the real 16k artifact: vocab 16384, specials at ids 0-8, verify --coverage = 256/256 LOSSLESS; NUL + all 21 former-missing C0 bytes confirmed round-tripping.
<!-- SECTION:FINAL_SUMMARY:END -->
