---
id: TASK-003.02
title: Train the BPE tokenizer
status: Done
assignee: []
created_date: '2026-08-21 07:15'
updated_date: '2026-08-23 02:02'
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
- [x] #1 Tokenizer trains on the prepared sample and saves the artifact to disk
- [x] #2 Vocab size is configurable (default 16384)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Train byte-level BPE tokenizer (16k vocab, configurable) with HF tokenizers on the prepared sample. New module src/kestrel/tokenizer/ (config.py strict Pydantic TokenizerConfig + train.py), config configs/tokenizer/train.yaml. Bake ChatML special tokens (im_start, im_end, im_system, im_user, im_assistant) plus tool-call tokens; output checkpoints/tokenizer/ (gitignored). Tests: config strictness + tiny-train integration. make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SPECIAL TOKENS FINDINGS (researched online per user request, not guessed): Qwen2.5 (our Track B base) and Mistral use the ChatML format and bake 5 tokens into the tokenizer: im_start, im_end, im_system, im_user, im_assistant. im_end doubles as the EOS/stop token. Each message is wrapped as im_start, role, newline, content, im_end. They also bake in 4 tool tokens: tool_call, tool_call_end, tool_response, tool_response_end. LLaMA uses a smaller set (a BOS, an end-of-turn token, and header tokens) with a partly plain-text chat format. DECISION: bake in the full set NOW at training time so their IDs are reserved in the 16k vocab, matching Qwen/Mistral. Do NOT defer to add_special_tokens later, and do NOT use just one token. Rationale: this repo builds a real modern agentic LLM at small scale, so we do not cut features; one token is not the modern standard. CONVENTIONS: run commands with uv run, not bare python (README fixed). make check now runs ruff format --check (gate fixed and committed). gitignore anchored to /data/ so it does not shadow src/kestrel/data/ (committed). STATE: TASK-003.01 (data prep) done and committed. TASK-003.02 (train tokenizer) in progress, plan and findings recorded, not yet implemented. Next: implement the tokenizer module.

IMPLEMENTED: src/kestrel/tokenizer/config.py (strict TokenizerConfig: vocab_size default 16384, train_dir, output_dir, eos_token, min_frequency, initial_alphabet, special_tokens; validators reject duplicates, eos not in specials, vocab too small) + train.py (HF tokenizers ByteLevelBPE: ByteLevel pre-tokenizer add_prefix_space=False, ByteLevel decoder, BpeTrainer with special_tokens + initial_alphabet, saves tokenizer.json) + configs/tokenizer/train.yaml (16k, 9 ChatML+tool specials, im_end=eos). Tests: test_tokenizer_config.py (strictness) + test_tokenizer_train.py (tiny-train integration: saves artifact, round-trips, fails w/o corpus). make check GREEN (lint+format+mypy+20 tests). TRAINED real 16k on data/tokenizer_train (web+code+jsonl ~1GB) -> checkpoints/tokenizer/tokenizer.json (1.1MB, vocab exactly 16384, specials at ids 0-8, lossless round-trip on EN/JSON/KR/emoji). NOTE: tokenizers 0.23.1 stubs are incomplete (untyped BpeTrainer ctor, no eos_token prop) -> 2 targeted type:ignore in train.py; eos is recorded via config.eos_token, not the tokenizer object.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented src/kestrel/tokenizer/ (strict TokenizerConfig + HF-tokenizers ByteLevelBPE train script), configs/tokenizer/train.yaml, and tests (config strictness + tiny-train integration). Trained the real 16k-vocab tokenizer on the ~1GB prepared sample -> checkpoints/tokenizer/tokenizer.json (specials at ids 0-8, lossless round-trips). make check green; both ACs verified.
<!-- SECTION:FINAL_SUMMARY:END -->
