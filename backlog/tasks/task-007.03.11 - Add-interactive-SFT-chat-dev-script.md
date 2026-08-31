---
id: TASK-007.03.11
title: Add interactive SFT chat dev script
status: Done
assignee:
  - '@Jongkuk Lim'
created_date: '2026-08-31 06:57'
updated_date: '2026-08-31 07:00'
labels:
  - sft
  - dev
  - eval
milestone: m-2
dependencies:
  - TASK-007.03.01
parent_task_id: TASK-007.03
priority: medium
ordinal: 51000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a manual multi-turn chat dev tool for inspecting SFT checkpoints.

This is separate from scripts/check_model.py, which remains a raw continuation smoke tool. The chat script should use the same SFT chat renderer used during training, but without tool calling for now.

Files:
- scripts/chat_sft.py
- src/kestrel/model/chat.py or src/kestrel/data/chat.py for reusable prompt-building/response-extraction helpers
- tests for the helper functions

Scope:
- Load model config, checkpoint, and tokenizer from CLI args.
- Maintain a multi-turn user/assistant history.
- Optional system prompt.
- Render prompts with render_sft() using only system/user/assistant messages.
- Generate assistant text with the existing generate() function.
- Extract and print only the assistant content, not raw role markers.
- Support max tokens, temperature, repetition penalty, and simple exit commands.
- No tool calling in this first version.

Acceptance:
- The script can chat for multiple turns against a checkpoint.
- Prompts use the same chat template as SFT training.
- Assistant response extraction removes role structure.
- Unit tests cover prompt building and response extraction without requiring a real checkpoint.
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 scripts/chat_sft.py supports multi-turn chat against a model checkpoint
- [x] #2 chat prompts are rendered with the same SFT renderer used for training
- [x] #3 assistant response extraction removes role structure before printing
- [x] #4 tool calling is not exposed in this first version
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/data/chat.py with build_chat_prompt(messages, tokenizer), extract_assistant_content(text), and mask_special_tokens(text).
2. build_chat_prompt validates system/user/assistant messages with SFTRow and renders using render_sft.
3. extract_assistant_content removes role structure by locating the assistant marker and cutting at any special token boundary.
4. Add scripts/chat_sft.py interactive CLI with --config, --checkpoint, --tokenizer, --system, --max-tokens, --temp, and --repetition-penalty.
5. Add tests/data/test_chat.py covering prompt construction, multi-turn history, response extraction, and special-token masking using local fixtures and the tiny tokenizer.
6. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added src/kestrel/data/chat.py with build_chat_prompt, extract_assistant_content, and mask_special_tokens.
Added scripts/chat_sft.py as an interactive multi-turn chat CLI for SFT checkpoints.
The chat script uses the same render_sft() template as SFT training, supports optional system prompt, and does not expose tool calling.
Assistant responses are extracted before printing, and any special tokens are masked as safe placeholders.
Added tests/data/test_chat.py and a tiny_sft_tokenizer_obj fixture in tests/data/conftest.py.
Manually verified multi-turn behavior with piped input against checkpoints/sft/50m-smoke-assistant-gsm8k/best.
Validation: make check passed with 236 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a proper interactive SFT chat dev script with reusable chat prompt/response helpers, tests, and safe display masking; tool calling remains out of scope.
<!-- SECTION:FINAL_SUMMARY:END -->
