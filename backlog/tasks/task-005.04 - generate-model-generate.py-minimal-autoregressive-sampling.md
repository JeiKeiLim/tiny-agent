---
id: TASK-005.04
title: generate() (model/generate.py) - minimal autoregressive sampling
status: Done
assignee:
  - 7477cb22-9a4d-4bfc-9c19-64c3784d2b3a
created_date: '2026-08-24 01:56'
updated_date: '2026-08-24 08:12'
labels: []
milestone: m-1
dependencies: []
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
parent_task_id: TASK-005
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minimal autoregressive text generation, needed to verify the pretrain validation run produces coherent text (and reused later by eval/serve/agent). Given a model + tokenizer + prompt, produce text by repeatedly predicting the next token. This is the SAME core generate() the plan assigns to serve/ (doc-001 §14); we place it in model/generate.py now (model inference, no server) so serve/ can wrap it later.

Files to create:
- src/kestrel/model/generate.py
- tests/test_generate.py

Design (generate() fn):
- signature: generate(model, tokenizer, prompt: str, max_tokens: int, temp: float = 0.0, stop_token_id: int | None = None) -> str
- encode prompt -> token ids.
- loop up to max_tokens:
  - forward the current sequence; use ONLY the last position's logits (logits[:, -1, :]) for the next token.
  - temp == 0 -> next = argmax(logits) (greedy, deterministic); else sample from softmax(logits / temp).
  - if next == stop_token_id (EOS): break; else append next.
- decode the generated ids (excluding the prompt) -> str.
- stop_token_id defaults to the tokenizer EOS id (im_end doubles as EOS per task-003.02).

Quantitative targets:
- produces at most max_tokens new tokens; stops early on EOS; temp=0 is deterministic (same output twice); returns a str.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generate() from a prompt produces <= max_tokens new tokens and returns a str
- [x] #2 temp=0 is deterministic: two calls with the same prompt/model return identical output
- [x] #3 stops on the stop/EOS token before reaching max_tokens (craft a tiny case where the model emits EOS)
- [x] #4 tests/test_generate.py uses a TINY model + TINY tokenizer (in-test) so it runs fast and does not depend on gitignored artifacts; make check green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented (2026-08-24): src/kestrel/model/generate.py (generate(model, tokenizer, prompt, max_tokens, temp=0.0, stop_token_id=None) -> str) + tests/test_generate.py (4 tests). make check green (72 tests, +4). Real smoke (real 16384-vocab tokenizer + tiny untrained Kestrel): returns str, exactly max_tokens new tokens, deterministic at temp=0.

DESIGN: model(x) returns logits (1,T,V); each step reads last-position logits model(x)[0,-1,:]; temp==0 -> argmax (greedy/deterministic), temp>0 -> mx.random.categorical(softmax(logits/temp)); stops on stop_token_id (default tokenizer.token_to_id('im_end')); no KV cache (re-runs full sequence each step - minimal, fine for 50M validation; a cached fast path can go in serve/). model typed Callable[[mx.array], mx.array] (matches Kestrel.__call__).

TESTS: TINY in-test WordLevel tokenizer (5 tokens incl im_end) + TINY Kestrel (vocab 5, 1 layer). Scripted stub model used for exact-count + EOS-stop tests (controlled token sequence); real tiny model for determinism + sampling. GOTCHA: counting generated tokens by re-encoding the decoded string is UNSTABLE for [UNK] (WordLevel: encode('[UNK]')==[0,0,0]), so exact-count tests use a scripted stub emitting non-UNK tokens. mypy: .item() returns int|float|complex -> cast(int, ...) for the token id.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built src/kestrel/model/generate.py (generate(model, tokenizer, prompt, max_tokens, temp=0.0, stop_token_id=None) -> str) + tests/test_generate.py (4 tests). make check green (72 tests, +4 generate). DESIGN: model(x) returns logits (1,T,V); each step reads last-position logits model(x)[0,-1,:]; temp==0 -> argmax (greedy/deterministic), temp>0 -> mx.random.categorical(softmax(logits/temp)); stops on stop_token_id (default tokenizer.token_to_id('im_end')); no KV cache (minimal, fine for 50M validation; a cached fast path can go in serve/). model typed Callable[[mx.array], mx.array] (matches Kestrel.__call__). TESTS: TINY in-test WordLevel tokenizer (5 tokens incl im_end) + TINY Kestrel (vocab 5, 1 layer); scripted stub model for exact-count + EOS-stop tests; real tiny model for determinism + sampling. GOTCHA: counting tokens by re-encoding the decoded string is unstable for [UNK] (WordLevel: encode('[UNK]')==[0,0,0]), so exact-count tests use a scripted stub emitting non-UNK tokens. mypy: .item() returns int|float|complex -> cast(int, ...) for the token id.
<!-- SECTION:FINAL_SUMMARY:END -->
