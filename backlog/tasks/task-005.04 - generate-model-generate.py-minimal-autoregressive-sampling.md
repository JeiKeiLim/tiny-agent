---
id: TASK-005.04
title: generate() (model/generate.py) - minimal autoregressive sampling
status: To Do
assignee: []
created_date: '2026-08-24 01:56'
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
- [ ] #1 generate() from a prompt produces <= max_tokens new tokens and returns a str
- [ ] #2 temp=0 is deterministic: two calls with the same prompt/model return identical output
- [ ] #3 stops on the stop/EOS token before reaching max_tokens (craft a tiny case where the model emits EOS)
- [ ] #4 tests/test_generate.py uses a TINY model + TINY tokenizer (in-test) so it runs fast and does not depend on gitignored artifacts; make check green
<!-- AC:END -->
