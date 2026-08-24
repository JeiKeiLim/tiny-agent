---
id: TASK-005.02
title: >-
  Pretrain dataset (data/pretrain_dataset.py) - tokenize to (input, target)
  batches
status: To Do
assignee: []
created_date: '2026-08-24 01:55'
labels: []
milestone: m-1
dependencies: []
references:
  - backlog/docs/doc-001 - Agentic-SLM-Training-Pipeline-—-Project-Plan.md
parent_task_id: TASK-005
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tokenize the corpus text into (input, target) batches for pretraining. Reads raw text (corpus builder output), tokenizes with the trained BPE tokenizer, packs tokens into fixed-length sequences of context_length, and yields (input_ids, target_ids) batches where target is input shifted left by one (next-token prediction).

Files to create:
- src/kestrel/data/pretrain_dataset.py
- tests/test_pretrain_dataset.py

Design (a PretrainDataset / iterator):
- inputs: text file path(s) or corpus output dir, tokenizer path, context_length (2048), batch_size, total_tokens (cap), seed.
- read text line by line (streaming, memory-safe), tokenize each line, accumulate tokens into a buffer, cut into sequences of length context_length.
- per sequence: input = seq, target = seq shifted left by 1 (target[t] == input[t+1]); the final position has no valid target (drop or mask it).
- yield batches of shape (B, T); stop once total_tokens of sequences are produced.
- tokenize via the HF tokenizers API (tokenizer.json) -> list[int].

Quantitative targets:
- for N input tokens, yields floor(N / context_length) full sequences (last partial dropped).
- target[t] == input[t+1] for every valid t.
- batch shape == (batch_size, context_length).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Given a small text + a tiny trained tokenizer, yields (input, target) batches of shape (batch_size, context_length)
- [ ] #2 next-token shift is correct: target[t] == input[t+1] for every position (assert on a known small case)
- [ ] #3 total_tokens cap is respected (iteration stops after the cap)
- [ ] #4 tests/test_pretrain_dataset.py builds a TINY tokenizer in-test (pattern in tests/test_model_check.py) so it does NOT depend on the gitignored checkpoints/tokenizer/tokenizer.json; make check green
<!-- AC:END -->
