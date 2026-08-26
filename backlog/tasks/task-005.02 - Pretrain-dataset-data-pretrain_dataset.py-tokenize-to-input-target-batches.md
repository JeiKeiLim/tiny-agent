---
id: TASK-005.02
title: >-
  Pretrain dataset (data/pretrain_dataset.py) - tokenize to (input, target)
  batches
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-24 01:55'
updated_date: '2026-08-26 00:28'
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
- [x] #1 Given a small text + a tiny trained tokenizer, yields (input, target) batches of shape (batch_size, context_length)
- [x] #2 next-token shift is correct: target[t] == input[t+1] for every position (assert on a known small case)
- [x] #3 total_tokens cap is respected (iteration stops after the cap)
- [x] #4 tests/test_pretrain_dataset.py builds a TINY tokenizer in-test (pattern in tests/test_model_check.py) so it does NOT depend on the gitignored checkpoints/tokenizer/tokenizer.json; make check green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1) src/kestrel/data/pretrain_dataset.py: PretrainDatasetConfig (strict Pydantic: input file-or-dir, tokenizer_path, context_length=2048, batch_size=8, total_tokens=None, seed=0) + PretrainDataset (iterable). 2) __iter__: Tokenizer.from_file; resolve files (dir -> sorted *.txt, shuffled by seed); stream lines -> encode(add_special_tokens=False).ids -> buffer; cut T-windows; input=seq, target=seq[1:]+[seq[-1]] (target[t]==input[t+1] for t<T-1, last dropped - matches check_model loss cross_entropy(logits[:,:-1], input[:,1:])); yield (B,T) int32 batches; stop at total_tokens cap; full batches only (trailing partial dropped). No line/document separators (raw packing). 3) tests/test_pretrain_dataset.py: tiny in-test tokenizer (test_model_check pattern, no gitignored dep); AC1 batch shape (B,T) int32; AC2 shift tgt[:,:-1]==inp[:,1:] + known-case inp[0]==encode(text).ids[:T]; AC3 cap (16 tokens -> exactly 1 batch); + config validation, dir input, partial-dropped. 4) make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented pretrain_dataset.py (PretrainDatasetConfig + PretrainDataset iterable) + test_pretrain_dataset.py (6 tests, tiny in-test tokenizer). make check green (57 tests). Real-usage smoke: real tokenizer + data/corpus/ dir at context_length=2048 -> batch (2,2048) int32, shift tgt[:,:-1]==inp[:,1:] holds. Shift convention matches check_model loss (cross_entropy(logits[:,:-1], input[:,1:])). Full batches only (trailing partial dropped); no line/doc separators (raw packing, per plan).

Follow-up: original PretrainDataset tokenizes physical lines and does not emit document boundaries. TASK-005.08.02 tracks JSONL document consumption, token-aware mixing, and doc_ids.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built the pretraining dataset (data/pretrain_dataset.py): PretrainDatasetConfig (strict Pydantic: input file-or-dir, tokenizer_path, context_length=2048, batch_size=8, total_tokens=None, seed=0) + PretrainDataset (streaming iterable). Streams corpus text line-by-line, tokenizes (add_special_tokens=False), packs into context_length sequences, yields (input, target) int32 batches of shape (batch_size, context_length). target[t]==input[t+1] for t<T-1 (last dropped) - matches scripts/check_model.py loss cross_entropy(logits[:,:-1], input[:,1:]). Full batches only (trailing partial dropped); no line/doc separators (raw packing). tests/test_pretrain_dataset.py (6 tests, tiny in-test tokenizer, no gitignored dep). Verified: make check green + real smoke (real tokenizer + data/corpus at context_length=2048 -> batch (2,2048) int32, shift holds). Post-split (005.01): point input at data/corpus/train (training) or data/corpus/val (validation).
<!-- SECTION:FINAL_SUMMARY:END -->
