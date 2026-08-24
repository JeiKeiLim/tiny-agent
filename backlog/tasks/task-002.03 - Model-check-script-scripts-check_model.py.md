---
id: TASK-002.03
title: Model check script (scripts/check_model.py)
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-24 00:27'
updated_date: '2026-08-24 01:11'
labels: []
milestone: m-0
dependencies:
  - TASK-002.01
  - TASK-002.02
parent_task_id: TASK-002
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A standalone CLI in scripts/check_model.py (mirroring scripts/visualize_tokenizer.py) for manually smoke-testing a Kestrel model. Loads a model via model/io.py load(config, checkpoint=None) - random-init when no checkpoint, a trained checkpoint when provided - tokenizes a sample input with the BPE tokenizer, runs a forward pass, and prints: param count, logits shape, CE loss, and a few top-k argmax token IDs. Built on the load() factory so it evolves to load a trained model by pointing --checkpoint at a pretraining output (no code change).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 uv run python scripts/check_model.py --config configs/kestrel/50m/model.yaml loads a random-init 50M model, tokenizes a sample sentence, runs a forward pass, and prints param count (~50.7M), logits shape (B, T, 16384), and a finite CE loss
- [x] #2 --checkpoint <path> loads a model from a checkpoint instead of random init (tested by saving a random-init model then reloading it)
- [x] #3 The pure logic is tested in tests/test_model_check.py (imported from the script via pytest pythonpath); make check green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
New file scripts/check_model.py: logic as importable functions + main() + if __name__ == "__main__" (mirror scripts/visualize_tokenizer.py). Depends on TASK-002.01 (Kestrel + count_params) and TASK-002.02 (model/io.py load/save).

Core function check_model(config, checkpoint=None, text=...) -> ModelReport (dataclass: param_count, logits_shape, loss, top_token_ids, top_tokens). It: model = load(config, checkpoint) (random-init if checkpoint is None); n = count_params(model); tokenize text with the BPE tokenizer (real artifact checkpoints/tokenizer/tokenizer.json) into an int32 (B=1, T) input; logits = model(input); loss = mx.loss.cross_entropy(logits[0, :-1], input[0, 1:]).item(); top-k = top argmax ids of logits[0, -1] + their decoded strings; return the report.

main(): argparse --config (default configs/kestrel/50m/model.yaml), --checkpoint (default None), --text (default a sample sentence), --top-k (default 5); print the report.

Tests: tests/test_model_check.py - import from check_model (pytest pythonpath includes scripts/); use a TINY model config (small vocab/layers/hidden) + a tiny trained tokenizer (like test_tokenizer_visualize.py) so it runs fast; assert the report has the right logits shape, finite loss, and that the --checkpoint path round-trips (save a tiny model, reload, identical output).

mypy strict + MLX untyped: targeted type: ignore. Gate: make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decisions/gotchas:
- Follows the scripts/ + tests/ import pattern: logic lives in the script as importable functions; tests import it via pythonpath = [scripts] (see tests/test_tokenizer_visualize.py).
- Built on load(config, checkpoint=None) so loading a trained model later = pass --checkpoint <pretraining output dir>; no code change.
- Untrained model: loss ~ ln(vocab) ~ 9.7 (16k) and gibberish argmax tokens - expected, not a bug.
- Sample text is tokenized with our BPE tokenizer (real 16k artifact for manual runs; a tiny trained tokenizer in the test).
- reference: plan doc-001 §6 (entry points in scripts/), §14 (generate() core).

MLX 0.32.1 gotchas (plan assumed mx.loss.cross_entropy + mx.topk indices): (1) mx.loss does NOT exist — cross-entropy is mlx.nn.losses.cross_entropy(logits, targets, reduction="mean") (default reduction="none" returns per-element, not a scalar). (2) mx.topk(x, k) returns top-k VALUES sorted ascending, NOT indices — for top-k token ids in descending order use mx.argsort(-x)[:k]. (3) mx.sort/argsort have no descending= param — negate for descending. (4) mx.array.item() -> scalar (int|float|bool|complex) and .tolist() -> list_or_scalar are too broad for mypy strict — use cast(float, ...) / cast(list[int], ...). Design: report_from_model(model, tokenizer, text, top_k) is the pure testable core; check_model(config, checkpoint, text, top_k, tokenizer_path) builds the model via load() + loads the tokenizer then delegates. Top tokens decoded per-id via tokenizer.decode([id]) (byte-level BPE, single tokens may show the leading-space marker — fine for a smoke test). main() checks the tokenizer artifact exists and prints a "train it first" hint (mirrors visualize_tokenizer.py). Tests use a tiny model (vocab 400, 2 layers, 64 hidden) + tiny trained tokenizer (vocab 400) so they run fast.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented scripts/check_model.py: ModelReport dataclass (param_count, logits_shape, loss, top_token_ids, top_tokens), report_from_model (pure forward-pass core), check_model (load(config, checkpoint) + tokenize + forward), and main() CLI (--config/--checkpoint/--text/--top-k/--tokenizer). tests/test_model_check.py (2): random-init report (logits shape (1,T,vocab), finite loss, top-k ids in range) and checkpoint round-trip (save tiny model, reload via check_model, identical logits shape + top-k + loss). Verified AC#1 manually on the real 50M config: 50,675,200 params, logits (1,12,16384), CE 10.37, gibberish top tokens (expected untrained). make check green (mypy 25 files, 42 tests).
<!-- SECTION:FINAL_SUMMARY:END -->
