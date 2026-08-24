---
id: TASK-002.03
title: Model check script (scripts/check_model.py)
status: To Do
assignee: []
created_date: '2026-08-24 00:27'
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
- [ ] #1 uv run python scripts/check_model.py --config configs/kestrel/50m/model.yaml loads a random-init 50M model, tokenizes a sample sentence, runs a forward pass, and prints param count (~50.7M), logits shape (B, T, 16384), and a finite CE loss
- [ ] #2 --checkpoint <path> loads a model from a checkpoint instead of random init (tested by saving a random-init model then reloading it)
- [ ] #3 The pure logic is tested in tests/test_model_check.py (imported from the script via pytest pythonpath); make check green
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
<!-- SECTION:NOTES:END -->
