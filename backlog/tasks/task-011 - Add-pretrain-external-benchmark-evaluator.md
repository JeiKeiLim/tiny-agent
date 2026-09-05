---
id: TASK-011
title: Add pretrain external benchmark evaluator
status: Done
assignee: []
created_date: '2026-09-05 00:50'
updated_date: '2026-09-05 00:59'
labels:
  - evaluation
  - pretraining
  - benchmarks
dependencies: []
references:
  - scripts/download_pretrain_eval_datasets.py
  - scripts/eval_pretrain.py
  - scripts/run_eval_sft.py
documentation:
  - >-
    backlog/docs/research/pretrain-evaluation/doc-004 -
    Pretrain-Evaluation-Research.md
modified_files:
  - src/kestrel/eval/pretrain_benchmarks.py
  - scripts/run_eval_pretrain_benchmarks.py
  - tests/eval/test_pretrain_benchmarks.py
  - README.md
  - pyproject.toml
  - >-
    backlog/docs/research/pretrain-evaluation/doc-004 -
    Pretrain-Evaluation-Research.md
priority: medium
ordinal: 60000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a read-only evaluator for Kestrel pretrained checkpoints on external benchmark datasets.

The evaluator should use the raw evaluation files produced by scripts/download_pretrain_eval_datasets.py. It must not require network access, must not download training splits, must not unpack archives, and must not hard-code any user-specific dataset path.

The goal is a publishable pretrain scorecard with two tracks:
- language-modeling BPB on fixed external text sets
- zero-shot multiple-choice accuracy on common-sense/factual tasks

This is separate from scripts/eval_pretrain.py, which evaluates corpus validation loss, and scripts/run_eval_sft.py, which evaluates SFT/tool-calling behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI script scripts/run_eval_pretrain_benchmarks.py loads a pretrain checkpoint using an existing pretrain config and evaluates local benchmark datasets from --data-dir
- [x] #2 Supports selecting datasets with --only, skipping large sets with --skip-large, capping LM tokens with --max-tokens, capping examples with --max-examples, and writing JSON output
- [x] #3 Reports BPB, loss, perplexity, bits/token, tokens, and bytes for language-modeling datasets: wikitext2, wikitext103, c4_en_validation, pile_test, and lambada as text
- [x] #4 Reports zero-shot acc and length-normalized acc for multiple-choice datasets: hellaswag, piqa, arc_easy, arc_challenge, winogrande, openbookqa, boolq, sciq, and mmlu
- [x] #5 Missing dataset directories are reported as missing instead of crashing when --allow-missing is used; by default missing selected datasets are an error
- [x] #6 Adds focused tests under tests/eval/test_pretrain_benchmarks.py covering local parquet/JSONL loading, BPB accumulation, multiple-choice loglikelihood, scorecard JSON, and CLI behavior
- [x] #7 Updates README with a short pretrain benchmark evaluation section using only a generic --data-dir /path/to/datasets placeholder
- [x] #8 make check passes
All checks passed!
uv run ruff format --check src tests scripts
93 files already formatted
uv run mypy src scripts
Success: no issues found in 60 source files
uv run pytest
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/limjk/Documents/GitHub/JeiKeiLim/tiny-agent
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0, anyio-4.14.2
collected 368 items

tests/data/test_chat.py ........                                         [  2%]
tests/data/test_sft_chat.py ...............                              [  6%]
tests/data/test_sft_dataset.py .................                         [ 10%]
tests/data/test_sft_internal_llm.py ........................             [ 17%]
tests/data/test_sft_mixture.py ..........                                [ 20%]
tests/data/test_sft_prepare_eval.py .....                                [ 21%]
tests/data/test_sft_prepare_gsm8k.py .....                               [ 22%]
tests/data/test_sft_prepare_public.py .............                      [ 26%]
tests/data/test_sft_prepare_tool.py ...                                  [ 27%]
tests/data/test_sft_public_tool.py ...............                       [ 31%]
tests/data/test_sft_schema.py ...........                                [ 34%]
tests/data/test_sft_tool_generator.py .........                          [ 36%]
tests/eval/test_sft_eval.py ..............                               [ 40%]
tests/test_config.py .....                                               [ 41%]
tests/test_corpus_builder.py ..............................              [ 50%]
tests/test_eval_pretrain.py ...................                          [ 55%]
tests/test_generate.py ........................                          [ 61%]
tests/test_model_cache.py ......                                         [ 63%]
tests/test_model_check.py ..                                             [ 63%]
tests/test_model_io.py ...                                               [ 64%]
tests/test_model_kestrel.py ..............                               [ 68%]
tests/test_pretrain.py ...........                                       [ 71%]
tests/test_pretrain_dataset.py ...........................               [ 78%]
tests/test_serve_dashboard.py ..............                             [ 82%]
tests/test_tokenizer_config.py ........                                  [ 84%]
tests/test_tokenizer_data_config.py .....                                [ 86%]
tests/test_tokenizer_train.py ..                                         [ 86%]
tests/test_tokenizer_verify.py .......                                   [ 88%]
tests/test_tokenizer_visualize.py ......                                 [ 90%]
tests/test_trainer.py .....................                              [ 95%]
tests/tools/test_schema_sampler.py .....                                 [ 97%]
tests/train/test_sft.py ..........                                       [100%]

============================= 368 passed in 21.16s ============================= passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/eval/pretrain_benchmarks.py with dataclasses and evaluation helpers.
2. Add local raw-file readers for Parquet and JSONL/JSONL.gz using pyarrow and stdlib gzip/json.
3. Add BPB evaluation over text rows using the Kestrel tokenizer and model forward pass.
4. Add multiple-choice loglikelihood evaluation using tokenizer offsets to score continuation tokens only.
5. Add scripts/run_eval_pretrain_benchmarks.py CLI reusing PretrainConfig for model/tokenizer/seq_len/batch size.
6. Add tests with tiny local datasets and a tiny Kestrel model.
7. Update README and run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design decisions:
- Use raw eval files plus manifest.json from the download script; do not use datasets.load_dataset for evaluation to avoid extra Hugging Face cache growth.
- Use pyarrow.parquet.ParquetFile.iter_batches for Parquet to avoid loading large sets fully into memory.
- Use gzip transparently for .gz files.
- BPB is total_nats divided by ln(2) times total_bytes.
- Multiple-choice acc uses argmax total continuation logprob.
- Multiple-choice acc_norm uses argmax mean continuation logprob per token.
- SciQ rows expose correct_answer as text plus distractors; shuffle the four choices deterministically per row to avoid a fixed correct position.
- WinoGrande rows contain a blank; score the option plus the suffix after the blank given the prefix before the blank.
- LAMBADA is evaluated as LM text in this first version, not as final-word accuracy.

Implemented local raw-file evaluator. Validated against local downloaded benchmark files with tiny max_tokens/max_examples for all selected benchmarks. Fixed continuation offset handling so BPE tokens spanning the context/continuation boundary are scored. make check passed with 379 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added scripts/run_eval_pretrain_benchmarks.py and src/kestrel/eval/pretrain_benchmarks.py. The evaluator reads raw Parquet/JSONL benchmark files locally, reports BPB for LM sets and zero-shot acc/acc_norm for MCQ sets, writes a JSON scorecard, and is covered by tests/eval/test_pretrain_benchmarks.py. make check passes.
<!-- SECTION:FINAL_SUMMARY:END -->
