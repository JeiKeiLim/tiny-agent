---
id: doc-003
title: Pretraining Dataset Packing and Token-Aware Mixing Research
type: guide
created_date: '2026-08-25 23:38'
updated_date: '2026-08-26 00:28'
tags:
  - research
  - data
  - pretraining
  - packing
---
# doc-003 - Pretraining Dataset Packing and Token-Aware Mixing Research

Status: research + fix plan
Date: 2026-08-26
Related tasks: TASK-005.08, TASK-005.08.01, TASK-005.08.02, TASK-005.08.03

## Root cause found after the M1 runs

The weak 50M/150M generation is caused by three compounding issues:

1. Undertraining.
   - 50M saw about 236M tokens, roughly 4.66 tokens/param.
   - 150M at the 40k step cap sees about 164M tokens, roughly 1.11 tokens/param.
   - This is far below the budget needed for coherent open-ended generation.

2. Byte-weighted line mixing.
   - TASK-005.02.01 mixed by file byte size, but consumed physical lines.
   - JSONL lines are long, code lines are short, so byte weights do not equal token or document weights.
   - The 50M run became code-only in the final steps.

3. Corpus flattening.
   - The original HF rows are documents.
   - Our pipeline wrote each document as `text + "\n"`.
   - Internal newlines became physical file lines.
   - The corpus builder then treated each physical line as a document.
   - Current `data/corpus/train/web.txt` and `code.txt` are therefore line fragments, not documents.

The third issue is the main structural blocker. The first two are budget and scheduler issues.

## Why this matters

A small decoder-only model can learn local statistics from line fragments, but it does not learn clean document structure. Code files and web pages contain newlines, indentation, blank lines, function boundaries, paragraph boundaries, and long-range dependencies. Flattening those into independent physical lines destroys the unit the model is supposed to learn from.

This also explains why simple newline insertion is not enough. The current corpus no longer knows where one document ended and the next began.

## Standard practice

Research from FineWeb, RedPajama, Dolma, SmolLM, The Pile, The Stack, StarCoder, PaLM, Llama 3.1, Megatron-LM, and TRL points to the same pattern:

- Text corpora use one JSONL or Parquet row per document.
- Internal newlines are preserved inside the document text.
- Code corpora use whole files as documents, not physical lines.
- Packing uses document boundaries, usually EOS/EOD or document IDs.
- Attention and position encoding should not cross document boundaries.
- Loss should not train a token to predict the start of the next unrelated document.
- Domain mixing should be token-aware, not byte-weighted line sampling.

## Current Kestrel defect

The flattening happens in two places:

```python
line = text + "\n"
f.write(line)
```

in `src/kestrel/data/prepare_tokenizer_data.py`, and the same physical-line assumption in `src/kestrel/corpus/builder.py`.

Consequences:

- `data/tokenizer_train/web.txt` is lossy.
- `data/tokenizer_train/code.txt` is lossy.
- `data/corpus/train/web.txt` and `code.txt` are lossy.
- `jsonl.txt` is mostly okay because Alpaca rows were already serialized as one JSON line.

The current web/code corpus cannot be repaired by re-splitting. It must be regenerated from the HF sources.

## Fix plan

Tracked under TASK-005.08.

### TASK-005.08.01 - Corpus builder

- Write document-level JSONL.
- One physical JSONL row equals one document.
- Preserve internal newlines inside the JSON `text` field.
- Split train/val by document hash.
- Write `manifest.json` with doc_count, byte_count, token_count or estimated_token_count, and target_fraction.

Target layout:

```text
data/corpus/train/web.jsonl
data/corpus/train/code.jsonl
data/corpus/train/jsonl.jsonl
data/corpus/train/manifest.json
data/corpus/val/web.jsonl
data/corpus/val/code.jsonl
data/corpus/val/jsonl.jsonl
data/corpus/val/manifest.json
```

### TASK-005.08.02 - PretrainDataset

- Read JSONL documents plus manifest.
- Tokenize `document.text` with internal newlines preserved.
- Emit `im_start` and `im_end` around documents.
- Yield `(input, target, doc_ids)`.
- Mix domains by token deficit, not byte-weighted line sampling.
- Expose `estimated_steps()` for auto LR schedule horizon.

### TASK-005.08.03 - Model/trainer

- Accept optional `doc_ids` in `Kestrel.__call__`.
- Block attention across different `doc_id` values.
- Reset RoPE positions at document boundaries.
- Consume `(input, target, doc_ids)` in trainer train and validation.
- Support `num_steps <= 0` to run until dataset exhaustion.
- Use `estimated_steps()` as the LR schedule horizon in auto mode.

## Design decisions

- Use document-level JSONL, not plain text, because plain text cannot unambiguously represent documents with internal newlines.
- Use `doc_ids` plus position reset rather than relying only on EOS tokens.
- Keep dense attention for M1. At seq_len 1024/2048, a dense document-aware mask is simple and fast enough.
- Do not add third-party varlen attention for M1.
- Use `num_steps <= 0` for auto mode. This separates the stop condition from the LR schedule horizon.
- Regenerate the corpus from HF sources. The current web/code text files are unrecoverable.

## Sources

- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- RedPajama: https://github.com/togethercomputer/RedPajama-Data
- Dolma: https://github.com/allenai/dolma
- SmolLM corpus: https://huggingface.co/datasets/HuggingFaceFW/smoltalk-corpus
- The Pile: https://bigcode.github.io/the_pile/
- The Stack: https://github.com/bigcode-project/bigcode-dataset
- StarCoder data: https://github.com/bigcode-project/starcoder
- PaLM: https://arxiv.org/abs/2204.02311
- Llama 3.1: https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- TRL: https://huggingface.co/docs/trl/main/en/preprocessing
