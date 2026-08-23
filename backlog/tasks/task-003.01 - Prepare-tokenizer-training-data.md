---
id: TASK-003.01
title: Prepare tokenizer training data
status: Done
assignee: []
created_date: '2026-08-21 07:15'
updated_date: '2026-08-21 07:42'
labels: []
milestone: m-0
dependencies: []
parent_task_id: TASK-003
ordinal: 2100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Obtain a representative text sample to train the BPE tokenizer: a few GB from the target domain (FineWeb-Edu + Python + JSONL mix). This is a precursor to the full pretrain corpus (milestone 1) — same sources, smaller sample. Decide the mix, download, and clean it into a form ready for tokenizer training.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A representative text sample is downloaded and cleaned, ready for tokenizer training
- [x] #2 The sample reflects the target-domain mix (web text + code + JSONL)
- [x] #3 Sample size is sufficient for a stable 16k-vocab BPE (a few GB)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1) uv add datasets (HF) for streaming. 2) Verify source availability: FineWeb-Edu (web), a Python code set, a JSONL set. 3) Write config-driven prep script (src/kestrel/data/) that streams samples per ~85/10/5 mix -> data/tokenizer_train/ (gitignored). 4) Run it (~1-2 GB). 5) Verify size + mix + content sanity; add data/ to .gitignore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Sources: web=HuggingFaceFW/fineweb-edu (text), code=theothertom/codeparrot-python-only (code), jsonl=tatsu-lab/alpaca (row serialized to JSONL). Rejected: bigcode/the-stack* (gated), codeparrot/github-code (deprecated script). New deps: datasets (HF streaming) + truststore (fixes SSL cert verify in this env; certifi lacks the system/corporate CA). Config-driven: configs/tokenizer/train_data.yaml -> TokenizerTrainDataConfig (strict Pydantic; fractions must sum to 1.0); prep script src/kestrel/data/prepare_tokenizer_data.py. Result (data/tokenizer_train/, gitignored): web.txt 912MB/4.26M lines, code.txt 107MB/2.86M lines, jsonl.txt 49MB/52k lines; total ~1.07GB, mix ~85/10/4.6. make check green (10 tests incl. 5 new for the config model).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared a ~1.07 GB representative tokenizer training sample (web 85% / code 10% / JSONL 4.6%) in data/tokenizer_train/ via a config-driven prep script. Added datasets + truststore deps and a strict Pydantic config model (with validator) + 5 tests. make check green.
<!-- SECTION:FINAL_SUMMARY:END -->
