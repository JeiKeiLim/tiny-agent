---
id: TASK-005.01
title: Corpus builder (corpus/) - pluggable weighted text mix
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
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the pluggable corpus builder that assembles the pretraining text as a weighted mix of sources. Generalizes the existing src/kestrel/data/prepare_tokenizer_data.py (weighted HF streaming) into a reusable corpus/ module supporting two source types: 'hf' (stream from HuggingFace) and 'local' (read an existing file). For the M1 validation run we use 'local' sources pointing at the existing 1GB tokenizer sample (data/tokenizer_train/{web,code,jsonl}.txt) so NO new download is needed; the full ~1B run (later) will use 'hf' sources.

Files to create:
- src/kestrel/corpus/config.py - Pydantic models: SourceConfig (discriminated by 'type': hf | local), ComponentConfig (name, source, fraction), CorpusConfig (total_bytes, seed, output_dir, components).
- src/kestrel/corpus/builder.py - build(config) -> dict[str,int] (per-source bytes written).
- configs/kestrel/corpus.yaml - validation config: 'local' sources -> data/tokenizer_train/*.txt, fractions web 0.85 / code 0.10 / jsonl 0.05.
- tests/test_corpus_builder.py
(src/kestrel/corpus/__init__.py already exists as an empty package - verify.)

Config fields (configs/kestrel/corpus.yaml -> CorpusConfig):
- total_bytes: int  (raw-text target; the dataset step's total_tokens is the authoritative training cap)
- seed: int
- output_dir: str  (e.g. data/corpus)
- components: list of { name: str, source: {...}, fraction: float }
  - hf source:  { type: hf, dataset: str, config: str|None, text_field: str|None }
  - local source: { type: local, path: str }
- component fractions must sum to 1.0 (validated, like the tokenizer config).

Behavior:
- local: read the file, take up to (total_bytes * fraction) bytes, write output_dir/<name>.txt.
- hf: stream via datasets.load_dataset(..., streaming=True) up to the byte target (reuse the prepare_tokenizer_data.py pattern, incl. truststore).
- text_field None -> serialize each row to a JSONL line (same as tokenizer prep).
- Byte counting = len(line.encode('utf-8')) (match tokenizer prep).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CorpusConfig rejects component fractions that do not sum to 1.0 and rejects unknown keys (strict Pydantic -> ValidationError)
- [x] #2 build() with 'local' sources assembles the weighted mix: one output file per component under output_dir, each <= its byte target, total within ~5% of total_bytes
- [x] #3 build() with an 'hf' source streams up to the byte target (test with a tiny target so it is fast)
- [x] #4 tests/test_corpus_builder.py uses tiny local fixture files + a tiny byte target (NOT the 1GB sample) so it runs fast; make check green
- [x] #5 corpus is split into train/val(/test) by a deterministic per-line hash: no line appears in two splits, and train+val(+test) == all input lines
- [x] #6 split is reproducible (same seed -> identical split); val_fraction + test_fraction > 1.0 is rejected (ValidationError)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Split (Option A, hash-based): (1) config.py CorpusConfig gains val_fraction=0.1, test_fraction=0.0 + validator val+test<=1.0 (reuses seed). (2) builder.py: per-line hash split _split_for(line,seed,val,test)=sha256(f'{seed}:{line}') -> uniform -> train/val/test; _split_writers context manager opens output_dir/{train,val[,test]}/<name>.txt; _build_local/_build_hf route each line to its split file, stop at per-component byte target; build() returns per-component total bytes. Output layout: data/corpus/{train,val[,test]}/<name>.txt (was flat). (3) corpus.yaml: val_fraction 0.1, test_fraction 0.0. (4) tests: update existing for new layout (val_fraction=0.0 -> train path) + add split tests (no leakage, completeness, determinism, ratio, config val+test>1 rejects). Dataset 005.02 unchanged (points at data/corpus/train or /val). 005.03/005.05 plans updated for in-loop val loss (plan-only). make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Built corpus/ module: config.py (HfSourceConfig/LocalSourceConfig discriminated union on 'type', ComponentConfig, CorpusConfig + fractions-sum-to-1.0 validator), builder.py (build() -> per-component bytes; _build_local copies up to target bytes; _build_hf streams via datasets with lazy truststore; _extract_text handles text_field or JSONL row). configs/kestrel/corpus.yaml = local sources -> data/tokenizer_train/*.txt (0.85/0.10/0.05). tests/test_corpus_builder.py: 9 tests (config validation x5, local build x2, hf build x2 with mocked load_dataset). make check green (51 tests, mypy 27 files).

Real build verified (not just fixtures): ran build() on configs/kestrel/corpus.yaml -> data/corpus/{web,code,jsonl}.txt. web 912,680,728 B (hit 85% target), code 107,319,876 B (source exhausted ~62KB short, CRLF->LF normalization), jsonl 49,016,386 B (alpaca JSONL exhausted ~49MB < 51MB target). Total ~1.069 GB, proportions match 85/10/5. No standalone CLI by design: build() is a library step called inline by 005.05 pretrain(); user opted to keep it inline (no run_corpus.py).

Reopened: adding deterministic train/val(/test) split. This was slipped out of the M1 decomposition - no M1 task owned it, though doc-001:336/313/316 intend pretrain perplexity on a held-out set (assigned to the later eval milestone, not M1). Option A (corpus pre-split) chosen over dataset-level split.

Split implemented (Option A, hash-based): CorpusConfig gains val_fraction=0.1/test_fraction=0.0 + validator; builder routes each line via sha256(f'{seed}:{line}') to output_dir/{train,val[,test]}/<name>.txt. 63 tests green (was 57; +6 split tests). Real build: data/corpus/train ~922MB + val ~112MB (~90/10), source mix 85/10/5 preserved in each split. Dataset 005.02 unchanged (point at data/corpus/train or /val).

Decision (2026-08-24): test_fraction=0 for M1. For a from-scratch base model there is no single fixed test set; the 'test' is (a) held-out perplexity (the val slice can serve this) and (b) downstream benchmarks (GSM8K test, BFCL unseen, agent tasks) in the eval milestone (doc-001:75/313/316/336) using EXTERNAL datasets, not a corpus slice. If a clean unbiased final perplexity is wanted later, bump test_fraction to ~0.05 (val for in-loop/stopping, test for final perplexity) - one-line config change. In-loop val loss is cheap at our scale (50M model), so we use the nanoGPT pattern (val loss every eval_every steps), not the large-scale 'periodic held-out perplexity only' approach. Verified via web research (nanoGPT train.py: train.bin+val.bin, estimate_loss() over eval_iters, val loss every eval_interval, best-val checkpointing).

Follow-up: original corpus builder preserved weighted mixing but not document structure. TASK-005.08.01 tracks document-level JSONL output and manifest generation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built the pluggable corpus builder (corpus/) with a deterministic train/val(/test) split. config.py (HfSourceConfig/LocalSourceConfig discriminated union on 'type', ComponentConfig, CorpusConfig + fractions-sum-to-1.0 + val/test<=1.0 validators), builder.py (build() -> per-component bytes; _build_local/_build_hf stream up to target; _split_for routes each line via sha256(f'{seed}:{line}') to output_dir/{train,val[,test]}/<name>.txt), configs/kestrel/corpus.yaml (local sources -> data/tokenizer_train/*.txt, 0.85/0.10/0.05, val_fraction 0.1, test_fraction 0.0), tests/test_corpus_builder.py (15 tests incl. 6 split tests). Verified: make check green (63 tests) + real build -> data/corpus/train (~922MB) + data/corpus/val (~112MB), ~90/10, source mix preserved per split. Library by design (no CLI) - called inline by 005.05. test_fraction=0 by design: pretrain 'test' is the downstream eval (doc-001 section 13, external benchmarks), not a held-out corpus slice; the val slice serves in-loop val loss + final perplexity.
<!-- SECTION:FINAL_SUMMARY:END -->
