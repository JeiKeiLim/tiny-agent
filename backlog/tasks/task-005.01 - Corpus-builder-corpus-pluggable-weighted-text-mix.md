---
id: TASK-005.01
title: Corpus builder (corpus/) - pluggable weighted text mix
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-24 01:55'
updated_date: '2026-08-24 04:21'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. config.py: HfSourceConfig(type=hf,dataset,config?,text_field?) / LocalSourceConfig(type=local,path); SourceConfig = discriminated union on 'type'; ComponentConfig(name,source,fraction); CorpusConfig(total_bytes,seed,output_dir,components) + fractions-sum-to-1.0 validator (mirror tokenizer_data_config.py).
2. builder.py: build(config)->dict[str,int]; _build_local (read line-by-line up to target bytes -> output_dir/<name>.txt); _build_hf (stream via datasets.load_dataset streaming, lazy truststore, reuse prepare_tokenizer_data.py pattern); _extract_text (text_field or JSONL row).
3. configs/kestrel/corpus.yaml: local sources -> data/tokenizer_train/{web,code,jsonl}.txt (0.85/0.10/0.05), total_bytes ~1GB, output_dir data/corpus.
4. tests/test_corpus_builder.py: config (valid/fractions/unknown-key/bad-type/real-config-loads) + local build (tiny fixtures: one file per component, each <= target, total within ~5%; exhausted source) + hf build (mock load_dataset, tiny target).
5. Gate: make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Built corpus/ module: config.py (HfSourceConfig/LocalSourceConfig discriminated union on 'type', ComponentConfig, CorpusConfig + fractions-sum-to-1.0 validator), builder.py (build() -> per-component bytes; _build_local copies up to target bytes; _build_hf streams via datasets with lazy truststore; _extract_text handles text_field or JSONL row). configs/kestrel/corpus.yaml = local sources -> data/tokenizer_train/*.txt (0.85/0.10/0.05). tests/test_corpus_builder.py: 9 tests (config validation x5, local build x2, hf build x2 with mocked load_dataset). make check green (51 tests, mypy 27 files).

Real build verified (not just fixtures): ran build() on configs/kestrel/corpus.yaml -> data/corpus/{web,code,jsonl}.txt. web 912,680,728 B (hit 85% target), code 107,319,876 B (source exhausted ~62KB short, CRLF->LF normalization), jsonl 49,016,386 B (alpaca JSONL exhausted ~49MB < 51MB target). Total ~1.069 GB, proportions match 85/10/5. No standalone CLI by design: build() is a library step called inline by 005.05 pretrain(); user opted to keep it inline (no run_corpus.py).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built the pluggable corpus builder (corpus/): config.py (HfSourceConfig/LocalSourceConfig discriminated union on 'type', ComponentConfig, CorpusConfig + fractions-sum-to-1.0 validator), builder.py (build() -> per-component bytes; _build_local copies up to target; _build_hf streams via datasets with lazy truststore; _extract_text handles text_field or JSONL), configs/kestrel/corpus.yaml (local sources -> data/tokenizer_train/*.txt, 0.85/0.10/0.05, ~1GB), tests/test_corpus_builder.py (9 tests). Verified: make check green (51 tests, mypy 27 files) + real build produced data/corpus/ (~1.069 GB, 3 files, correct proportions). Library by design (no CLI) - called inline by 005.05.
<!-- SECTION:FINAL_SUMMARY:END -->
