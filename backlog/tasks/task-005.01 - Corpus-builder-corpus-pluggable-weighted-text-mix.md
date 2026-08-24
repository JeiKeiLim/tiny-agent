---
id: TASK-005.01
title: Corpus builder (corpus/) - pluggable weighted text mix
status: To Do
assignee: []
created_date: '2026-08-24 01:55'
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
- [ ] #1 CorpusConfig rejects component fractions that do not sum to 1.0 and rejects unknown keys (strict Pydantic -> ValidationError)
- [ ] #2 build() with 'local' sources assembles the weighted mix: one output file per component under output_dir, each <= its byte target, total within ~5% of total_bytes
- [ ] #3 build() with an 'hf' source streams up to the byte target (test with a tiny target so it is fast)
- [ ] #4 tests/test_corpus_builder.py uses tiny local fixture files + a tiny byte target (NOT the 1GB sample) so it runs fast; make check green
<!-- AC:END -->
