---
id: TASK-005.08.01
title: 'Corpus builder: document-level JSONL + manifest'
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-26 00:24'
updated_date: '2026-08-26 00:39'
labels:
  - data
  - corpus
milestone: m-1
dependencies: []
documentation:
  - doc-003
modified_files:
  - src/kestrel/corpus/builder.py
  - src/kestrel/corpus/config.py
  - configs/kestrel/corpus.yaml
  - tests/test_corpus_builder.py
parent_task_id: TASK-005.08
priority: high
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the corpus builder so it preserves document structure.

Root cause:
- src/kestrel/data/prepare_tokenizer_data.py writes HF row text with `text + "\n"`.
- src/kestrel/corpus/builder.py does the same for HF sources and reads local sources by physical line.
- This flattens multi-line documents from FineWeb and codeparrot Python files.
- Current data/corpus/train/web.txt and code.txt are lossy and should not be used for a serious run.

Outcome:
- Corpus builder writes one JSON document per physical line.
- Internal newlines are preserved inside the JSON `text` field.
- Train/val split is by document, not by physical line.
- Manifest records document counts and token counts.

Files to modify:
- src/kestrel/corpus/config.py
- src/kestrel/corpus/builder.py
- configs/kestrel/corpus.yaml
- tests/test_corpus_builder.py

Config behavior:
- Add JSONL output as the default pretraining corpus format.
- HF source: extract row text, then write `json.dumps({"domain": name, "text": text}, ensure_ascii=False) + "\n"`.
- Local source: support `.jsonl` files as document-level input. Plain `.txt` may remain only as a deprecated fallback where one physical line equals one document.
- Split routing must hash the full document text, not each flattened physical line.
- Write `data/corpus/{train,val}/manifest.json`.

Manifest schema:
- split
- seed
- files list with path, domain, doc_count, byte_count, token_count or estimated_token_count, target_fraction.
- If a tokenizer path is available, compute exact token_count. Otherwise estimate token_count from byte_count and mark it as estimated.

Acceptance targets:
- A multi-line code document remains one physical JSONL row and round-trips to identical text.
- A multi-line web document remains one physical JSONL row and round-trips to identical text.
- Manifest doc_count equals the number of JSONL rows.
- make check is green.

Reference: doc-003.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Multi-line code document remains one physical JSONL row and round-trips to identical text
- [x] #2 Multi-line web document remains one physical JSONL row and round-trips to identical text
- [x] #3 Manifest doc_count equals the number of JSONL rows
- [x] #4 make check is green
- [x] #5 Small real HF build preserves multiline documents and writes manifest
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add output_format and tokenizer_path to CorpusConfig. 2. Change builder to emit one JSON document per line for HF/local JSONL/local txt. 3. Route splits by full document text. 4. Accumulate per-split/per-domain stats and write manifest.json. 5. Update tests for JSONL, internal newlines, manifest counts, and legacy txt behavior. 6. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented document-level JSONL corpus builder. Added CorpusConfig.output_format (default jsonl) and optional tokenizer_path. Builder writes one JSON document per physical line, supports local .jsonl and legacy .txt, splits by full document text, and writes per-split manifest.json with doc_count, byte_count, target_fraction, and estimated or exact token counts. tests/test_pretrain.py temporarily uses output_format=txt until TASK-005.08.02 updates PretrainDataset. make check green: 89 tests.

Reopened: unit tests and make check are not sufficient for this task. Need a real small HF corpus build smoke test before marking Done.

Real HF smoke build completed to /var/folders/11/jmxpptjn50s_v9k5k1qw30z40000gn/T/opencode/corpus-smoke with total_bytes=500000. Results: web 428696 bytes, code 60036 bytes, jsonl 27222 bytes. Train manifest total_doc_count=105, val=12. Web and code JSONL docs preserve internal newlines; jsonl docs are single-line serialized rows. make check remained green (89 tests).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corpus builder emits document-level JSONL plus manifest.json. Verified with unit tests, make check (89 tests), and a small real HF build that preserved multiline web/code documents.
<!-- SECTION:FINAL_SUMMARY:END -->
