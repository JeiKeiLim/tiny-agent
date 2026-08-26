---
id: TASK-005.08.01
title: 'Corpus builder: document-level JSONL + manifest'
status: To Do
assignee: []
created_date: '2026-08-26 00:24'
updated_date: '2026-08-26 00:25'
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
- [ ] #1 Multi-line code document remains one physical JSONL row and round-trips to identical text
- [ ] #2 Multi-line web document remains one physical JSONL row and round-trips to identical text
- [ ] #3 Manifest doc_count equals the number of JSONL rows
- [ ] #4 make check is green
<!-- AC:END -->
