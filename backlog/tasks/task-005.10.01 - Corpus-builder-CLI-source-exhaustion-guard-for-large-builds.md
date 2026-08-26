---
id: TASK-005.10.01
title: 'Corpus builder: CLI + source-exhaustion guard for large builds'
status: To Do
assignee: []
created_date: '2026-08-26 01:35'
labels:
  - data
  - pretraining
  - corpus
milestone: m-1
dependencies: []
modified_files:
  - src/kestrel/corpus/builder.py
  - src/kestrel/corpus/config.py
  - tests/test_corpus_builder.py
  - scripts/build_corpus.py
parent_task_id: TASK-005.10
priority: high
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a standalone corpus build entrypoint and prevent large corpus builds from silently succeeding when an HF or local source is exhausted before its byte target.

Files:
- create scripts/build_corpus.py
- modify src/kestrel/corpus/builder.py
- modify src/kestrel/corpus/config.py
- modify tests/test_corpus_builder.py

Config change:
- Add CorpusConfig.min_component_fill: float, default 0.9, ge 0.0, le 1.0.
- build() already computes target = int(config.total_bytes * comp.fraction).
- After each component build, if written < target * min_component_fill, raise ValueError naming the component, source, target bytes, and written bytes.
- Existing tests that intentionally use tiny local sources must set min_component_fill to 0 or another low value.

CLI:
- scripts/build_corpus.py accepts --config <corpus.yaml> and optional --force.
- Load CorpusConfig strictly, call build(config, force=args.force), and print the returned per-component byte counts.

Tests:
- A local source smaller than target raises when min_component_fill is 0.9.
- The same local source succeeds when min_component_fill is 0.
- The CLI builds a tiny local corpus config and writes manifests.
- Existing idempotent skip behavior remains intact.

Gate: make check green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 scripts/build_corpus.py --config <tiny local corpus config> builds the corpus and prints per-component byte counts
- [ ] #2 build() raises a clear error when a component writes less than min_component_fill of its target bytes
- [ ] #3 build() succeeds for an intentionally small local source when min_component_fill is 0
- [ ] #4 Existing idempotent corpus skip behavior still passes
- [ ] #5 make check is green
<!-- AC:END -->
