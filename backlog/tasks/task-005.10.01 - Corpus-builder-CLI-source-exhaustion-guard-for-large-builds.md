---
id: TASK-005.10.01
title: 'Corpus builder: CLI + source-exhaustion guard for large builds'
status: Done
assignee:
  - '@limjk'
created_date: '2026-08-26 01:35'
updated_date: '2026-08-26 02:29'
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
- [x] #1 scripts/build_corpus.py --config <tiny local corpus config> builds the corpus and prints per-component byte counts
- [x] #2 build() raises a clear error when a component writes less than min_component_fill of its target bytes
- [x] #3 build() succeeds for an intentionally small local source when min_component_fill is 0
- [x] #4 Existing idempotent corpus skip behavior still passes
- [x] #5 make check is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add CorpusConfig.min_component_fill with default 0.9. 2. In build(), after each component, raise ValueError if written bytes are below min_component_fill * target. 3. Create scripts/build_corpus.py with --config and optional --force, call build(), and print per-component byte counts. 4. Update existing tiny-source tests to set min_component_fill=0 where needed. 5. Add tests for failure, success with min_component_fill=0, CLI build, and idempotent skip. 6. Run make check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added CorpusConfig.min_component_fill (default 0.9), build() guard, and scripts/build_corpus.py. Updated existing tiny-source tests to min_component_fill=0. Added failure, zero-fill success, and CLI regression tests. make check passed with 112 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a corpus build CLI and a min_component_fill guard so large builds fail loudly when a source is exhausted. Verified with new corpus-builder/CLI tests and make check (112 tests passed).
<!-- SECTION:FINAL_SUMMARY:END -->
