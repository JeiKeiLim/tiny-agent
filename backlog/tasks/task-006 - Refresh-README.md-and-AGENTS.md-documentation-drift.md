---
id: TASK-006
title: Refresh README.md and AGENTS.md documentation drift
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-26 05:36'
updated_date: '2026-08-26 05:41'
labels:
  - docs
dependencies: []
modified_files:
  - README.md
  - AGENTS.md
ordinal: 32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit README.md and AGENTS.md against the current repository state and update stale descriptions, commands, repo layout, training/checkpoint/resume behavior, backlog conventions, and references. This is needed because the project has gained corpus building, tokenizer training, pretraining, checkpoint retention/resume, run logging, and updated code-quality workflow since the docs were last refreshed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 README.md accurately describes the current repo layout, install/check commands, corpus/tokenizer/pretrain workflow, checkpoint resume behavior, and any supported scripts
- [x] #2 AGENTS.md accurately describes the current stack, commands, code-quality gate, backlog task rules, and references without stale or contradictory guidance
- [x] #3 No documentation claims reference removed files, old checkpoint behavior, unsupported CLI flags, or outdated project scope
- [x] #4 make check remains green after documentation updates
- [x] #5 Backlog task notes record the drift findings and files changed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Dispatch one sub-agent to audit/update README.md against repo state. 2. Dispatch one sub-agent to audit/update AGENTS.md against repo state. 3. Review diffs for stale claims, unsupported commands, and contradictions. 4. Run make check. 5. Record findings/files in TASK-006 and finalize.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-26 dispatched two sub-agents (README-only and AGENTS-only). Drift fixed:
- README: status no longer says pretraining is only next; M1 is implemented and being validated; repo layout matches current configs/src/scripts; added corpus build, pretrain, checkpoint/resume workflow; fixed Makefile table (help/sync/lint/typecheck/coverage); marked planned stages/packages.
- AGENTS: added implemented vs not-yet-implemented status, make sync/clean, checkpoint/resume invariants, and rule to update README/AGENTS when behavior changes; preserved the Backlog.md guidelines block.
Verification: make check green (140 tests). Files changed: README.md, AGENTS.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed README.md and AGENTS.md against the current repository state. README now documents M1 pretraining, corpus build, resumable checkpoints, actual repo layout, and Makefile targets; AGENTS now includes current status, checkpoint/resume invariants, and a doc-update rule. Verified with make check (140 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
