---
id: TASK-004
title: >-
  Code quality infra: strict config validation (Pydantic) + ruff/mypy/pytest +
  Makefile
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-21 06:59'
updated_date: '2026-08-21 07:04'
labels: []
milestone: m-0
dependencies: []
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Set up the project code-quality toolchain and make config validation strict. (1) Config models use Pydantic v2 strict mode so a mistyped scalar (e.g. n_layers: "15" as a string) raises a clear ValidationError. (2) Ruff for lint+format. (3) mypy (strict) for static types. (4) pytest + pytest-cov for unit tests. (5) Makefile targets: install/format/lint/typecheck/test/coverage/check/clean/help. Run the full check on existing code and fix findings. Document the strategy in README; update plan doc §6/§5 (dataclass -> Pydantic).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Config loader rejects a mistyped scalar (e.g. n_layers: "15" as a string) with a Pydantic ValidationError; valid 50m/150m configs still load
- [x] #2 ruff check and ruff format pass on src/ and tests/ (ruff configured in pyproject)
- [x] #3 mypy (strict) passes on src/
- [x] #4 pytest + pytest-cov run via Makefile; existing loader tests pass
- [x] #5 Makefile provides install/format/lint/typecheck/test/coverage/check/clean/help targets
- [x] #6 README documents the code-quality strategy; plan doc §6/§5 updated (dataclass -> Pydantic)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. uv add pydantic; uv add --group dev ruff mypy pytest-cov
2. Refactor model/config.py ModelConfig -> Pydantic BaseModel (strict)
3. Refactor common/config.py loader -> Pydantic model_validate; keep load_config(path, type) API
4. Update tests/test_config.py: Pydantic models for fixtures, ValidationError assertions
5. Configure ruff + mypy (strict, mlx/tokenizers overrides) in pyproject.toml
6. Create Makefile (install/format/lint/typecheck/test/coverage/check/clean/help)
7. Run make check; fix all findings
8. Update README (code-quality section) + plan doc §6/§5 (dataclass -> Pydantic)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pydantic v2 strict (BaseConfig: strict=True, extra=forbid) for all config models; load_config uses PEP 695 type params. Ruff (E,W,F,I,UP,B,C4,SIM,RUF; line-length 100), mypy strict (added types-PyYAML; mlx/tokenizers override for when model/ imports MLX), pytest+pytest-cov. Makefile targets installed. make check green (ruff all-pass, mypy 17 files, 5 tests). Coverage 79%. Mistyped n_layers now rejected with ValidationError. README + plan doc §5/§6 updated (dataclass->Pydantic).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Set up strict code-quality tooling: config models are strict Pydantic (BaseConfig) so mistyped/unknown values raise ValidationError; Ruff (lint+format), mypy (strict), pytest+pytest-cov; Makefile (install/format/lint/typecheck/test/coverage/check/clean/help). make check green; README + plan doc updated.
<!-- SECTION:FINAL_SUMMARY:END -->
