---
id: TASK-001
title: 'Scaffolding: kestrel package + config + env'
status: Done
assignee:
  - '@opencode'
created_date: '2026-08-21 06:44'
updated_date: '2026-08-21 06:52'
labels: []
milestone: m-0
dependencies: []
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Set up the src/kestrel/ package skeleton and shared infrastructure so every later module has a home. Includes common/ (YAML -> typed-dataclass config loading, logging), configs/ + scripts/ directory skeletons per §6, and the Python environment (MLX, tokenizers, etc.) with pyproject.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 src/kestrel/ package imports cleanly; common/ config loader parses a sample YAML into a typed dataclass
- [x] #2 Python env (MLX, tokenizers) installs and imports; pyproject/requirements present
- [x] #3 configs/ and scripts/ directory structure matches §6 layout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. uv init --package --name kestrel; pin Python 3.13
2. uv add mlx tokenizers pyyaml numpy; uv add --group dev pytest
3. Create §6 subdirs under src/kestrel/ (common, model, tokenizer, corpus, data, train, train/rl, peft, tools, env, agent, serve, eval) with __init__.py
4. common/config.py: generic YAML->dataclass loader
5. common/logging.py: logging setup
6. model/config.py: ModelConfig dataclass (Kestrel shape)
7. configs/kestrel/50m/model.yaml: sample config
8. scripts/ skeleton; .gitignore (Python+uv)
9. tests/test_config.py: load sample YAML into ModelConfig, assert fields; uv run pytest
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scaffolding done via uv (Python 3.13.2, .venv). Created src/kestrel/ package with all §6 subdirs; common/config.py (generic YAML->dataclass loader), common/logging.py, model/config.py (ModelConfig). Sample config configs/kestrel/50m/model.yaml. Deps: mlx 0.32.1, tokenizers, pyyaml, numpy; dev: pytest. 3 tests pass (uv run pytest). Note: uv needs --system-certs in this env (TLS).

Added configs/kestrel/150m/model.yaml (32L/640H/10Q/2KV/1728F) + test — the 50m-only sample was an oversight; the project is a 50M/150M pair so both model configs ship from scaffolding. 4 tests pass.

Refactored tests: config VALUES are tunable data, not invariants — stopped pinning them. Loader tests now use synthetic fixtures (scalars/list/nested/defaults/unknown-key/non-dataclass); real model configs only checked for being loadable (isinstance), so editing a value won't fail tests. 5 tests pass.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Set up the uv-managed Kestrel package (Python 3.13): src/kestrel/ skeleton with all §6 subdirs, common/config.py (generic YAML->dataclass loader), common/logging.py, model/config.py (ModelConfig), sample configs/kestrel/50m/model.yaml, .gitignore, and scripts/ skeleton. Deps: mlx/tokenizers/pyyaml/numpy + pytest (dev). Verified: 3 tests pass (uv run pytest), package imports cleanly, config loader round-trips the 50M model config.
<!-- SECTION:FINAL_SUMMARY:END -->
