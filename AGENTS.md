
# Kestrel — Project Context

Kestrel is a small-scale, modern **agentic LLM training pipeline** on [MLX](https://github.com/ml-explore/mlx). It trains a **pair of small decoder-only models (50M and 150M)** from scratch through the full pipeline (pretrain → long-context → SFT → RL → serve + agent → eval), plus a **PEFT/LoRA track** on a pretrained base. The goal is learning the pipeline and the 50M↔150M scaling comparison — not raw capability.

## Stack
- **Python 3.13**, managed with **uv** (`.venv` + `uv.lock`; some environments need `uv --system-certs` for network).
- **MLX** for all ML; **Pydantic** (strict) for config models; **Ruff** (lint + format), **mypy** (strict), **pytest** + **pytest-cov**.

## Commands (Makefile)
- `make install` — create `.venv` + install deps (runtime + dev).
- `make check` — **lint (ruff check + format) + typecheck + test** (the gate for any code change).
- `make format` / `make lint` / `make typecheck` / `make test` / `make coverage` — individual steps.
- `make help` — list all targets.

## Code quality — MANDATORY
**After writing or changing any code, you MUST run `make check` and fix every failure (format, lint, typecheck, test) before the work is considered done.** Never hand off, commit, or mark a task complete while `make check` is red. If you add a dependency or tool, keep `make check` green.

Conventions:
- Configs are **strict Pydantic models** (subclass `BaseConfig`); a mistyped value or unknown key must raise `ValidationError`.
- Tests live in `tests/`, mirror the source layout, and test the **code** (not config data values).
- Keep code modular and config-driven (one module per concern; see plan doc §6).

## Backlog task authoring — MANDATORY

**Every task must be standalone: a fresh agent with no prior conversation context should be able to open the task and start working immediately.** After compaction or in a new session only the task document (not the chat) is available, so it must carry everything needed to do the work.

When creating or splitting a task, its description + plan + notes must include:
- **Concrete file paths** to create/modify (e.g. `src/kestrel/model/kestrel.py`, not just "the model").
- **Config locations + field names** when config-driven (where the YAML is, what the fields are).
- **Quantitative targets with a tolerance** so an AC can be asserted, not hand-waved (e.g. "param count within ~5% of 50M / 150M").
- **Test file names** and what each test asserts.
- **Design decisions + gotchas** (framework pitfalls, non-obvious choices + rationale) so they are not re-litigated or re-discovered.
- **The verification gate** (`make check`) and how to verify the result.
- **References** to the plan doc (`backlog/docs/doc-001 §N`) for the "why".

Before marking a task ready, audit it for standalone-ness: if a cold reader would have to guess at anything, add the missing detail.

## References
- `README.md` — overview, repo layout, code-quality strategy.
- `backlog/docs/doc-001 …` — the full project plan (architecture, corpus, SFT/RL, serve + agent, eval, reasoning-effort).

<!-- BACKLOG.MD GUIDELINES START -->
<CRITICAL_INSTRUCTION>

## Backlog.md Workflow

This project uses Backlog.md for task and project management.

**For every user request in this project, run `backlog instructions overview` before answering or taking action.**

Use the overview to decide whether to search, read, create, or update Backlog tasks.

Use the detailed guides when needed:
- `backlog instructions task-creation` for creating or splitting tasks
- `backlog instructions task-execution` for planning and implementation workflow
- `backlog instructions task-finalization` for completion and handoff

Use `backlog <command> --help` before running unfamiliar commands. Help shows options, fields, and examples.

Do not edit Backlog task, draft, document, decision, or milestone markdown files directly. Use the `backlog` CLI so metadata, relationships, and history stay consistent.

</CRITICAL_INSTRUCTION>
<!-- BACKLOG.MD GUIDELINES END -->
