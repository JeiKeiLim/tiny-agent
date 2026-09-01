
# Kestrel — Project Context

Kestrel is a small-scale, modern **agentic LLM training pipeline** on [MLX](https://github.com/ml-explore/mlx). It is meant to train a **pair of small decoder-only models (50M and 150M)** from scratch through the full pipeline (pretrain → long-context → SFT → RL → serve + agent → eval), plus a **PEFT/LoRA track** (Track B) on a pretrained base. The goal is learning the pipeline and the 50M↔150M scaling comparison — not raw capability.

## Status

- **Implemented:** scaffolding + strict YAML→Pydantic config loader, the Kestrel model (`src/kestrel/model/`) with `generate()`, the byte-level BPE tokenizer (`tokenizer/`), the corpus builder (`corpus/`), the document-aware pretrain dataset (`data/pretrain_dataset.py`), the shared MLX trainer (`train/trainer.py`) with checkpoint retention/resume and `run.jsonl` logging, the pretrain entry point (`scripts/run_pretrain.py`), read-only pretrain checkpoint evaluation (`scripts/eval_pretrain.py`), the M2 SFT chat/renderer/schema/dataset/trainer stack (`src/kestrel/data/sft_*.py`, `src/kestrel/train/sft.py`, `scripts/run_sft.py`), SFT data prep for public assistant, GSM8K, local tool, public tool, and optional internal LLM sources (`scripts/run_prepare_sft.py`), held-out SFT eval bundle prep (`scripts/run_prepare_sft.py --source eval`, `src/kestrel/data/sft_prepare_eval.py`), the SFT mixture builder (`src/kestrel/data/sft_mixture.py`, `scripts/run_build_sft_mixture.py`), interactive SFT chat (`scripts/chat_sft.py`), and the inference-only SFT eval harness + scorecard (`src/kestrel/eval/sft.py`, `src/kestrel/eval/tool_calling.py`, `scripts/run_eval_sft.py`, `configs/kestrel/50m/eval_sft.yaml`). The SFT chat renderer exposes `row.tools` as a compact loss-masked system block so tool schemas are part of both training and eval prompts. The no-KV-cache `generate()` path releases unused MLX allocator cache every `clear_cache_every` generated tokens (default `64`) to bound retained cache memory; `TASK-008.02` tracks the proper KV-cache generation fix.
- **Not yet implemented:** long-context, RL, serve + agent, and Track B (PEFT/LoRA). They are designed in `doc-001` but do not have complete code yet — do not assume their modules, configs, or scripts exist.
- **Internal LLM data prep:** endpoint, API key, and model name are read only from environment variables named in `configs/kestrel/sft_data.yaml`. The committed config and repo must not contain secret values; `.env.example` documents the variable names, and `.env` is gitignored.

## Stack
- **Python 3.13**, managed with **uv** (`.venv` + `uv.lock`; some environments need `uv --system-certs` for network).
- **MLX** for all ML; **Pydantic** (strict) for config models; **Ruff** (lint + format), **mypy** (strict), **pytest** + **pytest-cov**.

## Commands (Makefile)
- `make install` — create `.venv` + install deps (runtime + dev). `make sync` is an alias.
- `make check` — **lint (ruff check + format) + typecheck + test** (the gate for any code change).
- `make format` / `make lint` / `make typecheck` / `make test` / `make coverage` / `make clean` — individual steps.
- `make help` — list all targets.

## Code quality — MANDATORY
**After writing or changing any code, you MUST run `make check` and fix every failure (format, lint, typecheck, test) before the work is considered done.** Never hand off, commit, or mark a task complete while `make check` is red. If you add a dependency or tool, keep `make check` green.

Conventions:
- Configs are **strict Pydantic models** (subclass `BaseConfig`); a mistyped value or unknown key must raise `ValidationError`.
- Tests live in `tests/`, mirror the source layout, and test the **code** (not config data values).
- Keep code modular and config-driven (one module per concern; see plan doc §6).
- When behavior changes (commands, layout, checkpoint format, pipeline stages), update `README.md` and this file in the same change.

## Checkpoint / resume invariants

- A **full** checkpoint dir is self-describing and resumable: `weights.npz` + `optimizer.npz` + `state.json` (training state + config/tokenizer/corpus hashes) + optional `config/` and `run.jsonl` snapshots. `scripts/run_pretrain.py --resume DIR` re-validates those hashes against the current config before resuming.
- When touching `src/kestrel/train/`, preserve resumability: `state.json` stays complete and is written last inside the atomic tmp-dir + rename flow in `train/checkpoint.py`.
- Weights-only checkpoints (no `state.json`) are **not** resumable — never treat a dir with just `weights.npz` as resumable.

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
