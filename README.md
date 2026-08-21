# Kestrel

A small-scale, modern **agentic LLM training pipeline** on [MLX](https://github.com/ml-explore/mlx), built to learn the full flow end-to-end:

```
pretrain → long-context → SFT → RL → serve + agent → eval
```

Kestrel trains a **pair of small decoder-only models (50M and 150M)** from scratch and compares them at every stage. The goal is understanding the modern training pipeline and the 50M↔150M scaling comparison — not raw capability.

## Two tracks

- **Track A (from scratch):** train Kestrel-50M and Kestrel-150M through the whole pipeline with full fine-tuning.
- **Track B (PEFT / LoRA):** fine-tune a pretrained base (default Qwen3-1.7B) with LoRA and LoRA-like methods (QLoRA, DoRA, adapters) behind a pluggable `PEFTMethod` interface. Shares the SFT / RL / serve / eval code with Track A.

## Pipeline (Track A)

1. **Tokenizer** — byte-level BPE, 16k vocab (shared by both sizes).
2. **Pretrain** — ~1B tokens (FineWeb-Edu + Python + synthesized JSONL).
3. **Long-context** — RoPE interpolation + staged continuation (4k → 8k → 16k).
4. **SFT** — hybrid tool-calling + chain-of-thought data, unified chat template.
5. **RL** — GRPO. **RL-A** (pure math) + **RL-B** (synthetic agentic tool environment).
6. **Serve + agent** — MLX `generate()` + optional OpenAI-compatible server; a robust single-loop agent.
7. **Eval** — a 4-checkpoint scorecard per size (pretrain → +SFT → +RL-A → +RL-B).

## Status

Scaffolding is complete (Backlog `TASK-001`): the `src/kestrel/` package, the generic YAML→dataclass config loader, `ModelConfig`, sample configs, and the uv environment. The remaining modules are designed in the project plan and are being built milestone by milestone.

## Repo layout

```
tiny-agent/
  configs/            # by model
    kestrel/          # from-scratch (Track A) — family of 2 sizes
      tokenizer.yaml  corpus.yaml
      50m/            model.yaml  pretrain.yaml  long_context.yaml
                      sft.yaml  rl_a.yaml  rl_b.yaml  serve.yaml  agent.yaml  eval.yaml
      150m/           model.yaml  ...  peft_sft.yaml  peft_rl.yaml
    qwen3_1_7b/       # pretrained base (Track B)
  src/kestrel/
    common/           # config (YAML→dataclass), logging, utils
    model/            # config.py, kestrel.py, pretrained.py, io.py (load/save)
    tokenizer/        # (Track A) train BPE
    corpus/           # (Track A) pluggable corpus builder
    data/             # dataset prep (pretrain, sft)
    train/            # trainer.py + pretrain.py, long_context.py, sft.py, rl/
    peft/             # PEFTMethod iface + lora/qlora/dora/adapter + registry
    tools/            # shared tool registry + impls + task-suite generator
    env/              # agentic environment
    agent/            # loop, client, parse, context, trace
    serve/            # generate.py (MLX), server.py (OpenAI-compatible)
    eval/             # metrics, math, tool_calling, agent_task, perplexity
  scripts/            # one entry point per phase
  tests/
  data/  checkpoints/  outputs/
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (pinned in `.python-version`).

```bash
make install     # create .venv and install dependencies (runtime + dev)
make check       # lint + typecheck + test
```

> Note: in some environments `uv` needs `--system-certs` for network operations (TLS).

## Code quality

Strict, standard tooling — run it all with `make check`:

- **Config validation** — configs are [Pydantic](https://docs.pydantic.dev/) models in **strict mode** (`extra="forbid"`): a mistyped value (e.g. `n_layers: "15"`) or an unknown key raises a clear `ValidationError`.
- **Lint + format** — [Ruff](https://docs.astral.sh/ruff/) (`ruff check` + `ruff format`).
- **Static types** — [mypy](https://mypy.readthedocs.io/) in `strict` mode.
- **Unit tests** — [pytest](https://docs.pytest.org/) + [pytest-cov](https://pytest-cov.readthedocs.io/). Tests live in `tests/`, mirror the source layout, and test the *code* (not config data values).

Makefile targets:

| Target | Runs |
|--------|------|
| `make install` | `uv sync --all-groups` |
| `make format` | `ruff format` + `ruff check --fix` |
| `make lint` | `ruff check` |
| `make typecheck` | `mypy src` (strict) |
| `make test` | `pytest` |
| `make coverage` | `pytest --cov` |
| `make check` | lint + typecheck + test |
| `make clean` | remove caches/artifacts |

## Project plan

The full design — architecture, corpus, SFT/RL, serve + agent, evaluation, and reasoning-effort — lives in the Backlog document `doc-001` under `backlog/docs/`.
