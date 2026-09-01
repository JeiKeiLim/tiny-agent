# Kestrel

A small-scale, modern **agentic LLM training pipeline** on [MLX](https://github.com/ml-explore/mlx), built to learn the full flow end-to-end:

```
pretrain → long-context → SFT → RL → serve + agent → eval
```

Kestrel trains a **pair of small decoder-only models (50M and 150M)** from scratch and compares them at every stage. The goal is understanding the modern training pipeline and the 50M↔150M scaling comparison — not raw capability.

## Two tracks

- **Track A (from scratch):** train Kestrel-50M and Kestrel-150M through the whole pipeline with full fine-tuning.
- **Track B (PEFT / LoRA, planned):** fine-tune a pretrained base (default Qwen3-1.7B) with LoRA and LoRA-like methods (QLoRA, DoRA, adapters) behind a pluggable `PEFTMethod` interface. Shares the SFT / RL / serve / eval code with Track A.

## Pipeline (Track A)

1. **Tokenizer** — byte-level BPE, 16k vocab (shared by both sizes).
2. **Pretrain** — Chinchilla-capped (~1B tokens for 50M; a single pass over the corpus for 150M) on FineWeb-Edu + Python + synthesized JSONL.
3. **Long-context** — RoPE interpolation + staged continuation (4k → 8k → 16k).
4. **SFT** — hybrid tool-calling + chain-of-thought data, unified chat template.
5. **RL** — GRPO. **RL-A** (pure math) + **RL-B** (synthetic agentic tool environment).
6. **Serve + agent** — MLX `generate()` + optional OpenAI-compatible server; a robust single-loop agent.
7. **Eval** — a 4-checkpoint scorecard per size (pretrain → +SFT → +RL-A → +RL-B).

## Status

**Milestone M0 (Foundation) is complete:**

- **Scaffolding** (`TASK-001`) — the `src/kestrel/` package, the generic YAML→Pydantic config loader, `ModelConfig`, sample configs, and the uv environment.
- **Code quality infra** (`TASK-004`) — strict Pydantic configs, Ruff + mypy (strict) + pytest, and the `make check` gate.
- **BPE tokenizer** (`TASK-003`) — training-data prep, 16k-vocab byte-level BPE training, round-trip + byte-coverage verification (as tests), and an interactive explorer. Guarantees a lossless round-trip for any byte sequence (all 256 byte-tokens are in the vocab).
- **Kestrel model** (`TASK-002`) — the decoder-only transformer (`model/kestrel.py`: pre-norm RMSNorm, RoPE, GQA, SwiGLU, tied embeddings), model I/O (`model/io.py`: `load(config, checkpoint)` + `save(model, path)`), and a smoke-test CLI (`scripts/check_model.py`).

**Milestone M1 (Pretraining) is implemented and currently being validated** with a Kestrel-50M run:

- **Corpus builder** — pluggable weighted text mix (`corpus/`) producing document-level JSONL + per-split manifests; built by `scripts/build_corpus.py` (currently a ~12 GiB corpus in `data/corpus-12g`).
- **Pretrain dataset** — JSONL documents → tokenized `(input, target)` batches with document-aware mixing (`data/pretrain_dataset.py`).
- **Trainer** — shared MLX training loop (`train/trainer.py`) with a live `run.jsonl` log, `step_NNNNNN` / `best` / `final` checkpoints, resume from a full checkpoint, and a retention policy.
- **Pretrain entry point** — `scripts/run_pretrain.py` with `configs/kestrel/50m/pretrain.yaml` and `configs/kestrel/150m/pretrain.yaml`, plus `generate()` (`model/generate.py`) for autoregressive sampling.
- **Pretrain evaluation** — read-only `scripts/eval_pretrain.py` reports token-weighted held-out loss, perplexity, bits/token, and per-domain loss for saved checkpoints.

**Milestone M2 (SFT validation) is partially implemented:** the SFT chat renderer, masked SFT dataset, SFT trainer, raw SFT source prep (`scripts/run_prepare_sft.py`), held-out SFT eval bundle prep (`scripts/run_prepare_sft.py --source eval`), SFT mixture builder (`scripts/run_build_sft_mixture.py`), SFT training entry point (`scripts/run_sft.py`), interactive SFT chat (`scripts/chat_sft.py`), and the inference-only SFT eval harness + scorecard (`scripts/run_eval_sft.py`). Data prep supports public assistant, GSM8K, local rule-based tool, public tool, and an optional internal LLM source. The SFT chat renderer exposes `row.tools` as a compact loss-masked system block so tool schemas are part of both training and eval prompts. The internal LLM source is disabled by default and reads endpoint, API key, and model name only from environment variables named in `configs/kestrel/sft_data.yaml`; `.env.example` lists the required variable names, and `.env` is gitignored.

**Planned, not yet built:** long-context, RL, serve + agent, and Track B (PEFT/LoRA).

## Repo layout

```
tiny-agent/
  configs/
    tokenizer/        # train.yaml, train_data.yaml
    kestrel/          # from-scratch (Track A) — family of 2 sizes
      corpus.yaml     # pretrain corpus (output: data/corpus-12g)
      sft_data.yaml   # SFT raw-source data prep, held-out eval bundle, and mixture recipe
      50m/            # model.yaml, pretrain.yaml, sft.yaml, eval_sft.yaml
      150m/           # model.yaml, pretrain.yaml, sft.yaml
  src/kestrel/
    common/           # config (YAML→Pydantic), logging
    model/            # config.py, kestrel.py, io.py (load/save), generate.py
    tokenizer/        # config.py, train.py, visualize.py
    corpus/           # config.py, builder.py (pluggable corpus builder)
    data/             # tokenizer/pretrain/SFT datasets, chat rendering, SFT source prep/mixture
    train/            # trainer.py, pretrain.py, sft.py, checkpoint.py, rl/ (empty)
    tools/            # schema_sampler.py
    eval/             # pretrain.py, sft.py, tool_calling.py (checkpoint evaluation + SFT scorecard)
    peft/  env/  agent/  serve/   # planned (empty packages)
  scripts/            # build_corpus.py, check_model.py, run_pretrain.py,
                      # eval_pretrain.py, visualize_tokenizer.py, run_prepare_sft.py,
                      # run_build_sft_mixture.py, run_sft.py, chat_sft.py,
                      # run_eval_sft.py
  tests/
  data/  checkpoints/ # runtime artifacts (gitignored)
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (pinned in `.python-version`).

```bash
make install     # create .venv and install dependencies (runtime + dev)
make check       # lint + typecheck + test
```

> Note: in some environments `uv` needs `--system-certs` for network operations (TLS).

The pretraining workflow is: prepare tokenizer data → train tokenizer → build corpus → check model → pretrain (resumable).

### Tokenizer training data

The BPE tokenizer trains on a representative sample (web + code + JSONL) assembled by a config-driven script:

```bash
uv run python -m kestrel.data.prepare_tokenizer_data   # -> data/tokenizer_train/
```

Sources and total size are set in `configs/tokenizer/train_data.yaml` (default ~1 GB, tunable). The sample is a runtime artifact (gitignored) and is regenerated on demand; re-runs are fast because `datasets` caches the HuggingFace shards. In environments with a custom/corporate CA, the script uses `truststore` so the download works without manual cert setup.

### Tokenizer training

The byte-level BPE tokenizer (16k vocab, configurable) is trained on that sample with HuggingFace `tokenizers`. ChatML + tool-call special tokens are baked in at training time; the artifact is shared by both model sizes:

```bash
uv run python -m kestrel.tokenizer.train   # -> checkpoints/tokenizer/tokenizer.json
```

Vocab size, special tokens, and paths are set in `configs/tokenizer/train.yaml`. The artifact is a runtime output (gitignored) and is regenerated on demand.

The base alphabet is seeded with all 256 byte-tokens (the GPT-2/Qwen convention), so the tokenizer guarantees a **lossless round-trip for any byte sequence** — not just the text observed during training. Round-trip losslessness and raw-byte coverage are verified by `tests/test_tokenizer_verify.py` (part of `make check`); you can also check the trained artifact against any file directly:

```bash
uv run python tests/test_tokenizer_verify.py FILE [FILE ...] [--coverage]
```

An interactive explorer renders any text as color-blocked token spans with the token ids on the line below in matching colors (plus `:vocab`, `:specials`, `:id`, `:token`, `:file` commands). `--verbose` adds the full token/id/bytes/kind table:

```bash
uv run python scripts/visualize_tokenizer.py [--verbose]
```

### Corpus build

The pretrain corpus is a weighted mix (web + code + synthetic) written as document-level JSONL with per-split manifests:

```bash
uv run python scripts/build_corpus.py --config configs/kestrel/corpus.yaml [--force]
```

Sources, fractions, and the total-byte budget are set in `configs/kestrel/corpus.yaml`. The output is currently `data/corpus-12g` (temporary name; the canonical `data/corpus` rename is pending the active run). Components that already verify against their manifest are skipped; `--force` rebuilds.

### Model check

A standalone smoke-test CLI loads a model (random-init or from a checkpoint), runs a forward pass, and prints the param count, logits shape, CE loss, and top-k tokens:

```bash
uv run python scripts/check_model.py --config configs/kestrel/50m/model.yaml
```

Add `--checkpoint <path>` to load a trained checkpoint instead of random init, and `--generate` to sample text after the report. The current no-KV-cache generation path releases unused MLX allocator cache every `--clear-cache-every N` generated tokens (default `64`, `0` disables) to bound retained cache memory during long generations. On an untrained model the loss is ~ln(vocab) (~9.7 for 16k) and the top tokens are gibberish — expected, not a bug.

### Pretrain

```bash
uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain.yaml
```

The 50M config is Chinchilla-capped (~1B tokens, a prefix of the corpus); the 150M config runs a single pass over the whole corpus. Both reference `configs/kestrel/corpus.yaml` and the trained tokenizer.

**Checkpoints and resume.** Training writes a live `run.jsonl` to the output dir (`checkpoints/pretrain/50m`) plus checkpoints: `step_NNNNNN` every `save_every` steps, `best` on a new best val loss, and `final` at the end. Each is a *full* checkpoint — `weights.npz`, `optimizer.npz`, `state.json`, config snapshots, and a `run.jsonl` snapshot — so any of them can be resumed:

```bash
uv run python scripts/run_pretrain.py --config configs/kestrel/50m/pretrain.yaml --resume <output_dir>/step_004000
# also: --resume <output_dir>/best  |  --resume <output_dir>/final
```

Retention: old `step_NNNNNN` dirs are pruned to `keep_latest_checkpoints` (5 in the 50M config); `best` is kept while `keep_best_checkpoint` is true. Weights-only checkpoint dirs (e.g. the archived `checkpoints/pretrain/archive-v0/`) are *not* resumable — resume requires the full checkpoint layout.

### Pretrain checkpoint evaluation

In-loop `val_loss` uses only `eval_iters` batches, so it is a training monitor rather than a final measurement. To evaluate a saved checkpoint over a larger validation sample:

```bash
uv run python scripts/eval_pretrain.py \
  --pretrain-config configs/kestrel/50m/pretrain.yaml \
  --checkpoint checkpoints/pretrain/50m/best \
  --max-tokens 1000000
```

The report includes token-weighted loss, perplexity, bits/token, evaluated tokens, and per-domain loss for `web`, `code`, and `synthetic`. `--max-tokens 0` evaluates the full split, `--json` emits machine-readable output, and `--generate` adds fixed greedy samples. Progress is printed to stderr every `--progress-every-tokens` tokens (default 100000, `0` disables) and includes an estimated percentage when the corpus manifest provides token totals, so `--json` stdout stays parseable. The command is read-only and can be run against a complete checkpoint while training continues.

### SFT data prep

Raw SFT sources are prepared into `data/sft/raw/` by:

```bash
uv run python scripts/run_prepare_sft.py --source all
uv run python scripts/run_prepare_sft.py --source assistant
uv run python scripts/run_prepare_sft.py --source gsm8k
uv run python scripts/run_prepare_sft.py --source tool
uv run python scripts/run_prepare_sft.py --source public_tool
uv run python scripts/run_prepare_sft.py --source internal_llm
uv run python scripts/run_prepare_sft.py --source eval
```

Sources, row targets, and output paths are set in `configs/kestrel/sft_data.yaml`. `target_rows` is the number of valid rows to write after conversion and context filtering; assistant prep uses `assistant.max_candidate_rows` as a safety cap on raw rows inspected. The internal LLM source is disabled by default; enable it in a local config copy and export the environment variables named in `.env.example` before running it. Progress and drop-debugging behavior are controlled by `internal_llm.progress_every`, `internal_llm.debug_drops`, and `internal_llm.debug_drop_limit`.

`--source eval` writes the held-out scorecard eval bundle to `data/sft/eval/`: assistant rows from Smol-SmolTalk `test`, GSM8K rows from `test/main`, and the local tool seen/unseen/no-call/missing-info eval sets. The eval bundle is separate from the training mixture and is controlled by the `eval` section of `configs/kestrel/sft_data.yaml`.

### SFT mixture builder

The raw sources are combined into the SFT training mixture by:

```bash
uv run python scripts/run_build_sft_mixture.py --config configs/kestrel/sft_data.yaml
```

The builder reads the per-source files from `data/sft/raw/`, writes `data/sft/mixture/sft-50k.jsonl` and `data/sft/mixture/manifest.json`, and records requested/actual counts, source hashes, seed, and config hash. The `mixture` section of `configs/kestrel/sft_data.yaml` controls the default 50k recipe, the no-internal-LLM fallback recipe, shuffle seed, and deficit policy.

### SFT training

The SFT phase loads a pretrain checkpoint and trains on the masked SFT dataset:

```bash
uv run python scripts/run_sft.py --config configs/kestrel/50m/sft.yaml
```

The 50M config expects the mixture file at `data/sft/mixture/sft-50k.jsonl`, produced by `scripts/run_build_sft_mixture.py`.

### SFT eval

The SFT scorecard evaluates checkpoints inference-only on the held-out bundle from `data/sft/eval/`:

```bash
uv run python scripts/run_eval_sft.py --config configs/kestrel/50m/eval_sft.yaml
```

It reports assistant sanity checks, GSM8K final-answer accuracy, local tool seen/unseen/no-call/missing-info metrics, and optional held-out pretrain perplexity. The committed 50M config scores the pretrain checkpoint plus the expected `5k`, `20k`, and `50k` SFT checkpoints; missing checkpoints are skipped by default so the pretrain baseline can be scored before SFT training completes. `--max-rows N` limits each eval set for smoke runs, `--output PATH` overrides the scorecard path, and `--skip-perplexity` skips the corpus perplexity measurement. The `generation.clear_cache_every` config field controls MLX allocator-cache release cadence during generation (default `64`, `0` disables).

### SFT chat

A manual multi-turn chat CLI renders prompts with the same SFT chat template used during training:

```bash
uv run python scripts/chat_sft.py --checkpoint checkpoints/sft/50m/final
```

It is for inspection only and does not expose tool calling.

## Code quality

Strict, standard tooling — run it all with `make check`:

- **Config validation** — configs are [Pydantic](https://docs.pydantic.dev/) models in **strict mode** (`extra="forbid"`): a mistyped value (e.g. `n_layers: "15"`) or an unknown key raises a clear `ValidationError`.
- **Lint + format** — [Ruff](https://docs.astral.sh/ruff/) (`ruff check` + `ruff format`).
- **Static types** — [mypy](https://mypy.readthedocs.io/) in `strict` mode.
- **Unit tests** — [pytest](https://docs.pytest.org/) + [pytest-cov](https://pytest-cov.readthedocs.io/). Tests live in `tests/`, mirror the source layout, and test the *code* (not config data values).

Makefile targets:

| Target | Runs |
|--------|------|
| `make help` | list available targets |
| `make install` | `uv sync --all-groups` |
| `make sync` | alias for `install` |
| `make format` | `ruff format` + `ruff check --fix` (src, tests, scripts) |
| `make lint` | `ruff check` + `ruff format --check` (src, tests, scripts) |
| `make typecheck` | `mypy src scripts` (strict) |
| `make test` | `uv run pytest` |
| `make coverage` | `pytest --cov=kestrel --cov-report=term-missing` |
| `make check` | lint + typecheck + test |
| `make clean` | remove caches/artifacts |

## Project plan

The full design — architecture, corpus, SFT/RL, serve + agent, evaluation, and reasoning-effort — lives in the Backlog document `doc-001` under `backlog/docs/`.
