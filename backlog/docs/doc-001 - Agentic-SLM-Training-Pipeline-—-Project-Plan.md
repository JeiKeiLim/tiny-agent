---
id: doc-001
title: Agentic SLM Training Pipeline — Project Plan
type: specification
created_date: '2026-08-19 22:12'
updated_date: '2026-08-21 07:04'
---
# Agentic SLM Training Pipeline — Project Plan

_Status: Draft / active discussion. No implementation started. This captures the plan and decisions from our discussion so far. Update as we converge. Tasks/milestones will be broken out only after the plan is finalized._

## 1. Vision

Build a **full, modern LLM training pipeline** — pretraining, supervised fine-tuning, and reinforcement learning — at a **small scale** that fits on a single machine. The primary goal is **learning**: understand each stage of how a modern, agentic, reasoning-capable LLM is actually trained. The end goal is a **custom agent built on the from-scratch (Track A) model**.

Key principle: **small scale, not a degraded approach.** We use the same modern techniques used at scale (BPE tokenization, transformer pretraining, SFT on tool-calling data, R1-style RL with verifiable rewards, chain-of-thought) — just at a smaller model/data scale that is feasible on the hardware.

## 2. Hardware & Feasibility

- **Machine:** Apple M4 Pro, 48GB unified memory.
- **Memory:** Not the constraint. A 150M model is tiny; even a 1B model fits comfortably.
- **Compute / time:** The real constraint. Pretraining 50M–150M needs far fewer tokens → hours to days (feasible). Pretraining ~1B to competence needs ~10B–20B tokens → weeks to months (not practical).
- **Throughput (rough, to be benchmarked on the actual machine):** 150M ≈ ~11,000 tok/s; 50M ≈ ~30,000 tok/s.

## 3. Two-Track Structure

**Primary focus: Track A.** The custom agent is built on the from-scratch Track A model. **Track B is optional** — by the time Track A is done, everything Track B would teach is already learned, so it can be skipped (or used later as a shortcut / cross-check).

### Track A — From scratch (the focus, and the agent)
- Train **50M and 150M** models, **same architecture, same data, same tokenizer**, from random init.
- Goal: understand pretraining end-to-end and observe **scaling** (loss/quality vs. size).
- **Both models go through the full pipeline and each becomes an agent** — we compare **50M-agent vs 150M-agent** at every stage (that's the scaling study).
- Includes training **our own BPE tokenizer**.
- Full pipeline: pretrain → long-context extension → SFT → RL.

### Track B — PEFT / LoRA (optional, distinct learning goal)
- **Goal:** learn **parameter-efficient fine-tuning (PEFT)** — LoRA (core) + LoRA-like methods (QLoRA, DoRA, adapters) — which Track A (full fine-tuning) does **not** cover.
- **The two tracks now differ by fine-tuning *method*, not just base:**
  - Track A = **full fine-tuning** (update all weights) on our from-scratch 50M/150M.
  - Track B = **PEFT** (freeze the base, train only low-rank adapters ≈ 1% of params) on a pretrained base.
- **Base:** a pretrained MLX model (default **Qwen3-1.7B**; QLoRA's 4-bit base unlocks up to ~27B on 48GB). Public `mlx-community` models.
- **Architecturally pluggable (adapter pattern):** one `PEFTMethod` interface; LoRA / QLoRA / DoRA / Adapter each implement it; a **method-agnostic trainer** resolves the method via a registry. Adding a method = implement the interface + register it. The LoRA impl **mirrors the `mlx_lm` interface** (same config, `lora_a`/`lora_b`, `alpha/rank` scaling, adapter format) so adapters are interchangeable with `mlx_lm.lora`.
- **Full pipeline (mirrors Track A):** SFT → RL (GRPO on the adapters) → serve → eval. Same shape as Track A, different fine-tuning method.
- **Cross-track reuse:** the PEFT framework is **shared** — it can also fine-tune our own 50M/150M, giving a **clean ablation** (PEFT vs full-FT, same base/arch/tokenizer, only the method differs).
- **Skippable:** optional; the core from-scratch learning goal is Track A.

## 4. Pipeline (Phases) — Track A

| # | Phase | Model | What | Notes |
|---|-------|-------|------|-------|
| 0 | Pretrain | 50M, 150M | Train from scratch on the corpus | Own BPE; plot loss + scaling |
| 1 | Long-ctx extension | 50M, 150M | Continue pretraining at 4k→8k→16k (long docs) | Still pretraining; small token budget |
| 2 | SFT | 50M, 150M | Fine-tune on tool-calling + reasoning data | **Same data for both** → clean comparison |
| 3 | RL (logic) | 50M, 150M | R1-style RL, verifiable math rewards (GRPO, RLOO fallback) | Slowest phase |
| 4 | Serve + Agent | 50M, 150M | Serve (OpenAI-compatible) + custom agent loop | Custom Python loop, not a framework |
| 5 | Eval | 50M, 150M | Tool-calling accuracy (BFCL-style) + math accuracy | Before/after RL, both sizes |

_Track B (optional) = phases 2–5 starting from a pretrained base instead of phase 0._

## 5. Decisions Made So Far

- **Framework: MLX end-to-end** (fastest on M4 Pro, one coherent framework).
- **Primary focus: Track A** (from scratch); the agent is built on the Track A model. Track B optional.
- **Track B (PEFT/LoRA):** distinct learning goal = parameter-efficient fine-tuning (LoRA core + QLoRA/DoRA/adapters) via a pluggable `PEFTMethod` interface (adapter pattern); full pipeline (SFT→RL→serve→eval) on a pretrained base (default Qwen3-1.7B, QLoRA unlocks ~27B); PEFT framework shared with Track A for a clean PEFT-vs-full-FT ablation; LoRA mirrors the `mlx_lm` interface. See §3.
- **Scaling pair:** 50M and 150M both run the **full pipeline** (pretrain → long-ctx → SFT → RL → agent → eval) and are compared at every stage — neither is a throwaway.
- **Full pipeline:** pretrain → long-context extension → SFT → RL.
- **RL is included** — R1-style, verifiable math rewards, GRPO (RLOO fallback).
- **Focus:** agentic (tool calling) + logical (reasoning / chain-of-thought).
- **Tokenizer:** our own byte-level BPE, 16k vocab (configurable). See §7.
- **Pretraining corpus:** FineWeb-Edu base + Python code + synthesized JSONL, config-driven pluggable corpus-builder; ~1B tokens (configurable). See §8.
- **Model architecture:** modern decoder-only (RMSNorm, RoPE, SwiGLU, **GQA**, tied embeddings); 50M = 15L×512, 150M = 32L×640 (deeper + narrower + GQA per research); 2k base context with built-in extension to 8–16k. See §9.
- **Tool calling:** structural JSON; validity defenses = simple format + lenient parse/retry, constrained decoding as stretch. See §10.
- **SFT:** hybrid data (real base for generalization + synthetic generator for our tools/edge cases) + GSM8K CoT + minimal general; unified format, single 1-epoch mixed run, full FT; split eval (our-tools + unseen-tools). See §11.
- **RL:** GRPO (KL to frozen SFT ref); RL-A = pure math (canonical R1, control), RL-B = synthetic agentic tool env (bounded diverse tools + diverse generated tasks + verifiable outcomes, multi-turn); both branch from the SFT checkpoint. See §12.
- **Evaluation:** held-out sets (GSM8K test, BFCL unseen, held-out generated tasks, perplexity) run at every stage; scorecard compares pretrain→SFT→RL-A→RL-B for BOTH sizes equally. See §13.
- **Reasoning-effort control:** a training-time property (model trained to think-first via SFT-CoT + RL-A, both config-toggleable, post-pretraining); exposed at inference as prompt steer (primary) + thinking budget (per-level ceiling, graceful cutoff); best-of-N out of scope (serving feature). See §16.
- **Serve + Agent:** MLX `generate()` (in-process core) + optional OpenAI-compatible FastAPI server; agent = separable `agent/` harness (robust parsing/repair, error feedback, context mgmt, loop control, tracing, effort integration); single robust loop (no multi-agent/planning/memory/RAG); agent eval = end-to-end task success; HF upload optional capstone. See §14.
- **Repo structure:** single `src/kestrel/` package (codename **Kestrel**), both tracks unified (track = config + entry-point axis, not a dir boundary); thin `model/io.py` factory; `train/` umbrella + `data/`; configs by model; checkpoint handoff enables incremental build. See §6.
- **Code quality:** strict, standard tooling from day one — configs are **strict Pydantic models**; Ruff (lint + format), mypy (strict), pytest + pytest-cov; a `Makefile` runs it all (`make check` = lint + typecheck + test). See README.
- **Agent:** custom Python loop (send msgs + tool defs → parse tool call → run function → feed result back → repeat), not an agent framework.
- **Code authorship:** the agent writes all the code; the human learns by reading the code and watching the runs.

## 6. Code Organization & Modularity (decided)

**Single package, both tracks.** Track A and Track B are **not** separate top-level packages — they share almost all the code (SFT, RL, serve, eval, agent, tools, env, PEFT). The only differences are the **base model** (from-scratch Kestrel vs. a pretrained one) and the **fine-tuning method** (full FT vs. PEFT) — both are config + method-selection concerns, not code boundaries. So the "track" is a **config + entry-point axis**, not a directory boundary.

**Codename: Kestrel.** The package (and our model) is `src/kestrel/`.

Design constraints:
- **Config-driven:** every tunable (model shape, vocab, lr, batch, dataset paths, context length, RL/PEFT hyperparams, agent settings) lives in **YAML configs** loaded into **strict Pydantic models** (mistyped values / unknown keys are rejected). Changing a run = editing a YAML, not code.
- **Thin model interface:** our model and a pretrained one are both just MLX `nn.Module`s; `model/io.py` exposes one `load(config)` factory (random-init / from-checkpoint / pretrained) + `save(model, path)`. Every phase/eval/serve gets its model the same way.
- **`train/` umbrella:** all training loops (`pretrain`, `long_context`, `sft`, `rl`) + one shared `trainer.py` (optimizer, full-FT/PEFT selection, step loop, checkpointing); all dataset prep lives in `data/`.
- **Checkpoint handoff:** phases are decoupled by checkpoints on disk — each phase loads a checkpoint, runs, saves a checkpoint; the next phase points its config at it. This is what makes **incremental, step-by-step build/test** work.
- **Entry points** in `scripts/` so each phase is a single command.

Layout:

```
tiny-agent/
  configs/                          # by model
    kestrel/                        # our from-scratch model (Track A) — family of 2 sizes
      tokenizer.yaml  corpus.yaml   # shared by both sizes
      50m/    model.yaml  pretrain.yaml  long_context.yaml
              sft.yaml  rl_a.yaml  rl_b.yaml  serve.yaml  agent.yaml  eval.yaml
      150m/   model.yaml  pretrain.yaml  long_context.yaml
              sft.yaml  rl_a.yaml  rl_b.yaml  serve.yaml  agent.yaml  eval.yaml
              peft_sft.yaml  peft_rl.yaml        # Track B ablation: PEFT on our 150M
    qwen3_1_7b/                     # pretrained base (Track B)
      model.yaml  peft_sft.yaml  peft_rl.yaml  serve.yaml  agent.yaml  eval.yaml

  src/kestrel/
    common/       # config (YAML→Pydantic), logging, utils
    model/        # config.py, kestrel.py (our model), pretrained.py (Qwen3 loader),
                  # io.py  →  load(config) [random-init / checkpoint / pretrained], save(model, path)
    tokenizer/    # (Track A) train BPE
    corpus/       # (Track A) pluggable corpus builder
    data/         # pretrain_dataset.py, sft_prepare.py, sft_synthetic.py
    train/        # trainer.py (shared: optimizer, full-FT/PEFT selection, step loop, checkpoint)
                  # pretrain.py, long_context.py, sft.py,
                  # rl/ (grpo.py, rloo.py, reward.py, train.py)
    peft/         # method.py (PEFTMethod iface), lora.py, qlora.py, dora.py, adapter.py, registry.py
    tools/        # shared tool registry + impls + task-suite generator
    env/          # agentic environment: execute calls, state, outcome, reward
    agent/        # loop.py, client.py, parse.py, context.py, trace.py
    serve/        # generate.py (MLX), server.py (OpenAI-compatible FastAPI)
    eval/         # run.py, metrics.py, math.py, tool_calling.py, agent_task.py, perplexity.py, datasets/

  scripts/        # Track A: run_tokenizer run_corpus run_pretrain run_longctx run_sft run_rl_a run_rl_b
                  # Track B: run_peft_sft run_peft_rl
                  # shared:  run_eval run_serve run_agent
  data/  checkpoints/  outputs/
```

**Incremental build order (each milestone independently runnable + testable):**
0. `common/` + `model/` + `tokenizer/` — model instantiates; forward pass + tokenizer round-trip.
1. + `corpus/` + `data/pretrain_dataset.py` + `train/pretrain.py` + `train/trainer.py` → **pretrain** (loss ↓, coherent text).
2. + minimal `eval/` (perplexity + generate) → validate the base.
3. + `tools/` + `env/` + `data/sft_*` + `train/sft.py` → **SFT** (follows instructions / tool calls).
4. + `train/rl/` → **RL** (reward ↑).
5. + `agent/` + `serve/` → **serve + agent** (end-to-end task success).
6. + full `eval/` → **scorecard** across all checkpoints.
7. + `peft/` + `model/pretrained.py` → **Track B (LoRA)** (Qwen3 + 150M ablation).

## 7. Tokenizer (decided)

### Track A — our own byte-level BPE
- **Algorithm:** byte-level BPE. Text → UTF-8 bytes (256 base symbols) → repeatedly merge the most frequent adjacent pair until the vocab is full. **No OOV possible** — any text is a byte sequence.
- **Vocab size: 16k, configurable** (`configs/track_a/tokenizer.yaml`). Small vocab is deliberate: embedding params scale with `vocab × hidden`, so 16k keeps the embedding a modest fraction of a 50M model (a 48k vocab would be ~half the params).
- **Same tokenizer for 50M and 150M** — clean scaling comparison (only model size changes).
- **Token IDs:** sequential integers = an index into the embedding matrix. Order: special tokens (PAD, EOS) → the 256 byte-tokens → merged tokens in merge-frequency order. IDs have **no Unicode meaning**.
- **Non-English / multilingual:** handled uniformly as bytes (Korean/Chinese char = 3 bytes, emoji = 4). Those bytes merge into single tokens only if frequent in the training corpus; otherwise they stay as multiple byte-tokens. Always representable (no OOV), efficient only if the language is in the corpus. Language-agnostic by construction; UTF-8 is uniquely decodable so distinct characters never collide.
- **Special tokens:** minimal — PAD, EOS.
- **JSON readiness:** include JSON/code in the pretraining corpus and/or pass the JSON structural chars `{ } [ ] " : ,` as the BPE `initial_alphabet` so they stay clean single tokens.
- **Training data + order (standard pipeline):** a sample of the pretraining corpus (same distribution). Order: curate corpus → train BPE on a sample → tokenize the full corpus → pretrain.
- **Tool:** HuggingFace `tokenizers` (BPE trainer + ByteLevel pre-tokenizer).

### Track B (optional)
- If Track B is done, it reuses its pretrained base model's tokenizer + chat template (no new tokenizer).

## 8. Pretraining Corpus (decided)

**Purpose:** give Track A's model general language ability + JSON/code fluency (tool-calling *readiness*). The **exact tool-call format is learned in SFT, not here** — pretraining only needs the model to find JSON "natural."

**Base (~85%):** a pre-processed clean web corpus — **FineWeb-Edu** (quality-filtered, good for downstream instruction-following); alt **SlimPajama**. We **inherit** modern data cleaning (dedup, quality filter, language filter) rather than rebuild it.

**Code/JSON mix (~15%):**
- **Code (~10%):** sample from **The Stack / StarCoder Data** (`bigcode/starcoderdata`), **Python-weighted** (our agent + tools are Python). Yields function defs, control flow, and embedded JSON.
- **JSON (~5%):** **synthesized JSONL** — a script that `json.dumps` structured records (facts/QA, Wikipedia summaries/infoboxes, generated key-value records) one per line. Clean, well-formed JSON exposure.

**Token budget:** ~**1B tokens per model** (50M and 150M), **configurable** (`total_target_tokens`). Deliberately under Chinchilla-optimal for a learning run. Do a **short validation run first** (~50M tokens), then the full run. **Same corpus for both sizes** (clean scaling comparison).

**Pluggable architecture (room for hand-built):** the corpus is a **weighted list of components**, each with a swappable `source` (`hf` / `local` / `url` / `generator`). A single corpus-builder module samples each component, concatenates + shuffles, then tokenizes to the target. **Hand-built = point a component at your own file — no code change.**

```yaml
# configs/track_a/corpus.yaml
total_target_tokens: 1000000000     # ~1B — configurable
seed: 42
components:
  - name: web
    source: { type: hf, id: "HuggingFaceFW/fineweb-edu", split: "train" }
    sample_gb: 3.0
    weight: 0.85
  - name: code
    source: { type: hf, id: "bigcode/starcoderdata", config: "python" }
    sample_gb: 0.4
    weight: 0.10
  - name: json
    source: { type: local, path: "data/track_a/json_synthetic.jsonl" }
    sample_gb: 0.2
    weight: 0.05
```

## 9. Model Architecture (decided)

### Research findings (best practices for 50M–150M)
Sources: SmolLM / MobileLLM, a 19-config "optimal architecture for small LMs" study, and a compute-constraints paper.
- **Architecture family barely matters at this scale** — GPT-2 / LLaMA3 / Qwen3 land within ~2% at ~70M. Modern components (RMSNorm, SwiGLU, RoPE) are fine but **not** the differentiator.
- **Depth-over-width is the main lever**; keep **hidden ≥ ~512**. (SmolLM "prioritizes depth over width"; the study's best 70M was 32L×384.)
- **GQA** (fewer KV heads than Q heads) — standard for small models (SmolLM, MobileLLM); cuts attention params + KV memory.
- **Embedding tying** — good for small models.
- **2048 context, extended later** — SmolLM trains at 2048 and extends via a long-context fine-tune (validates our plan).
- **Smaller vocab helps** — embedding params scale with `vocab × hidden`; 16k is better for 50M than 48k.
- **Caveat:** match complexity to the training budget (we train ~1B tokens vs SmolLM's 600B) — don't over-depth an under-trained model; A/B depth.

### Architecture (modern decoder-only, LLaMA-style)
Pre-norm **RMSNorm**, **RoPE** positional encoding, **SwiGLU** FFN, **GQA**, **tied embeddings**, no biases, dropout 0.

### The two shapes (deeper + narrower + GQA)

| | 50M | 150M |
|---|---|---|
| Layers | 15 | 32 |
| Hidden | 512 | 640 |
| Q heads | 8 | 10 |
| KV heads (GQA) | 2 | 2 |
| FFN (SwiGLU) | 1408 | 1728 |
| Vocab | 16384 | 16384 |
| Context | 2048 | 2048 |
| **Params** | **~51M** | **~148M** |

- `count_params()` asserts the actual count so the labels are verified, not hand-waved.
- **A/B (open):** 150M deep (32L, research-aligned) vs shallower (~20L×768). We train on ~1B tokens (SmolLM used 600B), so 32L may be under-trained in depth — the loss curve decides. Config-driven.

### Context length (decided)
- **Base pretrain: 2048** — fast, learns language + the scaling comparison. Most tokens go here.
- **Long-context extension path (built in)** so the same model reaches the ~8–16k an agent needs:
  1. **RoPE** → position interpolation / NTK / YaRN to stretch beyond the trained length.
  2. **Staged continuation:** after the 2k base pretrain, a *short* continuation at 4k→8k→16k on a small token budget, using **genuinely long documents** (not stitched short ones, so the model learns true long-range attention).
  3. **FlashAttention** in the MLX loop → O(n) memory, so 8–16k is memory-feasible.
- **Ordering:** base pretrain @2k → long-ctx pretraining continuation (still pretraining) → SFT → RL, all at the extended length. (SFT/RL need the model to already handle the target context.)
- **`context_length` is a config param** + a separate `long_context` continuation config → extending is a YAML change, not code.
- No 32k/128k at this model size (small models can't effectively use it; not needed for tool calling).

## 10. Tool Calling (decided)

**Approach: structural (JSON) tool calling.**
- The (Track A) model emits a simple JSON tool call as text; the custom agent loop parses it (`json.loads`), runs the function, feeds the result back. No native `tools` API (we don't build one).
- **SFT is where the model learns the format;** the serving endpoint is where the interface lives.
- (If Track B is done, its base model's native tool template can be used instead.)

**Ensuring a small model completes a valid structure (prioritized defenses):**
1. **Simple, flat, consistent format (SFT):** one-level JSON object, few short fields, clear delimiters + a stop token that ends the call. Train on many clean examples.
2. **Lenient parsing + retry/repair:** tolerant JSON extraction (missing closing brace, trailing comma); on failure, retry with a "valid JSON only" nudge, capped at N tries. The practical safety net most small-model agents rely on.
3. **Constrained decoding (stretch goal):** mask logits so only schema-valid JSON tokens can be emitted — a hard guarantee. Not turnkey on MLX (outlines/xgrammar/llguidance are PyTorch/vLLM-centric); we'd implement a small JSON state-machine mask inside the MLX generation loop.

**Plan:** ship **1 + 2** first; add **3** as a stretch for an ironclad guarantee.

**Tool-design best practices (apply in SFT + agent phase):** research shows ~80% of agent failures are *tool design*, not the model — verb-first names, typed/constrained schemas (enums), a small tool surface (6–12 tools), and actionable (typed, retryable) error returns.

## 11. SFT (decided)

**Goal:** teach the model (both 50M and 150M, **same data**) to (1) call tools in our format, (2) reason step-by-step, (3) behave as a well-formed assistant.

**Data — hybrid (real + synthetic), unified format:**

| Slice | ~Share | Source |
|---|---|---|
| Tool-calling | 50–60% | **Real base** (ToolACE / xLAM-functions, converted to our format, filtered to simple 1–2 step + small tool subset) for diversity + **generalization to unseen tools**, + **synthetic layer** for our tools + edge cases. real:synthetic ≈ 50/50 (tunable). |
| Reasoning / CoT | 30–40% | GSM8K train (core, matches capacity) + optional MetaMathQA / OpenThoughts. |
| General instruction | ~10% | Small permissive set, kept minimal. |

**Synthetic generator (`sft/synthetic.py`), rule-based (no teacher model):**
- **Shared tool registry** — one source of truth used by both the generator and the agent at inference. Defining tools is for *data generation* + *matching our agent*, **not** a limit on what the model can call: the model learns the tool-agnostic skill, and generalization to unseen tools comes from the diverse real base.
- Per tool: query templates + parameter sampling → fill query → emit exact JSON call → mock result → templated final answer.
- Vary: number of tools in context (3–8); 1-step (core) + some 2-step (150M).
- **Negatives:** "no tool needed" + near-miss tool selection.
- Wrap in our chat template; **validate every example** (schema check); seeded + config-driven.

**Combine + train:**
- All sources → **same chat template** → one unified, shuffled, ratio-balanced dataset (~10k total).
- **Single mixed run, 1 epoch** (small models overfit fast). Downsample real to target; generate synthetic to target.
- **Full fine-tuning** (both sizes small; LoRA only if we want to preserve weights).
- Fallback: light curriculum (general+CoT → tool-calling) if the mixed run doesn't balance.

**Eval (verifies both halves separately):**
- **Our-tools accuracy** (what synthetic taught) **+ unseen-tools accuracy** (held-out tools not in training, e.g., BFCL unseen split — what the real base taught). Both high → hybrid working.
- Held-out GSM8K test for reasoning.

**Config:** `configs/track_a/sft.yaml` (ratios, sizes, epoch, LR). `sft/prepare.py` builds the unified JSONL (tagged by source); `sft/train.py` trains.

## 12. RL (decided)

**Goal:** after SFT, improve reasoning (RL-A) and agentic tool-use (RL-B) with RL on verifiable rewards. **Both 50M and 150M run both branches** (scaling pair).

**Algorithm — GRPO** (R1's method), for both branches:
- Per prompt/task, sample a **group of G** completions/rollouts (G ≈ 8–16).
- **Advantage = (r − group mean) / group std** → group-relative (GRPO is critic-free by design — no separate value network to train).
- PPO-style clipped objective + **KL penalty to a frozen SFT reference** (prevents collapse).
- **RLOO** as a simpler fallback.

**RL-A — pure math (canonical R1), no tools:**
- GSM8K *train*; **reward = correct final answer** (numeric/exact match); no reward model.
- Trains **internal reasoning** (the R1 effect). Simplest build (single-turn, no environment). Serves as the **control**.
- Filter prompts where all G rollouts are all-correct or all-wrong (no variance → no signal).

**RL-B — synthetic agentic tool environment (the big build):**
- **Bounded diverse tool set** (not omnipotent Python): `calculator`, `unit_converter`, `date_math`, `lookup`, `db_query` (~4–6 tools) → **tool selection is a real skill**.
- **Diverse generated task suite** — math is *one* category among many (conversions, dates, lookups, db, multi-step, "no tool needed"); each task has a **verifiable outcome by construction**.
- **Multi-turn rollouts:** model → tool call → env executes → result → model → … → final answer. **Reward = outcome correctness.**
- **Reuses SFT infra** (shared tool registry + templated task generator); RL-B wraps it in a multi-turn RL loop + outcome reward.
- Short horizons (2–4 calls), modest steps. **Biggest single build in the project.**

**Branching:** RL-A and RL-B both **start from the same SFT checkpoint** (comparable endpoints; less compute than stacking).

**Expectation:** modest capability bump at this scale; the value is learning the RL pipeline end-to-end + a measurable before/after.

## 13. Evaluation (decided)

**Purpose:** measure what each pipeline stage contributes — **for both sizes equally** (the central artifact of the project).

**Comparison ladder (per size, 4 checkpoints):** pretrain-only → +SFT → +RL-A → +RL-B. (RL-A/RL-B branch from the SFT checkpoint.)

**Held-out eval sets — carved out BEFORE the data pipeline is built; all automatic/verifiable:**
- **Math/reasoning:** GSM8K *test* (+ optional 2nd held-out math set, e.g., SVAMP/AQUA, for generalization). Metric: numeric/exact match.
- **Agentic/tool:** **BFCL *unseen* functions** (generalization to unseen tools; AST/execution-verified) + **held-out generated task suite** (the specific RL-B workflow).
- **General/sanity:** **perplexity** on held-out text (pretrain quality) + a tiny general-instruction check.

**Principles:** held-out from training; **consistent** prompts + decoding (greedy or fixed temp+seed) + metrics across checkpoints; fully automatic; one `eval/` harness (`eval/run.py` + `eval/metrics.py` + `eval/datasets/`).

**Scorecard — TWO symmetric tables, 50M and 150M, both fully tested at every stage:**

| 50M | PPL ↓ | GSM8K-test | BFCL-unseen | Held-out tasks |
|---|---|---|---|---|
| pretrain-only | ✓ | (low) | (low) | (low) |
| + SFT | ✓ | ✓ | ✓ | ✓ |
| + RL-A | ✓ | ✓ | ✓ | ✓ |
| + RL-B | ✓ | ✓ | ✓ | ✓ |

| 150M | PPL ↓ | GSM8K-test | BFCL-unseen | Held-out tasks |
|---|---|---|---|---|
| pretrain-only | ✓ | (low) | (low) | (low) |
| + SFT | ✓ | ✓ | ✓ | ✓ |
| + RL-A | ✓ | ✓ | ✓ | ✓ |
| + RL-B | ✓ | ✓ | ✓ | ✓ |

Pretrain-only is tracked mainly by **perplexity** (it doesn't know our answer/tool format yet); capability metrics become comparable from SFT onward.

**Build early** (right after SFT data prep) so we measure at each stage, not retroactively. **Inference-only on tiny models → ~5–20 min per checkpoint; ~1–2 h for the full 8-checkpoint scorecard (both sizes)** — cheap relative to training.

## 14. Serve + Agent (decided)

**Serving the trained model (both 50M and 150M):**
- **Core = MLX `generate()`:** our training code already produces a `generate(prompt) -> text` function (needed for eval + RL rollouts anyway). The agent/eval call it **in-process** — simplest, fastest, single machine.
- **Optional: OpenAI-compatible HTTP server** — a thin FastAPI wrapper exposing `/v1/chat/completions` around `generate()`. Makes the model usable like any OpenAI endpoint (lets external clients / opencode plug in).
- **Not the fit:** Ollama (needs GGUF + arch recognition — poor fit for a custom small model); Unsloth (a *training* tool, not a server).
- **Why our own:** custom architecture (no off-the-shelf tool recognizes it) + reuses inference code we already need.

**The agent = model-as-brain / agent-as-harness:**
- The *intelligence* (reasoning, planning, tool selection) is the **model's** job (trained via SFT/RL). The **agent** is a robust **harness** around it — for a small, imperfect model, the harness is what makes it usable.
- **Separable `agent/` module:** depends only on (a) a model-call interface (`generate()` or the HTTP endpoint), (b) the tool registry, (c) the environment. Never touches model weights or training code → swappable/improvable independently; swap model/checkpoint without touching it.

**Agent robustness features (compensate for a small model's weaknesses):**
- **Core loop:** perceive → model decides → execute tool → observe → repeat → final answer.
- **Robust tool-call parsing + repair** (from §10): lenient JSON extraction + retry-with-"valid JSON only" nudge, capped.
- **Error handling with actionable feedback:** feed a structured, retryable error back (not a crash) so the model self-corrects.
- **Context management:** 2k→16k window; truncate/summarize old tool results, keep recent turns.
- **Loop control / termination:** max-steps cap, stall/repeat detection, distinguish "final answer" from "gave up."
- **Reasoning-effort integration:** the `effort` knob (§16) plugs in here.
- **Tracing / observability:** log every step (model output, tool call, result) — feeds the agent-eval diagnostics.
- **Tool registry + environment interface:** shared registry (from SFT/RL-B) + env that executes tools and knows ground-truth outcomes.

**Agent eval (end-to-end task success):**
- Distinct from **static tool-calling** (BFCL-unseen: does it emit the right call?). The **agent eval** runs the model in the actual agent loop on **held-out tasks** (separate from RL-B training tasks) and measures **task success** + diagnostics (valid-call rate, steps-to-completion, "no tool needed" correctness).
- Reuses: agent loop + RL-B environment + held-out tasks. No new infra.
- **Build the agent loop early** (it's an inference-side wrapper, independent of training) so it can be pointed at any checkpoint for the mid-pipeline scorecard.

**Scope line (not over-building):** single, robust agent loop. **No** multi-agent orchestration, complex planning/replanning, long-term memory, or RAG — a small model can't drive those anyway.

**Bonus plug-in test:** because serving is OpenAI-compatible, we can point **opencode** (or any OpenAI client) at the local endpoint and drive it with our model. Expect a poor job (weak model + trained on *our* tool format, not opencode's protocol) — a fun "how far can we push it" experiment + serving sanity check, not a working opencode agent.

**Hugging Face (optional capstone):** push a small repo per model — weights (safetensors, converted from MLX) + `config.json` + our model code + tokenizer, via a `trust_remote_code`-style setup so the custom arch is loadable.

## 15. Open Questions (still to decide)

- **150M depth A/B:** 32L (research-aligned) vs ~20L×768 (shallower, safer for 1B tokens) — decide via the loss curve.

## 16. Reasoning / "Reasoning Effort" (decided)

- **Training (the real substance):** the model learns to *think before answering* via **SFT (CoT data) + RL-A (R1)** — RL rewards the correct *final answer*, driving the model to develop its own reasoning (the R1 effect). Without this training there's no "thinking" to control — the model would just answer directly. Both components are **config-toggleable** (SFT CoT slice; RL-A branch) and both are **post-pretraining** phases, so toggling reasoning only re-runs SFT/RL (cheap) — never re-pretraining.
- **Inference knob = prompt steer + thinking budget** (the **inference engine** handles all the mechanics):
  - **Prompt steer (primary lever):** the effort level maps to a prompt ("think step by step" for high, "answer directly" for low). Since the model is trained to stop itself via a "done thinking" marker, the **prompt is what actually changes thinking length** in the normal case.
  - **Thinking budget (per-level ceiling):** set *low* for low-effort (actively truncates → brevity), *high* for high-effort (allows depth). It's a backstop the model doesn't "feel" — the engine counts tokens and, if the cap is hit, **steers to a final answer** (inject "Final answer:" / mask logits to force the end-thinking token) rather than truncating mid-thought.
  - The model has no internal budget counter; the engine maps effort → prompt + budget, counts, and enforces the graceful cutoff.
- **best-of-N is OUT of scope** for reasoning-effort — it's a serving / test-time-compute feature, independent of training (optional serving feature later, not part of this).
- **Bound:** at 150M, "think harder" reliably changes thinking *length* but gives only a modest *quality* gain — a real, visible knob, not a frontier reasoning switch. The value is learning the pipeline, not raw capability.

## 17. Expectations (honest)

The end result is a **pair of basic small agents** (50M and 150M): simple single/few-step tool calls + the most elementary logic after RL. Neither reliably does multi-step reasoning or complex tool sequencing — that needs far more parameters; the 50M is the weakest, the 150M stronger. For a **learning** project, the value is the **scaling comparison** (50M vs 150M at every stage) and understanding the full modern pipeline end-to-end — not raw capability.

## 18. Next Steps

- Planning is essentially complete — all stage decisions are locked (§6–§16). **No implementation yet.**
- Next: break the plan into Backlog tasks/milestones (following the §6 incremental build order).
