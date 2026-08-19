---
id: doc-001
title: Agentic SLM Training Pipeline — Project Plan
type: specification
created_date: '2026-08-19 22:12'
updated_date: '2026-08-19 22:35'
---
# Agentic SLM Training Pipeline — Project Plan

_Status: Draft / active discussion. No implementation started. This captures the plan and decisions from our discussion so far. Update as we converge. Tasks/milestones will be broken out only after the plan is finalized._

## 1. Vision

Build a **full, modern LLM training pipeline** — pretraining, supervised fine-tuning, and reinforcement learning — at a **small scale** that fits on a single machine. The primary goal is **learning**: understand each stage of how a modern, agentic, reasoning-capable LLM is actually trained.

Key principle: **small scale, not a degraded approach.** We use the same modern techniques used at scale (BPE tokenization, transformer pretraining, SFT on tool-calling data, R1-style RL with verifiable rewards, chain-of-thought) — just at a smaller model/data scale that is feasible on the hardware.

## 2. Hardware & Feasibility

- **Machine:** Apple M4 Pro, 48GB unified memory.
- **Memory:** Not the constraint. A 1B model fits easily (~16–20GB with AdamW optimizer states + gradients + activations). Full fine-tuning of 1B is comfortable.
- **Compute / time:** The real constraint. Pretraining a 1B model from scratch to competence needs ~10B–20B tokens → weeks to months on one M4 Pro (not practical). Pretraining 50M–150M needs far fewer tokens → hours to days (feasible).
- **Throughput (rough, to be benchmarked on the actual machine):** 1B ≈ ~1,500 tok/s; 150M ≈ ~11,000 tok/s; 50M ≈ ~30,000 tok/s.

## 3. Two-Track Structure

Two tracks, kept separate:

### Track A — Learning (from scratch)
- Train **50M and 150M** models, **same architecture, same data**, from random initialization.
- Goal: understand pretraining end-to-end and observe **scaling** (how loss/quality changes with model size).
- These models are **not** the deliverable; they are the learning vehicle.
- Includes training **our own BPE tokenizer** (the authentic "from scratch" artifact).

### Track B — Deliverable (the agent)
- Take a **pretrained** small model (Qwen family) and **fine-tune** it for agentic behavior.
- This is the model we actually run as an agent.
- Goes through SFT (tool-calling + reasoning) and RL (logic).

## 4. Pipeline (Phases)

| # | Phase | Track | What | Notes |
|---|-------|-------|------|-------|
| 0 | Pretrain | A | Train 50M, then 150M from scratch on a text corpus | Own BPE tokenizer; plot loss + scaling |
| 1 | SFT | B | Fine-tune Qwen on tool-calling + reasoning data | LoRA; Qwen chat template + tokenizer |
| 2 | RL (logic) | B | R1-style RL with verifiable math rewards (GRPO, RLOO fallback) | Slowest phase; 0.6B–1.5B |
| 3 | Serve + Agent | B | `mlx_lm.server` (OpenAI-compatible) + custom agent loop | Custom Python loop, not a framework |
| 4 | Eval | B | Tool-calling accuracy (BFCL-style) + math accuracy | Before/after RL comparison |

## 5. Decisions Made So Far

- **Framework: MLX end-to-end** (fastest on M4 Pro, one coherent framework, `mlx_lm` for SFT, custom RL loop).
- **Two-track structure** (learning from scratch + deliverable fine-tune).
- **RL is included** (not optional) — R1-style, verifiable math rewards, GRPO (RLOO fallback).
- **Full pipeline:** pretrain → SFT → RL.
- **Focus:** agentic (tool calling) + logical (reasoning / chain-of-thought).
- **Tokenizer:** our own byte-level BPE, 16k vocab (configurable), for Track A; reuse Qwen's (~151k) for Track B. See §7.
- **Tool calling:** structural JSON (Track A) + Qwen template (Track B); validity defenses = simple format + lenient parse/retry, constrained decoding as stretch. See §8.
- **Model family by size:** 50M and 150M for Track A.
- **Agent:** custom Python loop (send msgs + tool defs → parse tool call → run function → feed result back → repeat), not an agent framework.
- **Code authorship:** the agent writes all the code; the human learns by reading the code and watching the runs.

## 6. Code Organization & Modularity (design constraint)

The code **must** be modularized, easy to understand, and easy to configure/change. Concretely:

- **Config-driven:** every tunable (model size, vocab, lr, batch size, dataset paths, RL hyperparams, agent settings) lives in **YAML configs** loaded into typed dataclasses. Changing a run = editing a YAML, not code.
- **One module per concern:** tokenizer / pretrain / sft / rl / serve / agent / eval are separate packages, each independently runnable.
- **Small composable functions:** model definition is separate from the training loop; dataset is separate from model; reward is separate from the RL loop. Any piece can be swapped.
- **Track A and Track B are separate packages** so learning code doesn't tangle with deliverable code.
- **Entry points** in `scripts/` so each phase is a single command.

Proposed layout (to refine):

```
tiny-agent/
  configs/            # YAML — all tunables
    track_a/  pretrain_50m.yaml  pretrain_150m.yaml  tokenizer.yaml
    track_b/  sft.yaml  rl.yaml  serve.yaml  agent.yaml
  src/slm/
    common/           # config loading, logging, utils
    tokenizer/        # Track A: train BPE
    pretrain/         # Track A: model.py, dataset.py, train.py, evaluate.py
    sft/              # Track B: data.py, train_lora.py
    rl/               # Track B: reward.py, grpo.py, rloo.py, train.py
    serve/            # wraps mlx_lm.server
    agent/            # loop.py, client.py, tools/
    eval/             # tool_calling.py, math.py
  scripts/            # run_pretrain.sh, run_sft.sh, run_rl.sh, run_agent.sh
  data/  checkpoints/
```

## 7. Tokenizer (decided)

### Track A — our own byte-level BPE
- **Algorithm:** byte-level BPE. Text → UTF-8 bytes (256 base symbols) → repeatedly merge the most frequent adjacent pair until the vocab is full. **No OOV possible** — any text is a byte sequence.
- **Vocab size: 16k, configurable** (`configs/track_a/tokenizer.yaml`). Sized to the model: the embedding matrix is `vocab × hidden`, a large fraction of a small model (16k keeps it ~8–13%); large-model vocabs (50k–150k) would dominate a 50M model.
- **Same tokenizer for 50M and 150M** — clean scaling comparison (only model size changes).
- **Token IDs:** sequential integers = an index into the embedding matrix. Order: special tokens (PAD, EOS) → the 256 byte-tokens → merged tokens in merge-frequency order. IDs have **no Unicode meaning**.
- **Non-English / multilingual:** handled uniformly as bytes (Korean/Chinese char = 3 bytes, emoji = 4). Those bytes merge into single tokens only if frequent in the training corpus; otherwise they stay as multiple byte-tokens. Always representable (no OOV), efficient only if the language is in the corpus. Language-agnostic by construction; UTF-8 is uniquely decodable so distinct characters never collide.
- **Special tokens:** minimal — PAD, EOS.
- **JSON readiness:** include JSON/code in the pretraining corpus and/or pass the JSON structural chars `{ } [ ] " : ,` as the BPE `initial_alphabet` so they stay clean single tokens.
- **Training data + order (standard pipeline):** a sample of the pretraining corpus (same distribution). Order: curate corpus → train BPE on a sample → tokenize the full corpus → pretrain. (GPT-2/Llama train the tokenizer on their pretraining corpus; we do the same at small scale.)
- **Tool:** HuggingFace `tokenizers` (BPE trainer + ByteLevel pre-tokenizer).

### Track B — Qwen's tokenizer (reused)
- ~151k vocab, byte-level BPE. Reused to match the pretrained weights **and** the chat template (which carries the tool-calling format).

## 8. Tool Calling (decided)

**Approach: structural (JSON) tool calling.**
- **Track A:** the model emits a simple JSON tool call as text; the custom agent loop parses it (`json.loads`), runs the function, feeds the result back. No native `tools` API (we don't build one).
- **Track B:** Qwen's native tool-calling template via `mlx_lm.server`'s OpenAI-compatible `tools` param (template-based, **not** constrained-decoding-backed). Verify exact support in the serving phase.
- **SFT is where the model learns the format;** the serving endpoint is where the interface lives.

**Ensuring a small model completes a valid structure (prioritized defenses):**
1. **Simple, flat, consistent format (SFT):** one-level JSON object, few short fields, clear delimiters + a stop token that ends the call. Train on many clean examples.
2. **Lenient parsing + retry/repair:** tolerant JSON extraction (missing closing brace, trailing comma); on failure, retry with a "valid JSON only" nudge, capped at N tries. The practical safety net most small-model agents rely on.
3. **Constrained decoding (stretch goal):** mask logits so only schema-valid JSON tokens can be emitted — a hard guarantee. Not turnkey on MLX (outlines/xgrammar/llguidance are PyTorch/vLLM-centric); we'd implement a small JSON state-machine mask inside the MLX generation loop.

**Plan:** ship **1 + 2** first; add **3** as a stretch for an ironclad guarantee.

## 9. Open Questions (still to decide)

- **Base model size (Track B):** Qwen2.5-1.5B (better quality, slower RL) vs. Qwen3-0.6B (lighter, faster).
- **RL model size:** 0.6B vs. 1B vs. 1.5B.
- **Pretraining corpus:** which text corpus for Track A (e.g., SlimPajama subset, FineWeb-Edu sample, Wikipedia dump) — should include some JSON/code for tool-calling readiness. *The tokenizer sample follows from this.*
- **SFT datasets:** exact mix for tool-calling (ToolACE / xLAM / FireAct) and reasoning (GSM8K / MATH / OpenThoughts).
- **RL dataset:** which math set (GSM8K / MATH train split) + reward / answer-checking design.
- **Reasoning-effort control:** how to expose "how hard to think" at inference (prompt + thinking token budget + best-of-N).
- **Evaluation specifics:** exact tool-calling benchmark + math eval harness.
- **Repo / project structure:** finalize the layout in §6.
- **Scope of Track A:** pretraining only, or run the full pretrain→SFT→RL pipeline on the 150M too (whole pipeline at small scale)?

## 10. Reasoning / "Reasoning Effort"

- **Training:** bake chain-of-thought into SFT (reasoning traces in the data); RL rewards the correct *final answer*, which drives the model to develop its own reasoning (the R1 effect).
- **Inference:** no native "low/med/high" knob, but we approximate reasoning effort via (a) prompt ("think step by step"), (b) a thinking token budget, (c) best-of-N sampling.
- **Bound:** a ~1B model tops out around elementary math / simple logic. Not frontier reasoning.

## 11. Expectations (honest)

The end result is a **capable small agent**: solid single/few-step tool calls + elementary math/logic after RL. It will **not** reliably do hard multi-step reasoning or long-horizon agentic tasks — it's ~1B, not 400B. For a learning project, that is the right target.

## 12. Next Steps

- Continue detailed planning discussion (pretraining corpus, base model, Track A architecture, datasets, RL design, structure) — **no implementation yet**.
- Once the plan is finalized, break it into Backlog tasks/milestones.
