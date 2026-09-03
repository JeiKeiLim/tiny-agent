---
id: doc-010
title: Small-Model Agentic Tool-Calling and Pretrain Recipe Research
type: guide
created_date: '2026-09-03 22:53'
updated_date: '2026-09-03 22:54'
tags:
  - research
  - small-models
  - agentic
  - tool-calling
  - pretraining
  - sft
  - rl
  - json
  - constrained-decoding
  - kestrel
---
# Small-Model Agentic Tool-Calling and Pretrain Recipe Research

Research note summarizing how other small language models are trained for agentic behavior, tool calling, and reliable structured output, and what Kestrel-50M can learn from those recipes.

## Status

- Date: 2026-09-02
- Scope: small/open models roughly 50M–3B parameters with tool-use, function-calling, or agentic post-training evidence.
- Related Kestrel context:
  - `TASK-009 - Retrain 50M from scratch on the full 3.27B-token corpus`
  - `TASK-007.03.10 - Run 50M SFT data-scaling validation`
  - `doc-005 - 50M Pretrain Token Budget and 3B Continuation Research`
  - `doc-006 - M2 SFT Data Strategy Decision`
  - `doc-007 - M2 SFT Tool-Calling Format Decision`
  - `doc-008 - SFT Drift, Pretrain Perplexity, and Tool-Call Failure Research`

## Question

- How do other small models acquire agentic/tool-calling behavior?
- What pretraining recipes do they use?
- Does JSON/code/structured-data exposure in pretraining matter for tool calling?
- What is the most practical recipe for Kestrel-50M, given that it currently emits the `tool_call` marker but often produces malformed JSON payloads?

## Executive summary

Small agentic models usually do not rely on a single trick. The stronger public recipes combine:

1. an overtrained small base model,
2. code/JSON-rich pretraining or mid-training,
3. tool-specific SFT on full trajectories,
4. a simple and reliable call format,
5. often inference-time format enforcement,
6. optionally DPO/RL/rejection sampling later.

For Kestrel-50M, the most important takeaways are:

- Chinchilla-style token budgets are not the main constraint for tool calling.
- Code/JSON pretraining exposure is a strong predictor of structured-output reliability.
- SFT alone may not be enough at 50M if the base has weak JSON structure.
- Full trajectories are better than call-only examples.
- A staged tool curriculum is likely better than one flat 50k SFT mixture.
- Constrained decoding is the highest-leverage immediate fix for malformed JSON.
- A JSON/API continued-pretrain or mid-train ablation is the cleanest way to test the pretrain-data hypothesis.

## Comparable small agentic/tool-calling models

| Model | Size | Recipe highlights | Tool-use result / notes |
|---|---:|---|---|
| FunctionGemma | 270M | Gemma-3-270M base; function-calling fine-tune; compact escaped call format instead of raw JSON | Very relevant sub-1B evidence; BFCL Simple ~61.6 zero-shot; format simplicity matters at small scale |
| Granite 4.0 nano | 350M–1B | IBM small models with native structured-output/JSON focus | Evidence that sub-1B structured tool output is viable with format-native training |
| Hammer 2.1 | 0.5B–7B | Fine-tunes coder bases such as Qwen2.5-Coder; adds function masking at inference | Strong evidence that code/JSON pretraining exposure helps function calling |
| Qwen3 0.6B / 1.7B | 0.6B / 1.7B | 36T pretrain; reasoning RL; agent RL with environment feedback; strong-to-weak distillation | BFCL v3 FC: 0.6B ~55.8 thinking, 1.7B ~58.7; small Qwen3 models beat many larger non-reasoning models |
| Llama 3.2 1B / 3B | 1.2B / 3.2B | Pruning + logit distillation from Llama 3.1 8B/70B; up to 9T tokens; SFT + rejection sampling + DPO | BFCL v2: 1B ~25.7, 3B ~67.0; 3B is the more useful tool-calling size |
| xLAM-1B-fc-r | ~1.3B | DeepSeek-Coder base; ~60K synthetic tool trajectories; SFT + DPO | Strong small-model tool recipe; xLAM family was a top BFCL performer at release |
| TinyAgent-1.1B | 1.1B | TinyLlama base + LoRA on ~80K GPT-4-Turbo synthetic tool trajectories | Shows small models can learn tool use from synthetic trajectories, often with retrieval/scaffolding |
| SmolLM3 3B | 3B | 11T pretrain; mid-training; SFT including tool calling; APO/DPO-style alignment | SmolLM line adds tool calling at 3B, not at 135M/360M |
| Trellis-506M | 0.5B | Community pretrain mix deliberately includes JSON infoboxes, text-to-SQL, structured data | Closest “pretrain for function calling” analog: ~12.5% structured/JSON/SQL slice |

Caveats:

- These models are not directly comparable because of tokenizer, data quality, distillation, synthetic data, benchmark version, and inference scaffolding.
- Several use much larger token budgets than Kestrel currently has.
- Some rely on teacher models, execution environments, or constrained decoding that Kestrel does not yet have.
- The goal is not to copy a recipe blindly, but to extract the parts that are testable at Kestrel scale.

## Pretraining recipe patterns

### Small models are usually overtrained

Public sub-1B models are commonly trained far beyond Chinchilla-style token budgets.

| Model | Params | Pretrain tokens |
|---|---:|---:|
| SmolLM 135M | 135M | 600B |
| MobileLLM 125M | 125M | ~1T |
| Gemma 3 270M | 270M | ~6T |
| OLMo 1B | 1B | ~3T |
| Llama 3.2 1B | 1.2B | up to 9T |
| Qwen3 0.6B | 0.6B | 36T |
| Kestrel-50M current full corpus | 50.7M | 3.27B |

Interpretation for Kestrel:

- 3.27B tokens is still below the public sub-1B band.
- However, Kestrel is a learning pipeline, not a frontier-small-model competition.
- The 3.27B run is the right first full-corpus epoch.
- If validation and downstream metrics improve, 2–3 epochs or a larger high-quality corpus is evidence-backed.

### Common data mix

Most small-model pretraining recipes include:

- filtered educational web
- code
- math
- synthetic textbook-like data
- sometimes multilingual data
- sometimes explicit structured data: JSON, API schemas, SQL, config files

The structured-data component is the part most relevant to Kestrel’s tool-calling problem.

Many public recipes do not publish a dedicated “tool-call pretraining” slice, but several signals point the same direction:

- Trellis-506M deliberately over-weights JSON/SQL/structured data for function calling.
- Qwen2.5 adds tool-control tokens and emphasizes structured output.
- Hammer 2.1 starts from coder bases because code/JSON exposure helps function calling.
- MidTool shows that tool/API/code mid-training can improve agent/tool benchmarks before SFT/RL.

### Multi-stage training is common

A modern small-model recipe often has multiple stages:

```text
broad pretrain
→ mid-training on long context / reasoning / tool/API data
→ high-quality annealing
→ SFT / instruction tuning
→ optional DPO / RL / rejection sampling
```

Examples:

- OLMo 0724 uses a final high-quality anneal.
- Qwen2.5 uses staged pretraining mixture transitions.
- Llama 3.1/3.2 uses long-context extension and distillation.
- SmolLM3 uses pretrain → mid-training → SFT → alignment.
- MidTool uses base pretrain → tool-use mid-training → SFT/RL.

For Kestrel, the analogous stage would be:

```text
base pretrain
→ JSON/API/code/tool-format mid-train or continued pretrain
→ SFT assistant/math/tool
→ optional DPO/RL later
```

### Distillation is common at 1B+, uncertain at 50M

Llama 3.2 and Gemma 2 use teacher-logit distillation heavily. MobileLLM found distillation from LLaMA-2-7B was not clearly useful at 125M/350M.

For Kestrel-50M:

- distillation from Kestrel-150M or a small open teacher is plausible,
- but it should be treated as an experiment,
- not a default assumption.

### Architecture patterns

Modern small models commonly use:

- RoPE
- SwiGLU
- RMSNorm
- GQA
- tied embeddings
- byte-level BPE
- deeper/thinner shapes at very small scale

Kestrel already matches much of this stack. The current architecture is not the main suspected bottleneck for tool JSON; data and format are more actionable.

## Agentic post-training patterns

### Full trajectories beat call-only examples

Stronger recipes train on full agent trajectories:

```text
user
→ assistant tool_call
→ tool result
→ assistant final answer
```

Call-only examples teach syntax, but not:

- when to call a tool,
- when not to call,
- how to use the result,
- how to recover from missing information,
- how to stop.

Kestrel’s local tool slice already includes tool-result/final-answer behavior, which is the right direction.

### Synthetic teacher trajectories are standard

Common pattern:

1. Use a strong teacher model to generate tool-use trajectories.
2. Execute or validate them.
3. Reject invalid trajectories:
   - invalid JSON
   - hallucinated tool name
   - wrong arguments
   - empty loops
   - failed execution
4. Fine-tune the small model on the survivors.

Examples:

- ToolLLM / ToolBench
- ToolACE
- xLAM
- TinyAgent
- AgentSynth / SWE-smith-style execution-validated synthesis

For Kestrel, execution-validated synthetic trajectories are likely more valuable than blindly increasing total SFT row count.

### Curricula matter

A good small-model tool curriculum looks like:

1. **Format warmup**
   - one tool
   - one call
   - strict simple payload
   - short examples
2. **Single-tool with results**
   - call → mock result → final answer
3. **Boundary behavior**
   - no tool needed
   - missing information
   - clarification
   - distractor tools
4. **Multi-turn / multi-tool**
   - only after basic format is stable

Kestrel’s current 50k mixture already has some of this, but the observed failure mode suggests the early format-warmup stage may be too weak.

### DPO/RL is usually secondary at 50M

DPO/RL can help, but most small-model tool results come first from:

- better base pretraining,
- code/JSON exposure,
- clean SFT trajectories,
- a simple format,
- constrained decoding.

RL with verifiable rewards is more promising later, especially for:

- schema-valid reward,
- correct-tool reward,
- execution-success reward,
- no-call correctness reward.

At 50M, RL should be treated as an later experiment, not the first fix.

## Tool-call format findings

### Raw nested JSON is hard for very small models

OpenAI-style nested `tool_calls` JSON is syntactically heavy:

```json
{"tool_calls":[{"function":{"name":"...","arguments":"..."}}]}
```

Small models often fail on:

- missing `{`,
- extra `}`,
- quotes,
- commas,
- nested arrays,
- stringified JSON arguments,
- termination tokens.

Kestrel’s current failure matches this pattern: the model emits `tool_call` and plausible content, but not a clean JSON object.

### Simpler formats are more reliable at small scale

FunctionGemma uses a compact escaped format rather than raw JSON:

```text
<start_function_call>call:tool_name{arg:<escape>value<escape>}<end_function_call>
```

This is a strong signal that for sub-1B models, format simplicity can matter more than model scale.

Ranked format options for Kestrel:

1. **Constrained decoding / grammar masking**
   - makes JSON valid by construction
   - biggest immediate reliability win
2. **Compact line-based call format**
   - FunctionGemma-style or `name(k=v, k2=v2)`
   - fewer punctuation tokens
3. **Flat JSON**
   - current Kestrel format, but only one object, no nesting
4. **XML-style tags**
   - easier than JSON for some small models
5. **OpenAI-style nested JSON**
   - hardest at 50M scale

## JSON-pretraining hypothesis

The research supports the hypothesis that removing explicit JSON/structured data from pretraining is a likely contributor to Kestrel’s weak tool JSON.

Evidence:

- code-pretrained bases are much better at zero-shot JSON parsability,
- coder bases are preferred starting points for function calling,
- tool/API mid-training improves agent/tool benchmarks,
- small models still need SFT in the exact target format,
- at 50M, SFT alone may not be enough if the base has weak JSON structure.

Current Kestrel interpretation:

```text
not enough JSON/API structure in pretrain
+ exact JSON boundary is hard for 50M
+ SFT format underfitting
+ possible narrow-SFT drift
```

This is not yet proven as the single cause, but it is now a well-supported hypothesis.

## Recommended Kestrel experiments

### Immediate / cheap

1. **Add constrained decoding for tool calls**
   - implement a small JSON/grammar FSM in the MLX decode loop
   - mask tokens that cannot legally appear next
   - this should eliminate malformed JSON as a failure class
   - measure whether `valid_json_rate` goes to ~1.0 while tracking tool selection and argument accuracy separately

2. **Simplify the tool payload if possible**
   - keep flat JSON for the current M2 contract
   - consider a compact line-based format for a 50M-specific track
   - avoid nested OpenAI-style structures at this scale

3. **Run a staged SFT curriculum**
   - 10k strict format warmup
   - 20k single-tool + result + final answer
   - 10k no-call / missing-info / distractors
   - 10k harder multi-turn or unseen-schema examples

### Medium-term

4. **Add JSON/API mid-training**
   - add 5–15% of pretrain/mid-train tokens from:
     - JSON documents
     - API/OpenAPI schemas
     - config files
     - text-to-SQL
     - tool-call-style synthetic rows
   - or run a short continued-pretrain experiment from the 3B checkpoint

5. **Use teacher-generated, execution-validated trajectories**
   - generate diverse tool dialogues
   - execute/validate them
   - reject invalid calls
   - include negative “do not call” examples

### Later

6. **RL with verifiable rewards**
   - schema-valid reward
   - correct-tool reward
   - execution-success reward
   - no-call correctness reward

This should be considered after constrained decoding and better SFT data.

## Proposed comparison matrix

Once the 3B pretrain run finishes, the cleanest experiment matrix is:

| Base | SFT variant | Decoding | Expected signal |
|---|---|---|---|
| 1B pretrain | current 50k | unconstrained | baseline |
| 3B pretrain | current 50k | unconstrained | effect of more pretrain tokens |
| 3B pretrain | staged tool curriculum | unconstrained | effect of better SFT curriculum |
| 3B pretrain | current 50k | constrained JSON | effect of inference-time format enforcement |
| 3B + JSON mid-train | current 50k | unconstrained | effect of JSON/API pretrain exposure |
| 3B + JSON mid-train | staged tool curriculum | constrained JSON | combined recipe |

Primary metrics:

- pretrain validation loss / perplexity
- SFT validation loss
- tool `valid_json_rate`
- tool `schema_valid_rate`
- tool selection accuracy
- argument accuracy
- no-call correctness
- missing-info correctness
- GSM8K exact match
- assistant sanity / repetition
- pretrain drift after SFT

Decision rule:

- If constrained decoding fixes JSON validity, the remaining bottleneck is tool selection/argument quality.
- If JSON mid-training improves unconstrained validity, the pretrain-data hypothesis is supported.
- If staged SFT improves validity without constrained decoding, the SFT curriculum was underfit.
- If none of these help, capacity/format simplicity becomes the main suspect.

## References

- FunctionGemma: https://huggingface.co/google/functiongemma-270m-it
- FunctionGemma docs: https://ai.google.dev/gemma/docs/functiongemma
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Qwen2.5 technical report: https://arxiv.org/abs/2412.15115
- Llama 3.1 paper: https://arxiv.org/abs/2407.21783
- Llama 3.2 model card: https://huggingface.co/meta-llama/Llama-3.2-3B
- xLAM: https://arxiv.org/abs/2409.03215
- TinyAgent: https://arxiv.org/abs/2409.00608
- SmolLM blog: https://huggingface.co/blog/smollm
- SmolLM2: https://arxiv.org/abs/2502.02737
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- MobileLLM: https://arxiv.org/abs/2402.14905
- OLMo: https://arxiv.org/abs/2402.00838
- Gemma 2: https://arxiv.org/abs/2408.00118
- Gemma 3: https://arxiv.org/abs/2503.19786
- Phi-3: https://arxiv.org/abs/2404.14219
- Pythia: https://arxiv.org/abs/2304.01373
- Trellis pretraining: https://huggingface.co/mdonigian/trellis-pretraining
- Hammer 2.1: https://arxiv.org/abs/2410.04587
- Small Models, Big Tasks / small-model function calling study: https://arxiv.org/abs/2504.19277
- MidTool: https://arxiv.org/abs/2608.20314
- ToolLLM / ToolBench: https://arxiv.org/abs/2307.16789
- ToolACE: https://arxiv.org/abs/2409.00920
- BFCL leaderboard: https://gorilla.cs.berkeley.edu/leaderboard
- BFCL paper: https://proceedings.mlr.press/v267/patil25a.html
- τ-bench: https://arxiv.org/abs/2406.12045
- XGrammar: https://arxiv.org/abs/2411.15100
- Outlines / guided generation: https://arxiv.org/abs/2307.09702
- Grammar-Aligned Decoding: https://arxiv.org/abs/2405.21047
- FireFunction v2: https://fireworks.ai/blog/firefunction-v2-launch-post
