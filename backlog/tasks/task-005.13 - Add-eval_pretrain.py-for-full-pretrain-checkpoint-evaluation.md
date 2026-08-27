---
id: TASK-005.13
title: Add eval_pretrain.py for full pretrain checkpoint evaluation
status: Done
assignee: []
created_date: '2026-08-27 04:30'
updated_date: '2026-08-27 04:40'
labels:
  - eval
  - training
milestone: m-1
dependencies: []
documentation:
  - >-
    backlog/docs/research/pretrain-evaluation/doc-004 -
    Pretrain-Evaluation-Research.md
modified_files:
  - src/kestrel/eval/pretrain.py
  - scripts/eval_pretrain.py
  - tests/test_eval_pretrain.py
  - README.md
  - AGENTS.md
parent_task_id: TASK-005
priority: medium
ordinal: 33000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a read-only pretrain evaluation tool for saved Kestrel checkpoints.

Current in-loop validation loss uses only eval_iters batches, so it is a small sample of the validation split. That is fine for training monitoring and best-checkpoint selection, but it is not a final evaluation. This task adds a dedicated eval_pretrain workflow for reporting stable held-out loss, perplexity, bits/token, domain-wise loss, and optional generation samples.

Reference: doc-004 - Pretrain Evaluation Research.

Files:
- create src/kestrel/eval/pretrain.py
- create scripts/eval_pretrain.py
- create tests/test_eval_pretrain.py
- update README.md to document the new eval command
- update AGENTS.md only if a new agent-facing invariant is needed

CLI:
uv run python scripts/eval_pretrain.py --pretrain-config configs/kestrel/50m/pretrain.yaml --checkpoint checkpoints/pretrain/50m/best --max-tokens 1000000

Optional flags:
- --split val (default val; train allowed but should warn)
- --max-tokens int (default 100000; 0 or null for full split)
- --generate to print fixed samples
- --json to emit machine-readable output

Implementation:
- Load PretrainConfig from --pretrain-config.
- Load ModelConfig and CorpusConfig from paths in PretrainConfig.
- Load model weights from checkpoint dir using kestrel.model.io.load.
- Build PretrainDataset for corpus output_dir/split using trainer seq_len/batch_size and corpus seed.
- For mixed eval, iterate dataset until max_tokens or exhaustion.
- For domain eval, iterate each existing val/<domain>.jsonl file separately.
- Compute token-weighted loss: sum(loss * tokens) / sum(tokens), where tokens is batch_size * (seq_len - 1) for each full batch.
- Report mixed and per-domain loss, perplexity=exp(loss), bits/token=loss/ln(2), eval tokens.
- Optional generation uses kestrel.model.generate with fixed prompts and greedy decoding by default.

Design decisions:
- Do not reuse trainer.estimate_val_loss for final eval because it averages batch losses and is capped at eval_iters.
- Use PretrainDataset so eval uses the same tokenizer, document wrapping, doc_ids, and sequence construction as training.
- Per-domain evaluation intentionally evaluates single-domain files, not the mixed scheduler, because the purpose is domain diagnosis.
- The tool must be read-only and safe to run while another process is training, as long as the checkpoint is complete.
- Full checkpoints and weights-only checkpoints both expose weights.npz, so both should load.
- max_tokens default should prevent accidental full validation runs.

Tests:
- tests/test_eval_pretrain.py should use a tiny tokenizer and tiny corpus fixtures.
- Assert mixed eval returns finite loss, positive token count, perplexity=exp(loss), bits/token=loss/ln(2).
- Assert max_tokens stops after the requested token budget within one batch granularity.
- Assert domain report includes web/code/synthetic when those val files exist.
- Assert CLI --help works and config/checkpoint paths are validated with clear errors.
- Assert eval does not modify checkpoint directory contents.

Verification:
- make check must be green.
- Manually run the CLI against a tiny test checkpoint or a real 50M checkpoint if available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI evaluates a checkpoint and prints mixed val loss, perplexity, bits/token, and eval token count
- [x] #2 Loss is token-weighted across evaluated batches and respects --max-tokens
- [x] #3 Per-domain loss is reported for existing val/web.jsonl, val/code.jsonl, and val/synthetic.jsonl files
- [x] #4 Both weights-only and full checkpoint directories can be evaluated
- [x] #5 Optional --generate produces fixed greedy samples without affecting loss results
- [x] #6 README documents the eval_pretrain command and its purpose
- [x] #7 make check is green with tests/test_eval_pretrain.py
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add src/kestrel/eval/pretrain.py with token-weighted checkpoint evaluation, per-domain metrics, and optional greedy samples.
2. Add scripts/eval_pretrain.py CLI with --pretrain-config, --checkpoint, --split, --max-tokens, --generate, and --json.
3. Add tests/test_eval_pretrain.py covering metrics math, max_tokens, domains, weights-only/full checkpoints, CLI output/errors, and read-only behavior.
4. Update README.md with the eval_pretrain workflow.
5. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented src/kestrel/eval/pretrain.py, scripts/eval_pretrain.py, and tests/test_eval_pretrain.py. Updated README.md and AGENTS.md. make check passed with 155 tests. Manual smoke on checkpoints/pretrain/50m/best with --max-tokens 8192 reported mixed val loss 3.395943, perplexity 29.8428, and per-domain losses for web/code/synthetic; --generate produced fixed repetitive samples without changing loss.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added read-only pretrain checkpoint evaluation: src/kestrel/eval/pretrain.py, scripts/eval_pretrain.py, and tests/test_eval_pretrain.py. The CLI reports token-weighted mixed val loss, perplexity, bits/token, evaluated tokens, per-domain web/code/synthetic losses, and optional fixed greedy samples. Updated README.md and AGENTS.md. Verified with make check (155 tests passed) and a live smoke evaluation of checkpoints/pretrain/50m/best at --max-tokens 8192.
<!-- SECTION:FINAL_SUMMARY:END -->
