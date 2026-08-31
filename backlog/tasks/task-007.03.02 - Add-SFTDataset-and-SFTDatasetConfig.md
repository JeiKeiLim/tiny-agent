---
id: TASK-007.03.02
title: Add SFTDataset and SFTDatasetConfig
status: Done
assignee:
  - '@Jongkuk Lim'
created_date: '2026-08-31 01:20'
updated_date: '2026-08-31 01:43'
labels:
  - sft
  - implementation
milestone: m-2
dependencies:
  - TASK-007.03.01
parent_task_id: TASK-007.03
priority: high
ordinal: 42000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement example-based SFT dataset loading for M2.

Depends on:
- TASK-007.03.01

Files:
- src/kestrel/data/sft_dataset.py
- tests/data/test_sft_dataset.py

Scope:
- Add SFTDatasetConfig(BaseConfig) with strict fields:
  - input: unified SFT JSONL path
  - tokenizer_path
  - context_length: default 1024
  - batch_size
  - seed
  - max_examples: optional int
  - preserve_source_ratios: bool, default true
- Add SFTDataset that:
  - reads unified SFT JSONL rows
  - validates rows with the SFT schema
  - renders rows with the SFT chat template
  - tokenizes with the Kestrel tokenizer
  - truncates or filters rows longer than context_length
  - builds input/target/loss_mask batches
  - applies max_examples using deterministic seeded sampling
  - preserves source ratios as closely as practical when max_examples is set
- Do not reuse the pretrain packed-token iterator directly.

Acceptance:
- Unit tests use tiny local JSONL fixtures, not network data.
- max_examples=1 on a multi-source fixture samples proportionally where possible.
- Loss mask trains only assistant tokens.
- Batches have shape (batch_size, context_length).
- make check passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SFTDatasetConfig is strict Pydantic and supports max_examples
- [x] #2 SFTDataset reads unified JSONL and produces input/target/loss_mask batches
- [x] #3 max_examples preserves source ratios as closely as practical
- [x] #4 Tests cover truncation/filtering, batching, and source-ratio subset behavior
- [x] #5 make check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add SFTDatasetConfig(BaseConfig) in src/kestrel/data/sft_dataset.py with strict fields: input, tokenizer_path, context_length=1024, batch_size, seed, max_examples optional, preserve_source_ratios=true.
2. Add SFTDataset that reads unified SFT JSONL, validates SFTRow, applies deterministic seeded max_examples with source-ratio preservation, renders rows with render_sft, filters/truncates rows by context_length, and builds padded input/target/loss_mask batches.
3. Add tests/data/test_sft_dataset.py using tiny local JSONL fixtures, covering strict config, batching shape, loss masking, truncation/filtering, and proportional max_examples behavior.
4. Run make check and fix all failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented SFTDatasetConfig and SFTDataset in src/kestrel/data/sft_dataset.py.
Dataset reads unified SFT JSONL, validates SFTRow, renders with render_sft, filters rows longer than context_length, pads short rows with token id 0, and yields (input, target, loss_mask) int32 batches of shape (batch_size, context_length).
Target is the input sequence shifted left by one, matching the pretrain dataset convention. The final real token is loss-masked because its shifted target is padding or a repeated boundary token.
max_examples uses deterministic largest-remainder allocation across observed source tags when preserve_source_ratios=true, then seeded per-source and global shuffles.
Added tests/data/test_sft_dataset.py using a tiny local BPE tokenizer and local JSONL fixtures.
Validation: make check passed with 201 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added example-based SFT dataset loading with strict config, context filtering, padding, loss-masked batches, and deterministic source-ratio-preserving max_examples sampling. Verified with tests and make check.
<!-- SECTION:FINAL_SUMMARY:END -->
