---
id: TASK-007.03.02
title: Add SFTDataset and SFTDatasetConfig
status: To Do
assignee: []
created_date: '2026-08-31 01:20'
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
- [ ] #1 SFTDatasetConfig is strict Pydantic and supports max_examples
- [ ] #2 SFTDataset reads unified JSONL and produces input/target/loss_mask batches
- [ ] #3 max_examples preserves source ratios as closely as practical
- [ ] #4 Tests cover truncation/filtering, batching, and source-ratio subset behavior
- [ ] #5 make check passes
<!-- AC:END -->
