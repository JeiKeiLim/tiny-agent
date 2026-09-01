---
id: TASK-008
title: Reduce Kestrel inference memory and generation cost
status: To Do
assignee: []
created_date: '2026-09-01 00:42'
labels:
  - model
  - inference
  - memory
  - performance
dependencies: []
priority: high
ordinal: 54000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Kestrel inference currently uses a no-KV-cache generate() path that re-runs the full sequence every token. This causes O(T^2) compute and large MLX allocator-cache growth. Measured on the 50M pretrain checkpoint: 512-token greedy generation used ~0.19GiB active MLX memory and ~0.53GiB peak active memory, but retained ~18.26GiB in mx.get_cache_memory(); mx.clear_cache() released the cache. This makes scripts/check_model.py --generate and scripts/run_eval_sft.py appear to use 20-44GiB system memory. This parent tracks a short-term allocator-cache mitigation and the root-cause KV-cache generation fix.
<!-- SECTION:DESCRIPTION:END -->
