---
id: m-0
title: "M0: Foundation"
---

## Description

Foundation for the Kestrel pipeline: project scaffolding, the Kestrel decoder-only model (50M/150M), and a byte-level BPE tokenizer. Everything downstream (pretrain, SFT, RL, serve, eval) depends on this. Done when: both sizes instantiate with correct param counts, forward pass yields finite loss, and the tokenizer round-trips text losslessly.
