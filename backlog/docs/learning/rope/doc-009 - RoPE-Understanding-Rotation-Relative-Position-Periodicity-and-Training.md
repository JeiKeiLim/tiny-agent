---
id: doc-009
title: 'RoPE Understanding: Rotation, Relative Position, Periodicity, and Training'
type: guide
created_date: '2026-09-01 23:54'
updated_date: '2026-09-01 23:56'
tags:
  - learning
  - rope
  - attention
  - transformer
  - kestrel
---
# RoPE Understanding: Rotation, Relative Position, Periodicity, and Training

Personal learning notes from a discussion about how Rotary Position Embeddings (RoPE) work in Kestrel.

The goal is to build an intuitive but accurate mental model of:

- why `q · k` is the attention score
- how RoPE injects position into attention
- why rotating both `q` and `k` creates relative-position behavior
- what happens when rotation reaches 360°
- why real multi-frequency RoPE does not simply reset
- how training interacts with RoPE
- what RoPE does and does not teach the model

## Interactive visualization

A self-contained HTML visualization is stored at:

```text
backlog/docs/learning/rope/rope-visualization.html
```

Open it with:

```bash
open backlog/docs/learning/rope/rope-visualization.html
```

The visualization covers:

- general q/k rotation
- single-frequency 360° periodicity
- multi-frequency RoPE behavior
- attention weights over a generic sequence
- training forward/backward flow
- causal placement examples

## Short answer

RoPE rotates each token’s query and key vectors by an amount determined by that token’s position.

For a query at position `m` and a key at position `n`, the attention score becomes dependent on the relative distance:

```text
m - n
```

It does not add a separate learned “distance weight” after the fact. Instead, it changes the geometry of `q` and `k` before the dot product, so the dot product itself becomes position-aware.

## Attention without position

In decoder-only transformer attention, each token produces:

```text
q = query vector
k = key vector
v = value vector
```

For one query token and one key token:

```text
score = q · k
```

That score is one scalar.

For one query token against many allowed key tokens:

```text
scores = [q·k0, q·k1, q·k2, ...]
```

That produces a score vector.

For all query positions against all allowed key positions, the model conceptually produces an attention score matrix.

The scores are passed through softmax to become attention weights:

```text
weights = softmax(scores)
```

The output for the query is then a weighted mix of values:

```text
output = weight0 * v0 + weight1 * v1 + weight2 * v2 + ...
```

Important roles:

```text
q = “what am I looking for?”
k = “what can be matched?”
v = “what content gets copied if matched?”
```

RoPE rotates `q` and `k`, but not `v`.

## What RoPE does

RoPE applies position-dependent rotations to `q` and `k` after projection.

In Kestrel, this happens in:

```text
src/kestrel/model/kestrel.py
```

The attention layer projects `x` into `q`, `k`, and `v`, then applies rotary embeddings to `q` and `k`:

```python
q, k = apply_rotary_emb(q, k, cos, sin)
```

The rotation tables are computed from positions:

```python
freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))
angles = positions[..., None] * freqs
cos = mx.cos(angles)
sin = mx.sin(angles)
```

Then each vector is rotated using the position-specific `cos` and `sin` values:

```python
x1, x2 = x[..., : d // 2], x[..., d // 2 :]
return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
```

This is a 2D rotation applied across many dimension pairs.

## Why rotation creates relative position

In a simplified 2D case, suppose a query at position `m` and a key at position `n` are rotated by:

```text
q_m = rotate(q, m * f)
k_n = rotate(k, n * f)
```

where `f` is a rotation frequency.

Their dot product becomes:

```text
q_m · k_n = q · R((m - n) * f) · k
```

The absolute positions `m` and `n` mostly cancel. What remains is the relative offset:

```text
m - n
```

That is the core idea.

Without RoPE:

```text
same content at different distances -> same q·k score
```

With RoPE:

```text
same content at different distances -> different q·k scores
```

The model can then learn to use distance-aware attention patterns.

## Kestrel specifics

For Kestrel-50M:

```text
hidden_size = 512
n_heads = 8
head_dim = 512 / 8 = 64
rope_theta = 10000
context_length = 2048
```

Each attention head has:

```text
head_dim / 2 = 32 rotation frequencies
```

So Kestrel does not use one single rotation angle. It uses many simultaneous rotations across the head dimension.

The frequencies are computed as:

```python
freqs = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2) / head_dim))
```

With `head_dim = 64` and `rope_theta = 10000`, the first few frequencies are approximately:

```text
1.0
0.1
0.01
0.001
...
```

The fastest frequency changes quickly over short distances. The slowest frequencies change gradually over long distances.

This gives the model a multi-scale position signal.

Kestrel also supports document-aware training. When `doc_ids` are provided, positions reset at document boundaries, so RoPE encodes position within each document rather than one global position across packed documents.

## The 360° question

If RoPE used only one frequency, then yes: the signal is periodic.

A single frequency repeats whenever its rotation angle changes by:

```text
360° = 2π radians
```

If the frequency is `f` radians/token, the period is:

```text
period = 2π / f
```

Example:

```text
f = 1.0 rad/token
period ≈ 6.28 tokens
```

That means a single-frequency score can repeat at distances separated by about `6.28` tokens.

However, real RoPE uses many frequencies.

For the full RoPE pattern to repeat at a nonzero distance `d`, every frequency would need to realign at the same time:

```text
d * f1 = 2π * integer
d * f2 = 2π * integer
d * f3 = 2π * integer
...
```

Because the frequencies are generally not simple integer multiples of one another, this does not normally happen within practical context lengths.

So:

```text
single frequency: 360° can repeat the signal
full multi-frequency RoPE: no simple 360° reset
```

One frequency may complete a full cycle while the others are at different phases. The combined high-dimensional rotation pattern is therefore much richer and less likely to alias.

## Training and backprop

RoPE is differentiable.

Forward:

```text
token embeddings
  -> q_proj / k_proj / v_proj
  -> RoPE rotation
  -> q·k scores
  -> softmax + value mixing
  -> loss
```

Backward:

```text
loss
  -> attention gradients
  -> gradients wrt rotated q/k
  -> through RoPE rotation math
  -> gradients wrt q_proj / k_proj / v_proj
  -> token embeddings and rest of model
```

The rotation frequencies themselves are fixed.

Training updates:

```text
q_proj
k_proj
v_proj
o_proj
feed-forward weights
embeddings
final norm
```

Training does not update:

```text
rope_theta
frequency table
cos/sin position tables
```

Mental model:

```text
RoPE is a fixed positional lens.
Training learns q/k vectors that work well through that lens.
```

## What RoPE teaches vs what training teaches

RoPE does not directly teach the model which words go together.

The training objective teaches statistical patterns from data:

```text
which tokens often follow other tokens
which tokens are semantically related
which syntactic dependencies matter
which dependencies are local vs long-range
```

RoPE’s role is to make attention position-aware.

Without position information, attention would mostly rely on content matching. With RoPE, the model can also learn patterns like:

```text
this token usually depends on the token 1 step before
this pattern matters at short distance
that pattern can persist over longer distance
word order is important
```

Clean separation:

```text
training data + loss:
teaches statistical language patterns

q/k content projections:
learn which meanings/tokens match

RoPE:
makes those matches position-aware
```

## Causal placement example

Consider:

```text
A: random * 1000 + Apple Juice
B: Apple Juice + random * 1000
```

These are different because causal attention only allows a token to attend to previous tokens plus itself.

In A:

```text
Apple can attend to the previous 1000 random tokens.
Juice can attend to the previous 1000 random tokens plus Apple.
```

In B:

```text
Apple has no previous context.
Juice can attend only to Apple.
Later random tokens can attend back to Apple and Juice.
```

So the same words can have different attention contexts depending on where they are placed.

RoPE affects the distance relationships, but the causal mask determines which tokens are visible in the first place.

## Common misconceptions

### “RoPE adds a distance weight after q·k”

Not exactly.

It is not usually:

```text
final_score = q·k + distance_weight
```

Instead:

```text
final_score = rotated_q · rotated_k
```

The distance dependence is built into the rotated dot product.

### “Closer tokens always get higher attention”

No.

RoPE makes scores distance-dependent, but the model learns how to use that signal. In some contexts, farther tokens may be more relevant.

### “RoPE inserts an absolute position label”

Not directly.

RoPE is based on position indices, but the useful property is that attention scores depend on relative position.

### “360° means the whole model forgets position”

No.

A single frequency may repeat after 360°, but full multi-frequency RoPE generally does not reset as a whole.

### “v is rotated too”

No.

Kestrel rotates `q` and `k`, but not `v`.

## Key takeaways

```text
1. One q and one k produce one scalar attention score.

2. RoPE rotates q and k by position-dependent angles.

3. The rotated dot product depends on relative distance.

4. Kestrel uses 32 rotation frequencies per 50M attention head.

5. A single frequency is periodic and can repeat after 360°.

6. Full multi-frequency RoPE does not simply reset at one 360° event.

7. Training flows through RoPE, but the RoPE frequencies are fixed.

8. RoPE does not teach word associations by itself.
   It gives attention a position-aware geometry that training can learn to use.
```
