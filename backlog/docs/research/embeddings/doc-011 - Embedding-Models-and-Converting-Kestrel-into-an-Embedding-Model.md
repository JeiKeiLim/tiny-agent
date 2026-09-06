---
id: doc-011
title: Embedding Models and Converting Kestrel into an Embedding Model
type: other
created_date: '2026-09-06 04:34'
updated_date: '2026-09-06 04:34'
---
# Embedding Models and Converting Kestrel into an Embedding Model

## Summary

An **embedding model** is not a different kind of embedding layer. It is usually a full transformer model whose output is a single dense vector representing a whole text.

The token embedding layer inside a transformer is only the first lookup step:

```text
token id -> learned vector
```

A text embedding model does:

```text
text
  -> token embedding lookup
  -> transformer layers
  -> pooling over token hidden states
  -> optional projection
  -> normalization
  -> final text embedding
```

The important difference between a normal next-token model such as Kestrel and an embedding model is the **training objective**:

```text
Kestrel pretraining:
  predict the next token

embedding model:
  make similar texts close and dissimilar texts far
```

If a transformer is fine-tuned with a similarity objective, pooled hidden states, and usually a projection head, it becomes an embedding model.

## Token Embedding vs Text Embedding Model

A token embedding layer is a learned lookup table:

```text
vocab_size x hidden_size
```

Mathematically, it is equivalent to a single linear layer with a one-hot input and no bias or activation:

```text
y = Wx
```

but because `x` is one-hot, the operation reduces to selecting one row:

```text
y = W[token_id]
```

In Kestrel, this is:

```python
self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
```

at `src/kestrel/model/kestrel.py:241`.

A text embedding model is a different concept:

```text
"the cat sat on the mat"
  -> [0.013, -0.204, 0.881, ...]
```

The word “embedding” in “embedding model” refers to the final text representation, not the first token lookup table.

## What Modern Embedding Models Do

Most modern text embedding models are transformer-based. They differ from normal language models mainly in output head and training data/objective.

Common architecture:

```text
tokenizer
  -> transformer encoder or decoder
  -> pooling
  -> dense projection
  -> L2 normalization
```

Common pooling strategies:

```text
CLS token:
  use the special classification token hidden state

mean pooling:
  average hidden states over valid tokens

last-token pooling:
  use the final valid token hidden state
```

For encoder-only BERT-style models, `CLS` and mean pooling are common. For decoder-only LLMs, last-token or mean pooling is common.

## Research Findings

### Sentence-BERT

Sentence-BERT showed that raw BERT hidden states are not automatically good sentence embeddings for cosine similarity.

Key findings:

- Directly using BERT `CLS` or mean hidden states gives weak cosine-similarity performance.
- Fine-tuning BERT with a siamese or triplet structure produces strong sentence embeddings.
- Pooling strategies include `CLS`, mean, and max.
- Mean pooling was the default in the paper.
- Training objectives include classification, regression, and triplet loss.
- The final embeddings are compared with cosine similarity.

This supports the core idea:

```text
transformer + pooling + similarity fine-tuning = embedding model
```

Source:

- Reimers and Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks”, https://arxiv.org/pdf/1908.10084

### E5

E5 is a general-purpose text embedding model trained with weakly supervised contrastive learning.

Key findings:

- E5 uses large-scale curated text pairs called CCPairs.
- It is trained contrastively so that related text pairs become closer.
- It works for retrieval, clustering, and classification.
- It performs strongly both zero-shot and after task-specific fine-tuning.
- General-purpose embedding models can be trained on diverse weakly supervised pairs rather than only manually labeled similarity data.

Source:

- Wang et al., “Text Embeddings by Weakly-Supervised Contrastive Pre-training”, https://arxiv.org/abs/2212.03533

### GTE

GTE is a general text embedding model trained with multi-stage contrastive learning.

Key findings:

- GTE trains on a diverse mixture of datasets.
- It uses multi-stage contrastive learning across unsupervised and supervised data.
- A relatively small GTE base model, around 110M parameters, outperformed much larger embedding models and the then OpenAI embedding API on MTEB.
- Data diversity and staged contrastive training matter a lot for general embedding quality.

This is relevant to Kestrel because Kestrel’s 50M and 150M models are in the same small-model regime where data quality and training objective can matter more than raw scale.

Source:

- Li et al., “Towards General Text Embeddings with Multi-stage Contrastive Learning”, https://arxiv.org/abs/2308.03281

### LLM2Vec

LLM2Vec shows that decoder-only LLMs can be converted into strong text encoders.

Key findings:

- Decoder-only LLMs are not naturally ideal embedding models because they use causal attention and next-token training.
- LLM2Vec converts a decoder-only LLM into a text encoder using:
  1. bidirectional attention,
  2. masked next token prediction,
  3. unsupervised contrastive learning.
- The transformed models perform strongly on MTEB.
- Adding supervised contrastive learning improves performance further.

This is the closest research match to converting Kestrel, since Kestrel is also a decoder-only transformer.

Source:

- BehnamGhader et al., “LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders”, https://arxiv.org/abs/2404.05961

### Matryoshka Representation Learning

Matryoshka Representation Learning, or MRL, trains embeddings so that prefixes of the embedding vector are also useful.

Key findings:

- A single model can produce embeddings of multiple effective dimensions.
- The first `d` dimensions form a smaller embedding that is still useful.
- This allows trade-offs between embedding size, storage, speed, and quality.
- MRL can be added with minimal changes to standard representation-learning pipelines.

For Kestrel, MRL is optional and probably not needed for a first version, but it is a useful later enhancement.

Source:

- Kusupati et al., “Matryoshka Representation Learning”, https://arxiv.org/abs/2205.13147

### Sentence Transformers Documentation

The Sentence Transformers documentation describes the standard modular architecture used by many embedding models:

```text
transformer
  -> pooling
  -> dense projection
  -> optional normalization
```

It also documents many training losses, including:

```text
MultipleNegativesRankingLoss
ContrastiveLoss
CoSENTLoss
triplet losses
distillation losses
```

This confirms that the common production pattern is:

```text
base transformer + pooling + projection + contrastive or similarity loss
```

Source:

- Sentence Transformers documentation, https://www.sbert.net/

## Key Design Choices

### Encoder-only vs Decoder-only

Encoder-only models:

```text
bidirectional attention
often strong for static text embedding
common in BERT-style embedding models
```

Decoder-only models:

```text
causal attention
naturally trained for next-token prediction
can be converted to embedding models
may need bidirectional adaptation or careful pooling
```

Kestrel is decoder-only. The simplest first version can remain causal. A stronger version could experiment with LLM2Vec-style bidirectional adaptation.

### Pooling

For Kestrel, the likely first choices are:

```text
last valid token:
  simple, natural for decoder-only models

mean over valid tokens:
  often robust, requires padding/document masking
```

Mean pooling is probably the safest first experiment if masking is implemented correctly. Last-token pooling is the simplest baseline.

### Projection Head

A small linear projection is common:

```python
projection = nn.Linear(hidden_size, embedding_dim, bias=False)
```

For the first version:

```text
embedding_dim == hidden_size
```

is the simplest choice.

### Normalization

Final embeddings are usually L2-normalized:

```python
z = z / mx.linalg.norm(z, axis=-1, keepdims=True)
```

After normalization, cosine similarity becomes dot product:

```python
similarity = z_a @ z_b.T
```

### Loss

A minimal first loss is in-batch contrastive learning:

```text
for each anchor:
  one positive
  other batch items act as negatives
```

With L2-normalized embeddings and temperature `tau`:

```text
sim = z_anchor @ z_all.T / tau
loss = cross_entropy(sim, labels=arange(batch_size))
```

This is similar in spirit to `MultipleNegativesRankingLoss` in Sentence Transformers.

Later improvements:

```text
hard negatives
triplet loss
CoSENT
distillation from a stronger model
Matryoshka loss
```

## How to Convert Kestrel

Current Kestrel forward pass:

```python
h = self.embed(x)
for layer in self.layers:
    h = layer(h, cos, sin, doc_ids, cache)
h = self.final_norm(h)
return mx.matmul(h, self.embed.weight.T)
```

The final matrix multiply produces next-token logits using the tied embedding matrix.

For an embedding model, stop before the LM head and pool the hidden states:

```python
h = self.embed(x)
for layer in self.layers:
    h = layer(h, cos, sin, doc_ids, None)

h = self.final_norm(h)

pooled = pool(h, mask)
z = self.projection(pooled)
z = z / mx.linalg.norm(z, axis=-1, keepdims=True)
return z
```

The LM head is not used for embedding inference.

## Recommended Kestrel Implementation Plan

### 1. Expose Hidden States

Refactor `Kestrel._forward` or add a method such as:

```python
def forward_hidden(self, x, doc_ids=None) -> mx.array:
    ...
    return self.final_norm(h)
```

This returns:

```text
(B, T, hidden_size)
```

instead of:

```text
(B, T, vocab_size)
```

### 2. Add an Embedding Wrapper

Add a wrapper such as:

```python
class KestrelEmbedding(nn.Module):
    def __init__(self, model: Kestrel, embedding_dim: int | None = None):
        super().__init__()
        self.model = model
        dim = embedding_dim or model.config.hidden_size
        self.projection = nn.Linear(model.config.hidden_size, dim, bias=False)

    def encode(self, x, mask) -> mx.array:
        h = self.model.forward_hidden(x)
        pooled = masked_mean(h, mask)
        z = self.projection(pooled)
        return z / mx.linalg.norm(z, axis=-1, keepdims=True)
```

### 3. Choose Pooling

First version:

```text
masked mean pooling
```

Fallback baseline:

```text
last valid token pooling
```

Mean pooling requires a mask so padding or document separators do not contaminate the average.

### 4. Add Training Data

Training examples should be pairs or triplets:

```text
anchor, positive
anchor, positive, negative
```

Good data sources:

```text
paraphrase pairs
NLI entailment pairs
question-answer pairs
search query-document pairs
semantic textual similarity pairs
domain-specific document pairs
```

For Kestrel, start small and clean before scaling up.

### 5. Add Contrastive Trainer

Add:

```text
src/kestrel/data/embedding_dataset.py
src/kestrel/train/embedding.py
scripts/run_embedding_finetune.py
configs/kestrel/50m/embedding.yaml
```

The trainer should:

```text
load a pretrained Kestrel checkpoint
initialize the projection head
encode anchor and positive batches
compute in-batch contrastive loss
update projection first, then optionally fine-tune the backbone
```

### 6. Evaluation

Use simple held-out similarity or retrieval evaluation:

```text
given an anchor,
rank the positive above negatives
```

Metrics:

```text
recall@1
recall@5
recall@10
mean reciprocal rank
cosine similarity on STS-style pairs
```

Do not use next-token perplexity as the main embedding metric.

## Recommended First Experiment

The smallest useful Kestrel embedding experiment would be:

```text
backbone:
  pretrained Kestrel 50M or 150M

pooling:
  masked mean pooling

projection:
  linear hidden_size -> hidden_size
  bias = false

normalization:
  L2

loss:
  in-batch InfoNCE / MultipleNegativesRanking-style loss

training:
  freeze backbone first
  train projection head
  then optionally unfreeze backbone with small learning rate

evaluation:
  held-out anchor-positive-negative retrieval
```

This avoids changing Kestrel’s attention or pretraining contract and gives a clean baseline.

## Later Improvements

Possible later upgrades:

```text
LLM2Vec-style bidirectional attention
masked next-token objective before contrastive training
hard negative mining
Matryoshka embedding dimensions
distillation from a larger embedding model
task-specific prompt templates
domain-specific contrastive data
```

## Main Conclusion

There is no separate magic architecture for embedding models.

The practical formula is:

```text
embedding model =
  transformer backbone
  + pooling
  + projection
  + normalization
  + similarity/contrastive training
```

For Kestrel, the conversion is therefore straightforward:

```text
remove the LM head
pool transformer hidden states
add a projection head
train with contrastive text-pair data
```

The main research-backed caveat is that raw next-token hidden states are not automatically good embedding vectors. The similarity training step is the part that makes the model an embedding model.
