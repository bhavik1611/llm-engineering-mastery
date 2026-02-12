# M7: Mini GPT

## Problem Statement

Build a small autoregressive language model (GPT-style) by stacking transformer blocks. Train it on a text corpus (e.g., character-level or small token-level). Generate sequences by sampling from the model's output distribution. Document loss curves, perplexity, and sample quality.

## Mathematical Formulation

### Token Embedding
(Placeholder for embedding lookup)

### Positional Encoding
(Placeholder for sinusoidal or learned positions)

### Autoregressive Loss
(Placeholder for cross-entropy over next-token prediction)

### Sampling
(Placeholder for temperature, top-k, top-p if used)

## Intuition

(Placeholder for intuition on next-token prediction, teacher forcing, and sampling strategies)

## Implementation Plan

1. Define vocabulary and embedding layer
2. Add positional encoding (sinusoidal or learned)
3. Stack N transformer blocks (e.g., 2-4 for "mini")
4. Add output projection to vocabulary size
5. Implement training: forward pass, cross-entropy loss, backward, update
6. Implement generation: autoregressive sampling with temperature
7. Train on a small corpus (e.g., Shakespeare, WikiText)
8. Evaluate perplexity and inspect samples

(No code.)

## Required Experiments

- Train for sufficient steps; plot loss and perplexity
- Generate samples at temperature 0.5, 1.0, 1.5; compare coherence vs diversity
- (Optional) Compare character-level vs subword tokenization

## Required Visualizations

- Loss curve over training
- Perplexity over training (or validation)
- Sample outputs at different temperatures (table or side-by-side)

## Reflection Questions

1. Why does teacher forcing work during training despite exposure bias?
2. How does temperature affect the sampling distribution?
3. What are the tradeoffs between a smaller context window and a larger one?

## Completion Checklist

- [ ] Mini GPT trains and converges
- [ ] Loss and perplexity curves in `visualizations/`
- [ ] Temperature ablation documented in `experiments_log.md`
- [ ] Generated samples documented
- [ ] Journal entry in `learning_journal.md`
