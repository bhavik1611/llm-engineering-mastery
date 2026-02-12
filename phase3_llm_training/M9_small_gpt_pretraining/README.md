# M9: Small GPT Pretraining

## Problem Statement

Pretrain a small GPT model from scratch on a text corpus. Use your tokenizer (M8) or a standard one. Train with next-token prediction, track perplexity, and generate samples. Document training setup, hyperparameters, and results.

## Mathematical Formulation

### Pretraining Objective
(Placeholder for cross-entropy over next token)

### Perplexity
(Placeholder for exponent of average negative log-likelihood)

### Context Window
(Placeholder for truncation and batching of sequences)

## Intuition

(Placeholder for intuition on pretraining vs fine-tuning, scaling laws, and what the model learns at different stages)

## Implementation Plan

1. Prepare corpus: load text, tokenize, chunk into sequences
2. Configure GPT: layers, heads, embedding dim, context length
3. Implement data loader: random cropping, batching
4. Implement training loop: forward, loss, backward, optimizer step
5. Add checkpointing and logging
6. Compute and log perplexity (train and optionally validation)
7. Generate samples at checkpoints
8. Document final perplexity and qualitative sample quality

(No code.)

## Required Experiments

- Pretrain for a fixed number of steps or until convergence
- Report train (and validation) perplexity over time
- Generate samples at 25%, 50%, 75%, 100% of training; compare quality
- (Optional) Ablate context length or model size

## Required Visualizations

- Perplexity curve over training
- Sample generations at different checkpoints (table)
- (Optional) Loss curve and learning rate schedule

## Reflection Questions

1. What does perplexity tell you about model quality?
2. Why might validation perplexity increase while train perplexity decreases?
3. What factors limit the quality of a small GPT compared to a large one?

## Completion Checklist

- [ ] Small GPT pretrains successfully
- [ ] Perplexity and loss documented in `experiments_log.md`
- [ ] Sample generations documented
- [ ] Visualizations in `visualizations/`
- [ ] Journal entry in `learning_journal.md`
