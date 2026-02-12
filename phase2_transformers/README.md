# Phase 2: Transformers

## What This Phase Is About

Phase 2 focuses on the core architecture underlying modern LLMs: the transformer. You will implement scaled dot-product attention from scratch, assemble a full transformer block, and build a small autoregressive GPT model. The goal is to understand attention mechanisms and causal masking at a foundational level.

## Mathematical Focus

- Scaled dot-product attention: Q, K, V matrices and score computation
- Softmax and temperature scaling
- Causal masking for autoregressive generation
- Layer normalization and residual connections
- Positional encoding (sinusoidal or learned)

## Engineering Focus

- Implementing attention with correct tensor shapes and masking
- Building composable transformer blocks
- Training an autoregressive language model
- Debugging generation (sampling, temperature, repetition)

## Expected Competencies After Completion

- Explain the role of Q, K, V and derive the attention formula
- Implement causal masking for decoder-only transformers
- Assemble a transformer block and understand the flow of information
- Train a small GPT and interpret loss curves and generated samples

## Milestones in This Phase

| Milestone | Description |
|-----------|-------------|
| M5 | Attention from scratch |
| M6 | Transformer Block |
| M7 | Mini GPT |

## Exit Criteria

You have completed Phase 2 when:

- [ ] M5: Attention implemented; scores, weights, and outputs validated
- [ ] M6: Full transformer block with causal masking; correct for autoregression
- [ ] M7: Mini GPT trained; generates coherent sequences; loss and samples documented
