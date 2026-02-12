# M6: Transformer Block

## Problem Statement

Implement a full transformer block consisting of masked multi-head self-attention, residual connections, layer normalization, a feedforward MLP, and another residual + layer norm. Ensure causal masking is correct for autoregressive (decoder-only) use.

## Mathematical Formulation

### Multi-Head Attention
(Placeholder for splitting into heads, attention per head, concatenation)

### Pre-Norm vs Post-Norm
(Placeholder for layer norm placement and residual structure)

### Feedforward Network
(Placeholder for two linear layers with activation)

## Intuition

(Placeholder for intuition on residual connections, layer norm, and the flow of information through the block)

## Implementation Plan

1. Implement multi-head attention (reuse or extend M5)
2. Apply causal mask: lower triangular, -inf for future positions
3. Add residual connection and layer norm (pre-norm or post-norm)
4. Implement feedforward: Linear -> GELU/ReLU -> Linear
5. Add residual and layer norm
6. Verify gradient flow through the full block
7. Stack multiple blocks (optional for this milestone)
8. Test that causal masking prevents attending to future tokens

(No code.)

## Required Experiments

- Forward pass through block; verify output shape
- Verify that position i cannot attend to position j when j > i
- (Optional) Compare pre-norm vs post-norm stability during training

## Required Visualizations

- Diagram of block architecture (attention + FFN, residuals, norms)
- Causal mask illustration: which positions can attend to which
- (Optional) Attention heatmap showing causal structure

## Reflection Questions

1. Why use layer norm before (pre-norm) rather than after (post-norm) the sublayer?
2. What would break if the causal mask were removed?
3. How does the feedforward network differ in purpose from the attention layer?

## Completion Checklist

- [ ] Transformer block implemented with correct causal masking
- [ ] Architecture diagram in `visualizations/`
- [ ] Causal masking verified
- [ ] Journal entry in `learning_journal.md`
