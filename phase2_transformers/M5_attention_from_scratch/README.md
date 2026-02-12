# M5: Attention from Scratch

## Problem Statement

Implement scaled dot-product attention from first principles. Compute query, key, and value projections, attention scores, attention weights (after softmax), and the final output. Understand the role of each component and the effect of the scaling factor.

## Mathematical Formulation

### Query, Key, Value
(Placeholder for projection equations)

### Attention Scores
(Placeholder for dot-product and scaling)

### Attention Weights
(Placeholder for softmax over scores)

### Output
(Placeholder for weighted sum of values)

## Intuition

(Placeholder for intuition: what does each position "attend to," and why does scaling matter?)

## Implementation Plan

1. Define input dimensions: sequence length, embedding dim, head dim
2. Implement Q, K, V projections (linear layers)
3. Compute attention scores: Q @ K^T, scaled by sqrt(d_k)
4. Apply optional masking (causal or padding)
5. Apply softmax to get attention weights
6. Compute output: attention_weights @ V
7. Verify tensor shapes at each step
8. Test on a small example with known dimensions

(No code.)

## Required Experiments

- Run attention on a small input; print and verify shapes of Q, K, V, scores, weights, output
- Compare attention output with and without scaling; document numerical behavior
- (Optional) Visualize attention weights for a short sequence

## Required Visualizations

- Attention weight heatmap for a sample sequence (position vs position)
- (Optional) Effect of scaling: gradient magnitude or output variance with/without scaling

## Reflection Questions

1. Why scale by sqrt(d_k)? What happens when d_k is large?
2. What is the difference between bidirectional and causal attention?
3. How does multi-head attention extend this, and why is it useful?

## Completion Checklist

- [ ] Attention formula derived and documented
- [ ] Implementation produces correct shapes
- [ ] Scaling ablation documented
- [ ] Attention heatmap in `visualizations/`
- [ ] Journal entry in `learning_journal.md`
