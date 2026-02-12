# M4: PyTorch Internal Mechanics

## Problem Statement

Understand how PyTorch computes gradients through automatic differentiation. Trace the computational graph for a simple model, explain how `.backward()` propagates gradients, and document the relationship between operations and gradient storage.

## Mathematical Formulation

### Computational Graph
(Placeholder for graph representation of forward pass)

### Chain Rule in Autograd
(Placeholder for how gradients are accumulated through the graph)

### Gradient Storage
(Placeholder for when and where gradients are stored)

## Intuition

(Placeholder for intuition on dynamic vs static graphs, and why PyTorch builds the graph on-the-fly)

## Implementation Plan

1. Build a minimal model (e.g., linear layer + loss)
2. Enable gradient tracking and run forward pass
3. Call `.backward()` and inspect `.grad` on parameters
4. Trace which tensors require grad and how they connect
5. Document the sequence of operations and gradient flow
6. (Optional) Implement a simple custom autograd Function
7. Write a clear summary of the mechanics

(No code.)

## Required Experiments

- Trace and document the computational graph for a 2-layer MLP
- Verify that manually computed gradients match PyTorch's `.grad` for a small example
- (Optional) Compare `retain_graph=True` vs default behavior

## Required Visualizations

- Diagram of computational graph for the traced model (hand-drawn or generated)
- Table or diagram showing tensor shapes and gradient shapes at each step

## Reflection Questions

1. What is the difference between `requires_grad` and `volatile` (historical) or inference mode?
2. When does PyTorch free the computational graph?
3. How would you implement a custom layer that PyTorch does not natively support?

## Completion Checklist

- [ ] Computational graph traced and documented
- [ ] Gradient flow from loss to parameters explained
- [ ] Verification that manual and autograd gradients match
- [ ] Diagram or notes in `visualizations/` or milestone folder
- [ ] Journal entry in `learning_journal.md`
