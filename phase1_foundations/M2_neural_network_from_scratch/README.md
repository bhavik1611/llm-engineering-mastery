# M2: Neural Network from Scratch

## Problem Statement

Implement a multi-layer neural network with at least one hidden layer from first principles. Use only NumPy (or equivalent) for matrix operations. Train using backpropagation to compute gradients and gradient descent to update parameters. Verify gradients numerically.

## Mathematical Formulation

### Forward Pass
(Placeholder for layer-wise computation and activation functions)

### Backpropagation
(Placeholder for chain rule application through layers)

### Gradient Expressions
(Placeholder for derivatives of loss w.r.t. weights and activations)

## Intuition

(Placeholder for intuition on how backprop distributes error through layers)

## Implementation Plan

1. Implement linear layer (matrix multiply + bias)
2. Implement ReLU (or similar) activation
3. Implement forward pass through multiple layers
4. Implement cross-entropy (or MSE) loss
5. Derive backprop equations for each layer
6. Implement backward pass
7. Implement gradient descent update
8. Add numerical gradient check
9. Train on a non-linear classification dataset (e.g., XOR or 2D blobs)

(No code.)

## Required Experiments

- Train on a dataset that requires non-linear decision boundaries
- Run numerical gradient check; document that analytical and numerical gradients match within tolerance
- Compare 1-layer vs 2-layer network on the same task

## Required Visualizations

- Loss curve over training
- Decision boundary for 2D input
- Gradient check: plot analytical vs numerical gradients (or error magnitude) across parameters

## Reflection Questions

1. Why is the vanishing gradient problem more severe in deeper networks?
2. How does ReLU address gradient flow compared to sigmoid?
3. What is the role of initialization in avoiding dead neurons?

## Completion Checklist

- [ ] Backprop derivation documented
- [ ] Numerical gradient check passes
- [ ] Multi-layer network trains on non-linear task
- [ ] Depth comparison experiment in `experiments_log.md`
- [ ] Visualizations in `visualizations/`
- [ ] Journal entry in `learning_journal.md`
