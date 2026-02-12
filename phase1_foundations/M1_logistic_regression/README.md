# M1: Logistic Regression from Scratch

## Problem Statement

Implement logistic regression for binary classification from first principles. Train it using gradient descent without using a deep learning framework's autograd. The goal is to internalize the relationship between the loss function, its derivatives, and the parameter update rule.

## Mathematical Formulation

### Loss Function
(Placeholder for cross-entropy derivation)

### Gradient Derivation
(Placeholder for partial derivatives with respect to weights and bias)

### Update Rule
(Placeholder for gradient descent update equations)

## Intuition

(Placeholder for geometric and probabilistic intuition of logistic regression)

## Implementation Plan

1. Define the sigmoid function
2. Implement the forward pass (logits to probabilities)
3. Implement the cross-entropy loss
4. Derive and implement the gradient computation
5. Implement gradient descent update
6. Build a training loop
7. Evaluate on a binary classification dataset

(No code.)

## Required Experiments

- Train on a linearly separable dataset and report final accuracy
- Plot loss vs. iteration
- Experiment with learning rate: too small, appropriate, too large; document convergence behavior

## Required Visualizations

- Loss curve over training
- Decision boundary plot (if 2D feature space)
- Learning rate ablation: loss curves for at least 3 learning rates

## Reflection Questions

1. Why does the cross-entropy loss work well for classification compared to MSE?
2. What happens to the gradient magnitude when the model is confident and correct?
3. How does the learning rate affect the path of convergence in parameter space?

## Completion Checklist

- [ ] Gradient derivation documented
- [ ] Model trains and converges on binary classification
- [ ] Loss curve visualization in `visualizations/`
- [ ] Learning rate ablation documented in `experiments_log.md`
- [ ] Journal entry in `learning_journal.md`
