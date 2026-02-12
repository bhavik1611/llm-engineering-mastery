# Phase 1: Foundations

## What This Phase Is About

Phase 1 establishes the mathematical and computational foundations required for deep learning and, ultimately, transformer-based language models. You will implement gradient descent, backpropagation, and neural networks from scratch, study optimizer behavior, and trace how a framework like PyTorch computes and stores gradients.

## Mathematical Focus

- Loss functions and derivatives for binary classification
- Chain rule and backpropagation through layered functions
- Gradient descent, momentum, and adaptive learning rates
- Computational graphs and automatic differentiation

## Engineering Focus

- Implementing training loops without high-level abstractions
- Numerical verification of gradients
- Comparing optimizer convergence behavior
- Understanding autograd mechanics in PyTorch

## Expected Competencies After Completion

- Derive gradients for logistic regression and multi-layer networks by hand
- Implement backpropagation from first principles
- Explain the tradeoffs between SGD, momentum, and Adam
- Trace the computational graph for a PyTorch model and explain how gradients flow

## Milestones in This Phase

| Milestone | Description |
|-----------|-------------|
| M1 | Logistic Regression from scratch |
| M2 | Neural Network from scratch |
| M3 | Optimizer Study (SGD, Adam, variants) |
| M4 | PyTorch Internal Mechanics |

## Exit Criteria

You have completed Phase 1 when:

- [ ] M1: Logistic regression trains on binary classification; gradient derivation is documented
- [ ] M2: Multi-layer neural network trains with backprop; gradients verified numerically
- [ ] M3: Optimizer comparison experiment is documented with loss curves and ablations
- [ ] M4: Autograd trace for a simple model is documented; computational graph is understood
