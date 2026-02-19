# M2: Neural Network from Scratch

## Problem Statement

Implement a multi-layer neural network with at least one hidden layer from first principles. Use only NumPy (or equivalent) for matrix operations. Train using backpropagation to compute gradients and gradient descent to update parameters. Verify gradients numerically.

## Mathematical Formulation

### Forward Pass

$$
z_1 = X W_1 + b_1, \quad a_1 = \sigma(z_1)
$$
$$
z_2 = a_1 W_2 + b_2, \quad \hat{y} = \sigma(z_2)
$$

Loss: BCE \( L = -\frac{1}{n}\sum_i [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)] \)

### Backpropagation (Chain Rule)

From output toward input:
1. \(\frac{\partial L}{\partial z_2} = \hat{y} - y\) (BCE + sigmoid cancellation)
2. \(\frac{\partial L}{\partial W_2} = \frac{1}{n} a_1^T \frac{\partial L}{\partial z_2}\)
3. \(\frac{\partial L}{\partial b_2} = \frac{1}{n}\sum \frac{\partial L}{\partial z_2}\)
4. \(\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} W_2^T\)
5. \(\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1)\), with \(\sigma'(z) = \sigma(z)(1-\sigma(z))\)
6. \(\frac{\partial L}{\partial W_1} = \frac{1}{n} X^T \frac{\partial L}{\partial z_1}\)
7. \(\frac{\partial L}{\partial b_1} = \frac{1}{n}\sum \frac{\partial L}{\partial z_1}\)

### Gradient Expressions (Summary)

| Param | Gradient |
|-------|----------|
| \(W_2\) | \(a_1^T (\hat{y}-y) / n\) |
| \(b_2\) | \(\operatorname{mean}(\hat{y}-y)\) |
| \(W_1\) | \(X^T \bigl[(\hat{y}-y) W_2^T \odot \sigma'(z_1)\bigr] / n\) |
| \(b_1\) | \(\operatorname{mean}\bigl[(\hat{y}-y) W_2^T \odot \sigma'(z_1)\bigr]\) |

## Intuition

Backprop distributes the output error \(\hat{y}-y\) backward through the network. Each layer multiplies by its local derivative (sigmoid or linear). Gradients flow from output to input; saturated sigmoids shrink the signal (vanishing gradient).

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

- [x] Backprop derivation documented
- [x] Numerical gradient check passes
- [x] Multi-layer network trains on non-linear task
- [x] Depth comparison experiment in `experiments_log.md`
- [x] Visualizations in `visualizations/`
- [x] Journal entry in `learning_journal.md`
