# M1: Logistic Regression from Scratch

## Problem Statement

Implement logistic regression for binary classification from first principles. Train it using gradient descent without using a deep learning framework's autograd. The goal is to internalize the relationship between the loss function, its derivatives, and the parameter update rule.

## Mathematical Formulation

### Loss Function

The loss function for logistic regression is the binary cross-entropy (BCE), derived from the likelihood of the data under the Bernoulli model:

Given:

- Input features: $x \in \mathbb{R}^d$
- Weights: $w \in \mathbb{R}^d$
- Bias: $b \in \mathbb{R}$
- Target label: $y \in \{0, 1\}$
- Model prediction: $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$, where $z = w^T x + b$

Probability model: $P(y|x) = \hat{y}^y (1 - \hat{y})^{1-y}$

Cross-entropy loss for a single sample:
L(y, 𝑦̂ ) = - [ y · log(𝑦̂ ) + (1 − y) · log(1 − 𝑦̂ ) ]

### Gradient Derivation

Let $z = w^T x + b$ and $\hat{y} = \sigma(z)$. For a single sample $(x, y)$, the gradients are:

- Gradient w.r.t. weights:
  $$
  \frac{\partial L}{\partial w} = ( \hat{y} - y ) x
  $$
- Gradient w.r.t. bias:
  $$
  \frac{\partial L}{\partial b} = \hat{y} - y
  $$

For a batch, average the gradients across all samples.

### Update Rule

The parameters are updated via gradient descent:

- Update for weights:
  $$
  w := w - \eta \cdot \frac{1}{N} \sum_{i=1}^N ( \hat{y}^{(i)} - y^{(i)} ) x^{(i)}
  $$
- Update for bias:
  $$
  b := b - \eta \cdot \frac{1}{N} \sum_{i=1}^N ( \hat{y}^{(i)} - y^{(i)} )
  $$

where $\eta$ is the learning rate, $N$ is the number of samples in the batch.

## Intuition

Logistic regression models the probability that a sample belongs to the positive class. The sigmoid function $\sigma(z)$ maps the linear combination of features to a value between 0 and 1, interpreted as the estimated probability of the positive class. The loss function penalizes confident wrong predictions heavily, encouraging well-calibrated probabilities. Geometrically, in 2D, logistic regression finds a linear decision boundary that best separates the two classes in terms of log-odds.

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

   **Answer:** Cross-entropy is aligned with maximizing the likelihood of the correct class, penalizing incorrect confident predictions more than MSE. It provides improved gradients for learning when predictions are far from true labels.

2. What happens to the gradient magnitude when the model is confident and correct?

   **Answer:** When the model is confident and correct (i.e., $\hat{y}$ is near $y$), the gradient magnitude becomes very small, resulting in tiny parameter updates and stabilizing training.

3. How does the learning rate affect the path of convergence in parameter space?

   **Answer:** A small learning rate leads to slow convergence, while a large learning rate can cause overshooting or divergence. An appropriate learning rate enables efficient and stable convergence along the gradient direction.

## Completion Checklist

- [x] Gradient derivation documented
- [x] Model trains and converges on binary classification
- [x] Loss curve visualization in `visualizations/`
- [x] Learning rate ablation documented in `experiments_log.md`
- [x] Journal entry in `learning_journal.md`
