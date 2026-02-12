# M3: Optimizer Study

## Problem Statement

Compare the behavior of different optimization algorithms (SGD, SGD with momentum, Adam) on a neural network task. Understand how each algorithm affects convergence speed, final performance, and sensitivity to hyperparameters. Document ablations.

## Mathematical Formulation

### SGD
(Placeholder for update rule)

### Momentum
(Placeholder for velocity accumulation and update)

### Adam
(Placeholder for biased moment estimates and update rule)

## Intuition

(Placeholder for intuition on momentum, adaptive learning rates, and why Adam often converges faster)

## Implementation Plan

1. Implement SGD
2. Implement SGD with momentum
3. Implement Adam (or use a framework with known-correct implementation)
4. Fix architecture and dataset
5. Run each optimizer with tuned (or default) hyperparameters
6. Record loss curves and final metrics
7. Run learning rate ablations for each optimizer
8. Document findings

(No code.)

## Required Experiments

- Compare SGD, SGD+momentum, and Adam on the same task
- Learning rate sensitivity: for each optimizer, test at least 3 learning rates
- Document which optimizer is most robust to poor learning rate choices
- (Optional) Compare Adam vs AdamW if exploring weight decay

## Required Visualizations

- Overlaid loss curves: SGD, momentum, Adam
- Learning rate ablations: separate plot per optimizer showing multiple LR curves
- Table or plot of final loss/accuracy vs optimizer and learning rate

## Reflection Questions

1. When might SGD (with or without momentum) be preferred over Adam?
2. Why does Adam often require less learning rate tuning?
3. What is the relationship between batch size and effective learning rate?

## Completion Checklist

- [ ] All three optimizers implemented or correctly used
- [ ] Comparison experiment with loss curves in `experiments_log.md`
- [ ] Learning rate ablation for each optimizer
- [ ] Visualizations in `visualizations/`
- [ ] Journal entry in `learning_journal.md`
