# Experiments Log

## Purpose

This document records all experiments conducted during the LLM engineering mastery journey. It enables:

- Reproducibility of results
- Comparison across ablations and configurations
- Tracking of metrics and visualizations over time
- Clear documentation for portfolio and technical discussions

## Experiment Logging Template

Use this structure for each experiment:

```markdown
## Experiment: [Short descriptive name]

**Date:** YYYY-MM-DD
**Milestone:** M[N]_[name]
**Objective:** [One sentence: what are you testing or validating?]

### Setup
- Model: (architecture, size, hyperparameters)
- Data: (dataset, split, preprocessing)
- Optimization: (optimizer, lr, batch size, epochs)
- Hardware: (optional, for reproducibility)

### Metrics
| Metric | Value |
|--------|-------|
| (metric name) | (value) |

### Observations
- (Qualitative notes on behavior, failure modes, surprises)

### Visualization
- Path: `visualizations/[filename]`
- Description: (What the plot shows and why it matters)

### Conclusion
- (One sentence: pass/fail, key takeaway)
```

## Metrics Tracking Template

For milestones that produce quantitative results, use this table format:

| Experiment | Metric 1 | Metric 2 | Metric 3 | Notes |
|------------|----------|----------|----------|-------|
| (name) | (value) | (value) | (value) | (brief) |

Common metrics by phase:

- **Phase 1:** Loss, accuracy, convergence speed
- **Phase 2:** Loss, perplexity, sample quality
- **Phase 3:** Perplexity, BLEU/F1, retrieval accuracy
- **Phase 4:** Task success rate, latency, cost

## Visualization Tracking Template

| Experiment |    Visualization   |  File  |       Purpose        |
|------------|--------------------|--------|----------------------|
| (name)     | (e.g., loss curve) | (path) | (why it was created) |

---

## Experiments

(Experiments will be appended below as they are run.)

## Experiment: M1 Logistic Regression - Main Run

**Date:** 2026-02-13
**Milestone:** M1_logistic_regression
**Objective:** Baseline logistic regression on linearly separable 2D data.

### Setup

- Model: Logistic regression (BCE, sigmoid), gradient descent
- Data: Synthetic 2D blobs, 80/20 train/test split
- Optimization: lr=0.5, epochs=200

### Metrics

| Metric | Value |
|--------|-------|
| Train Accuracy | 100.0% |
| Test Accuracy | 100.0% |
| Final BCE Loss (train) | 0.0000 |

### Visualization

- Path: `visualizations/m1_*.png`
- Plots: loss curve, decision boundary, learning rate ablation, confusion matrix

### Conclusion

- Baseline run complete. Train and test accuracy reported.

---

## Experiment: M2 Two-Layer NN — Gradient Check

**Date:** 2026-02-15
**Milestone:** M2_neural_network_from_scratch
**Objective:** Verify analytical gradients match numerical (finite-difference) gradients.

### Setup
- Model: TwoLayerNN (2, 3, 1), sigmoid activations, BCE loss
- Data: Random 8×2 inputs, random binary labels
- epsilon: 1e-5 for finite difference

### Metrics
| Parameter | Max Relative Error |
|-----------|--------------------|
| W1 | ~3e-8 |
| b1 | ~5e-8 |
| W2 | ~1e-10 |
| b2 | ~2e-11 |
| **Overall** | **< 1e-5** |

### Conclusion
- PASS: Analytical and numerical gradients match within tolerance.

---

## Experiment: M2 Two-Layer NN — XOR (Non-Linear)

**Date:** 2026-02-15
**Milestone:** M2_neural_network_from_scratch
**Objective:** Train 2-layer NN on XOR; compare with 1-layer (logistic regression).

### Setup
- 2-layer: input_dim=2, hidden_dim=8, lr=0.5, weight_init_std=0.5, 5000 epochs
- 1-layer: logistic regression, lr=0.5, 5000 epochs
- Data: XOR dataset (100 samples, 4 corners, class 0 at (0,0)/(1,1), class 1 at (0,1)/(1,0))

### Metrics
| Model | XOR Train Accuracy | Final Loss |
|-------|--------------------|------------|
| 1-layer (linear) | ~50% (random) | ~0.69 |
| 2-layer (non-linear) | ~100% | < 0.1 |

### Visualization
| File | Purpose |
|------|---------|
| m2_xor_data.png | XOR dataset scatter |
| m2_loss_and_decision_boundary.png | Loss curve + 2-layer decision boundary |
| m2_depth_comparison_loss.png | 1-layer vs 2-layer loss curves; 1-layer decision boundary |
| m2_2layer_decision_boundary.png | 2-layer non-linear boundary |
| m2_gradient_check.png | Analytical vs numerical gradients per parameter |

### Conclusion
- 2-layer NN solves XOR; 1-layer cannot. Depth enables non-linear decision boundaries.
