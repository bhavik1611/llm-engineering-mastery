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

**Date:** 2026-02-13 (updated 2026-03-10)
**Milestone:** M1_logistic_regression
**Objective:** Baseline logistic regression on linearly separable 2D data, with full evaluation suite.

### Setup

- Model: Logistic regression (BCE, sigmoid), gradient descent
- Data: Synthetic 2D blobs (`generate_linearly_separable`), 80/20 train/test split
- Optimization: lr=0.5, epochs=200

### Metrics

| Metric | Value |
|--------|-------|
| Train Accuracy | 100.0% |
| Test Accuracy | 100.0% |
| Final BCE Loss (train) | ~0.0000 |
| AUC (ROC) | 1.0 |

### Observations

- Linearly separable data: model converges quickly with lr=0.5
- LR ablation: lr=0.01 converges slowly; lr=0.5 optimal; lr=2.0 may oscillate but can still converge on this simple task

### Visualization

| File | Purpose |
|------|---------|
| `visualizations/m1_loss_and_grad_norm.png` | Loss and gradient norm vs epoch |
| `visualizations/m1_linear_model_boundary.png` | 2D decision boundary and data |
| `visualizations/m1_decision_boundary_evolution.png` | Boundary evolution during training |
| `visualizations/m1_learning_rate_ablation.png` | Loss curves for lr=0.01, 0.5, 2.0 |
| `visualizations/m1_confusion_matrix.png` | TN, FP, FN, TP on test set |
| `visualizations/m1_roc_curve.png` | ROC curve with AUC |

### Findings by Visualization

- **m1_loss_and_grad_norm**: Loss drops to near zero within ~50–100 epochs; gradient norm collapses as model converges. When confident and correct, gradients shrink—validating the "small gradients at convergence" intuition from the math.
- **m1_linear_model_boundary**: Single linear hyperplane cleanly separates the two blob classes. Points near the boundary are correctly classified; no margin violations for linearly separable data.
- **m1_decision_boundary_evolution**: The boundary rotates and translates over epochs, moving from a poor initial guess to the final separating line. Shows how gradient descent refines parameters step by step.
- **m1_learning_rate_ablation**: lr=0.01 decays slowly; lr=0.5 converges fast and smoothly; lr=2.0 may show small oscillations but still reaches the minimum on this convex problem.
- **m1_confusion_matrix**: On linearly separable data, expect zeros off the diagonal (perfect classification). Any non-zero off-diagonal counts indicate misclassifications.
- **m1_roc_curve**: AUC=1.0 for perfect separation. Curve hugs the top-left corner; threshold sweep shows that many cutoffs achieve 100% TPR and 0% FPR.

### Notebook

- Source: `phase1_foundations/M1_logistic_regression/logistic_regression_full_flow.ipynb`

### Conclusion

- PASS: M1 complete. All required visualizations generated; ROC/AUC computed.

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

### Visualization

| File | Purpose |
|------|---------|
| m2_gradient_check.png | Analytical vs numerical gradients per parameter |

### Findings by Visualization

- **m2_gradient_check**: Bar chart per parameter (W1, b1, W2, b2). Bars should be near zero—max relative error &lt; 1e−5 confirms backprop matches finite-difference gradients. Critical sanity check before training.

### Notebook

- Source: `phase1_foundations/M2_neural_network_from_scratch/M2_experiments_and_visualizations.ipynb`

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
| m2_vanishing_gradient_depth.png | Gradient magnitude vs network depth (reflection) |
| m2_relu_vs_sigmoid.png | ReLU vs sigmoid gradient flow (reflection) |
| m2_dead_neurons_init.png | Dead neuron heatmap with bad init (reflection) |

### Findings by Visualization

- **m2_xor_data**: Four clusters at corners; class 0 at (0,0) and (1,1), class 1 at (0,1) and (1,0). No single line can separate them—visual proof of linear inseparability.
- **m2_loss_and_decision_boundary**: Left: loss drops to &lt; 0.1 by ~2000 epochs. Right: 2-layer boundary is curved, dividing the four regions; decision surface wraps around the data.
- **m2_depth_comparison_loss**: 1-layer loss plateaus near 0.69 (random); 2-layer converges. Left panel shows 1-layer's futile linear cut; right shows 2-layer's curved boundary solving XOR.
- **m2_2layer_decision_boundary**: Non-linear boundary with smooth transitions; correctly separates all four XOR corners. Hidden layer learned effective features (e.g. AND, OR).
- **m2_vanishing_gradient_depth**: Gradient norm per layer decreases exponentially with depth for sigmoid; product of σ′(z) ≤ 0.25 across layers. Explains why deep sigmoid nets are hard to train.
- **m2_relu_vs_sigmoid**: ReLU derivatives stay at 1 for z &gt; 0; sigmoid derivatives saturate. Bar chart or curve shows sigmoid gradient collapse vs ReLU's sustained flow.
- **m2_dead_neurons_init**: Heatmap (neurons × epochs): bad init yields dark horizontal bands (dead ReLUs, z &lt; 0 always); He init keeps neurons active (lighter, evolving patterns).

### Notebook

- Source: `phase1_foundations/M2_neural_network_from_scratch/M2_experiments_and_visualizations.ipynb`

### Conclusion

- 2-layer NN solves XOR; 1-layer cannot. Depth enables non-linear decision boundaries.

---

## Experiment: M2 Initialization & Deep Dive

**Date:** 2026-02-20
**Milestone:** M2_neural_network_from_scratch
**Objective:** Explore initialization strategies, activation depth, loss landscapes, and gradient dynamics using MultiLayerNN on two-moons.

### Setup

- Model: MultiLayerNN with configurable layer specs, per-layer activations
- Data: Two-moons dataset
- Initialization: Xavier, He, Small, Large compared
- Sections: Init effects, activation depth (sigmoid vs ReLU), dead neurons, loss landscape, explosion/vanishing

### Metrics

| Init | Convergence | Notes |
|------|-------------|-------|
| Xavier | ✓ Clean | Suitable for sigmoid/tanh |
| He | ✓ Clean | Preferred for ReLU |
| Small | ✗ Collapse | Tiny activations, slow/no learning |
| Large | ✗ Overshoot/divergence | Fragile on small nets |

### Visualization

| Section | Output | Purpose |
|---------|--------|---------|
| 1. Initialization | Loss curve animation | Xavier/He vs Small/Large convergence |
| 2. Activation depth | Gradient norms animation | Sigmoid vanishing vs ReLU flow at 2/3/5 layers |
| 3. Dead neurons | Activation heatmap animation | Bad init → dark rows (dead ReLUs) |
| 4. Loss landscape | Trajectory animation | 2D weight slice, non-convexity, basins |
| 5. Explosion/vanishing | Gradient norms animation | Deep sigmoid (Xavier vs Large) |

### Findings by Visualization

- **Section 1 (Initialization)**: Xavier and He curves descend smoothly to low loss; Small init flattens (almost no learning); Large may overshoot or diverge initially. Log-scale loss makes differences stark.
- **Section 2 (Activation depth)**: Bar chart of gradient norm per layer over time. Sigmoid: early layers get tiny gradients as depth increases; ReLU: norms stay more uniform across layers.
- **Section 3 (Dead neurons)**: Heatmap rows = neurons, cols = epochs. Bad init: many dark rows from the start. He/Xavier: lighter, dynamic activity; fewer permanently dead units.
- **Section 4 (Loss landscape)**: 2D contour slice of loss; trajectory animates from init to converged weights. Non-convex: multiple basins, saddle-like regions; trajectory path depends on init.
- **Section 5 (Explosion/vanishing)**: Xavier on deep sigmoid → vanishing (early layers near zero). Large init → exploding gradients (early layers spike). Contrast clarifies why careful init matters.

### Notebook

- Source: `phase1_foundations/M2_neural_network_from_scratch/M2_initialization_experiments.ipynb`

### Conclusion

- Xavier and He converge cleanly; Small init causes collapse; Large can diverge. Sigmoid gradients vanish with depth; ReLU preserves flow. Bad initialization produces dead ReLU neurons.

---

## Experiment: M3 Optimization Dynamics Laboratory

**Date:** 2025-03-08
**Milestone:** M3_optimization_dynamics
**Objective:** Compare SGD, Momentum, RMSProp, and Adam on 2D loss landscapes and neural network training; document convergence behavior and ablation.

### Setup
- 2D loss: Elliptical $L = a(w_1-1)^2 + b(w_2-2)^2$ (ill-conditioned when $a \gg b$)
- NN: MultiLayerNN on two-moons (M2 modules)
- Optimizers: SGD (η=0.05), Momentum (β=0.9), RMSProp (ρ=0.9), Adam (β₁=0.9, β₂=0.999)

### Metrics
| Optimizer | 2D convergence | NN loss curve | Notes |
|-----------|---------------|---------------|-------|
| SGD | Zig-zag in valley | Oscillatory | Single global lr |
| Momentum | Smoother path | Between SGD and Adam | Accumulates velocity |
| RMSProp | Faster along shallow dir | — | Per-parameter scaling |
| Adam | Fastest typically | Smoothest, fastest | Momentum + adaptive |

### Visualization
| File | Purpose |
|------|---------|
| m3_optimization_landscape.png | Section 1: Loss contours + gradient descent trajectory |
| m3_ill_conditioned_zigzag.png | Section 3: Zig-zag on elliptical valley |
| m3_sgd_vs_momentum.png | Section 4: SGD vs Momentum trajectories |
| m3_optimizer_comparison.png | Section 5: SGD, Momentum, RMSProp, Adam paths |
| m3_saddle_point.png | Section 6: Loss along saddle trajectory |
| m3_nn_optimizer_comparison.png | Section 7: NN loss curves (SGD vs Momentum vs Adam) |

### Findings by Visualization

- **m3_optimization_landscape**: 3D surface + contour view. Trajectory (red line) crosses contours roughly perpendicularly—gradient points steepest ascent, steps go opposite. Convex quadratic: straight path to minimum.
- **m3_ill_conditioned_zigzag**: Elongated elliptical contours; trajectory bounces between valley walls instead of going straight down. High condition number (a ≫ b) causes slow progress in the shallow direction.
- **m3_sgd_vs_momentum**: Same valley, two paths. SGD (red): zig-zag. Momentum (cyan): smoother, more direct descent. Velocity smooths oscillations; fewer steps to reach the minimum.
- **m3_optimizer_comparison**: Left: four optimizer trajectories on contours. Right: loss vs step (log scale). Adam and RMSProp typically reach the minimum fastest; SGD lags; Momentum sits between.
- **m3_saddle_point**: Loss vs step for a saddle ($L = w₁² − w₂²$). Flat region near the origin—gradients small, progress slow. Illustrates why saddles (common in high-dim NNs) can stall training.
- **m3_nn_optimizer_comparison**: BCE loss vs epoch on two-moons. Adam (magenta) converges fastest and smoothest; SGD (red) more oscillatory; Momentum (cyan) intermediate. Same 2D principles scaled to NN training.

### Notebook
- Source: `phase1_foundations/M3_optimization_dynamics/Optimization_Dynamics_Laboratory.ipynb`

### Conclusion
- Adam and Momentum outperform vanilla SGD on ill-conditioned and NN tasks. Same principles (zig-zag, momentum smoothing, per-parameter scaling) apply in 2D and high-dimensional training.

---

## Experiment: M4 PyTorch Autograd Trace

**Date:** 2026-03-10
**Milestone:** M4_pytorch_internal_mechanics
**Objective:** Trace the computational graph for a 2-layer MLP, verify manual gradients match autograd, and document gradient flow.

### Setup
- Model: 2-layer MLP (2→3→1), manual forward pass (no nn.Module)
- Data: Fixed 4×2 input, 4×1 target
- Loss: BCE
- Lab: M4_autograd_trace.ipynb (Coursera-style fill-in)

### Metrics
| Check | Result |
|-------|--------|
| Tensor shapes printed | z1(4,3), a1(4,3), z2(4,1), y_hat(4,1) |
| Gradient shapes | W1.grad(2,3), b1.grad(3,), W2.grad(3,1), b2.grad(1,) |
| Manual ∂L/∂W2 vs W2.grad | Match (torch.allclose) |
| retain_graph experiment | Second backward fails without retain_graph=True |

### Visualization
| File / Location | Purpose |
|-----------------|---------|
| `visualizations/m4_computational_graph.png` | Autograd graph (torchviz) for 2-layer MLP |
| Graph trace (notebook Section 3) | Computational graph: X→z1→a1→z2→y_hat→loss |
| Shape table (notebook Section 4) | Tensor and gradient shapes at each step |

### Notebook
- Source: `phase1_foundations/M4_pytorch_internal_mechanics/M4_autograd_trace.ipynb`

### Conclusion
- PASS: Computational graph traced; manual and autograd gradients match; gradient flow and graph lifecycle (retain_graph) understood. Journal entry documents reflection answers.
