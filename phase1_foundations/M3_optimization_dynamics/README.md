# M3 — Optimization Dynamics Laboratory

This module contains the **Optimization Dynamics Laboratory** notebook, part of Phase 1 (Foundations) of the LLM Engineering Mastery curriculum.

## Purpose

Build deep intuition about how optimization works in machine learning and deep learning through:

1. **Visual exploration** — Loss surfaces, contours, trajectories
2. **Comparisons** — Learning rates, SGD vs Momentum vs Adam
3. **Connection to practice** — Neural network training using M2's `MultiLayerNN`

## Contents

- `Optimization_Dynamics_Laboratory.ipynb` — Interactive lab with 7 conceptual sections

## Dependencies

- NumPy, Matplotlib (Sections 1–6)
- scikit-learn (make_moons for Section 7)
- M2 modules: `multi_layer_nn`, `layer`, `activations` (Section 7)

## Running the Notebook

Ensure you are in the `M3_optimization_dynamics` directory when running the notebook (or that the working directory allows the M2 path to be found via `os.path.dirname(os.getcwd())`).

## Completion Checklist

- [x] Optimizer comparison (SGD, Momentum, Adam) on 2D loss and NN training
- [x] Ablation documented in `experiments_log.md`
- [x] Visualizations in `visualizations/`
- [x] Journal entry in `learning_journal.md`
