# Visualizations

## Purpose of Visualization Tracking

Visualizations in this repository serve to:

- Validate understanding (e.g., gradient flow, attention patterns)
- Document experiments (loss curves, metrics, ablations)
- Communicate results for portfolio and technical discussions
- Debug model behavior (e.g., attention heatmaps revealing failure modes)

## Types of Visualizations Expected

| Type | Phase | Purpose |
|------|-------|---------|
| Loss curves | 1, 2, 3 | Track convergence, compare optimizers or setups |
| Decision boundaries | 1 | Verify model learned correct classification regions |
| Gradient check plots | 1 | Confirm analytical gradients match numerical |
| Optimizer comparison | 1 | Overlaid loss curves for SGD, momentum, Adam |
| Attention heatmaps | 2, 3 | Show what tokens attend to; debug causality |
| Perplexity curves | 2, 3 | Track language model quality over training |
| Sample outputs | 2, 3, 4 | Qualitative assessment of generation |
| Pipeline diagrams | 3, 4 | RAG flow, agent loop, multi-agent architecture |
| Evaluation tables | 3, 4 | Metrics across experiments and ablations |

## Why Visualization Is Critical

**Gradients:** Plotting gradient magnitudes across layers or over training reveals vanishing/exploding gradients, dead ReLUs, and optimization instabilities. Numerical gradients alone are not enough; visualization connects numbers to intuition.

**Attention:** Attention weight heatmaps show whether the model attends to the right tokens, whether causal masking works, and whether heads specialize. Patterns often reveal bugs (e.g., attending to padding) or surprising behavior (e.g., "copy" heads).

**Experiments:** Loss and metric curves are the primary record of whether a setup works. Ablations without visualization are harder to interpret and harder to present.

Store files in this folder with descriptive names (e.g., `m5_attention_heatmap.png`, `m3_optimizer_comparison.png`) and reference them from `experiments_log.md` and milestone READMEs.
