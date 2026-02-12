# M10: LoRA Fine-Tuning

## Problem Statement

Apply Low-Rank Adaptation (LoRA) to fine-tune a pretrained language model on a downstream task. Compare LoRA to full fine-tuning in terms of trainable parameters, memory, and task performance. Document the LoRA formulation and how to merge weights for inference.

## Mathematical Formulation

### LoRA Decomposition
(Placeholder for W' = W + BA, low-rank update)

### Rank and Dimension
(Placeholder for choice of rank r, A and B dimensions)

### Training and Merging
(Placeholder for training only LoRA params, merging for inference)

## Intuition

(Placeholder for intuition on why low-rank updates work, and when LoRA is sufficient vs when full fine-tuning is needed)

## Implementation Plan

1. Load a pretrained model (your M9 GPT or a small HuggingFace model)
2. Identify target layers (e.g., attention projections)
3. Inject LoRA layers: A (r x d) and B (d x r) per target matrix
4. Freeze base model; train only LoRA parameters
5. Implement merge: W_merged = W + B @ A
6. Evaluate on downstream task (e.g., classification, generation)
7. Compare: LoRA vs full fine-tune vs frozen (zero-shot) on same task
8. Report parameter count, memory, and metrics

(No code.)

## Required Experiments

- Fine-tune with LoRA on a downstream task
- Compare LoRA vs full fine-tuning: performance and trainable parameter count
- Ablate LoRA rank: r=4, 8, 16 (or similar); document effect on performance
- (Optional) Compare different target layers (e.g., only attention vs attention + MLP)

## Required Visualizations

- Table: trainable params, memory, task metric for LoRA vs full
- Rank ablation: task performance vs rank
- (Optional) Training curve for LoRA vs full

## Reflection Questions

1. Why does a low-rank update often capture task-specific adaptations?
2. When might LoRA underperform full fine-tuning?
3. How do you choose which layers to apply LoRA to?

## Completion Checklist

- [ ] LoRA implemented and applied to pretrained model
- [ ] Fine-tuning completes on downstream task
- [ ] Comparison to full fine-tune documented in `experiments_log.md`
- [ ] Rank ablation documented
- [ ] Journal entry in `learning_journal.md`
