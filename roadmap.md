# Roadmap: LLM Engineering Mastery

## Purpose

This document defines the progression rules, learning objectives, and exit criteria for the entire learning journey. It is the single source of truth for what must be completed before advancing.

## Progression Rules

1. **Sequential advancement:** You cannot start a milestone until all previous milestones in the same phase are completed.
2. **Phase completion:** You cannot enter a phase until the previous phase is fully complete.
3. **Exit criteria:** Each milestone has explicit exit criteria. All must be satisfied.
4. **Documentation:** Experiments and reflections must be logged in `experiments_log.md` and `learning_journal.md` respectively.

## Phase 1: Foundations

**High-level description:** Build the mathematical and computational foundations for deep learning. Understand gradients, backpropagation, and the mechanics of autodiff frameworks.

**Learning objectives:**
- Derive and implement gradient descent for a simple model
- Build a neural network with backpropagation from scratch
- Compare optimizer behaviors (SGD, momentum, Adam)
- Understand how PyTorch computes and stores gradients

**Milestones:**

| Milestone | Exit Criteria |
|-----------|---------------|
| M1 | Logistic regression trained on binary classification; manual gradient derivation documented |
| M2 | Multi-layer neural network with backprop; verified against analytical gradients |
| M3 | Comparison of SGD, Adam, and one variant on convergence; ablation documented |
| M4 | Trace of autograd for a simple model; understanding of computational graph |

---

## Phase 2: Transformers

**High-level description:** Implement attention and transformer blocks from first principles. Build a small autoregressive GPT to understand the architecture end-to-end.

**Learning objectives:**
- Implement scaled dot-product attention with masking
- Assemble a transformer block (attention + MLP + norms)
- Train a small GPT on a corpus and validate generation

**Milestones:**

| Milestone | Exit Criteria |
|-----------|---------------|
| M5 | Attention scores, weights, and outputs computed; intuition for Q, K, V |
| M6 | Full transformer block; correct causal masking for autoregression |
| M7 | Mini GPT trained; generates coherent sequences; loss curve documented |

---

## Phase 3: LLM Training

**High-level description:** Move from toy models to real LLM training pipelines. Cover tokenization, pretraining, parameter-efficient fine-tuning, and retrieval-augmented generation.

**Learning objectives:**
- Implement a tokenizer (BPE or similar)
- Pretrain a small GPT from scratch
- Fine-tune with LoRA
- Build a RAG system with retrieval and generation

**Milestones:**

| Milestone | Exit Criteria |
|-----------|---------------|
| M8 | Tokenizer encodes/decodes; vocabulary and merge rules documented |
| M9 | Small GPT pretrained on dataset; perplexity and samples reported |
| M10 | LoRA fine-tuning on a downstream task; comparison to full fine-tune |
| M11 | RAG pipeline: retrieval + generation; evaluation on QA task |

---

## Phase 4: Applications

**High-level description:** Apply LLMs to real-world systems: tool use, multi-agent coordination, and a capstone project integrating prior work.

**Learning objectives:**
- Implement an agent that calls external tools
- Design and build a multi-agent system
- Complete a capstone project demonstrating end-to-end LLM engineering

**Milestones:**

| Milestone | Exit Criteria |
|-----------|---------------|
| M12 | Agent correctly parses intent, selects tools, and uses outputs |
| M13 | Multi-agent system with defined roles and communication protocol |
| M14 | Capstone project documented; design, implementation, and evaluation complete |

---

## How This Repository Evolves

1. **Milestone completion:** As each milestone is completed, its README checklist is updated and code (if any) is added to the milestone folder.
2. **Experiments:** Each experiment is logged in `experiments_log.md` with date, objective, setup, metrics, and visualization path.
3. **Journal:** Reflections are added to `learning_journal.md` after completing milestones or encountering significant insights.
4. **Papers:** Notes from relevant papers are added to `papers_notes/` as the journey progresses.
5. **Visualizations:** Plots and diagrams are stored in `visualizations/` and referenced from experiments and journal entries.

The repository remains minimal until work is done. Empty sections are intentional placeholders for future content.
