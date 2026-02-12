# llm-engineering-mastery

## Project Title

Mastering Large Language Models: From First Principles to Production Systems

## Vision Statement

This repository documents a systematic journey of building deep understanding in Large Language Model engineering. The goal is to move from foundational machine learning concepts through transformer architectures, pretraining, fine-tuning, and production-grade applications. Every concept is derived from first principles, implemented explicitly, and validated through experiments.

## Learning Philosophy

This project follows a deliberate balance:

- **30-40% Theory:** Mathematical foundations, intuition-building, and literature review
- **60-70% Implementation:** Hands-on coding, debugging, and experimentation

Concepts are not absorbed passively. Each milestone requires formulation, implementation, validation, and reflection.

## Milestone-Based Progression

Progression is structured by **milestones**, not by calendar weeks. Each milestone has:

- A clear problem statement
- Mathematical formulation requirements
- Implementation deliverables
- Required experiments and visualizations
- Exit criteria (completion checklist)

You cannot advance to the next milestone until all exit criteria for the current milestone are satisfied. This ensures no gaps in understanding.

## Overview of Phases

| Phase | Focus | Milestones |
|-------|-------|------------|
| **Phase 1: Foundations** | Core ML and autodiff mechanics | M1-M4 |
| **Phase 2: Transformers** | Attention and transformer blocks | M5-M7 |
| **Phase 3: LLM Training** | Tokenization, pretraining, fine-tuning, RAG | M8-M11 |
| **Phase 4: Applications** | Tool use, multi-agent systems, capstone | M12-M14 |

## Milestone Checklist

### Phase 1: Foundations
- [ ] M1: Logistic Regression from scratch
- [ ] M2: Neural Network from scratch
- [ ] M3: Optimizer Study (SGD, Adam, variants)
- [ ] M4: PyTorch Internal Mechanics

### Phase 2: Transformers
- [ ] M5: Attention from scratch
- [ ] M6: Transformer Block
- [ ] M7: Mini GPT

### Phase 3: LLM Training
- [ ] M8: Tokenizer from scratch
- [ ] M9: Small GPT Pretraining
- [ ] M10: LoRA Fine-Tuning
- [ ] M11: RAG System

### Phase 4: Applications
- [ ] M12: Tool Agent
- [ ] M13: Multi-Agent System
- [ ] M14: Capstone Project

## Folder Structure

```
llm-engineering-mastery/
├── README.md              # This file: project overview and navigation
├── roadmap.md              # Detailed progression rules and exit criteria
├── learning_journal.md     # Reflection and conceptual synthesis
├── experiments_log.md      # Experiment tracking and metrics
│
├── phase1_foundations/     # Core ML: gradients, optimization, autodiff
├── phase2_transformers/   # Attention, transformer blocks, small GPT
├── phase3_llm_training/   # Tokenization, pretraining, fine-tuning, RAG
├── phase4_applications/   # Agents, tools, multi-agent, capstone
│
├── papers_notes/          # Research paper summaries and insights
└── visualizations/        # Plots and diagrams for understanding
```

Each phase contains a `README.md` describing the phase and a folder per milestone (e.g., `M1_logistic_regression/`). Each milestone folder contains a `README.md` with problem statement, math, implementation plan, and completion checklist.

## Supporting Documents

- **experiments_log.md:** Records all experiments run during milestones. Each entry includes objective, setup, metrics, and visualization references. Used to compare ablations and track reproducibility.

- **learning_journal.md:** Structured reflections on concepts, breakthroughs, and misconceptions. Entries follow a template to ensure depth. Used for synthesizing understanding and preparing for interviews or portfolio discussions.
