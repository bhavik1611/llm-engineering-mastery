# Experiments Log

## Purpose

This document records all experiments conducted during the LLM engineering mastery journey. It enables:

- Reproducibility of results
- Comparison across ablations and configurations
- Tracking of metrics and visualizations over time
- Clear documentation for portfolio and technical discussions

## Experiment Logging Template

Use this structure for each experiment:

```
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

| Experiment | Visualization | File | Purpose |
|------------|----------------|------|---------|
| (name) | (e.g., loss curve) | (path) | (why it was created) |

---

## Experiments

(Experiments will be appended below as they are run.)
