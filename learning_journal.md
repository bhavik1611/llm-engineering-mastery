# Learning Journal

## Purpose

This journal captures reflections, conceptual syntheses, and corrected misconceptions from the LLM engineering journey. It serves as:

- A record of how understanding evolved over time
- A resource for interview preparation and portfolio discussions
- A forcing function for deep processing of concepts

## How Entries Should Be Written

- **Concise:** No filler. Each sentence should add information.
- **Rigorous:** Reference mathematics and code where relevant.
- **Honest:** Document confusion, mistakes, and breakthroughs.
- **Actionable:** Note what was unclear and what resolved it.

## Reflection Template

Use this structure for each journal entry:

```markdown
## Entry: [Date] — [Topic/Milestone]

### What I learned
- [Bullet points summarizing key insights]

### What was difficult
- [Stumbling blocks, confusions, or areas that required extra effort]

### What I would do differently
- [Changes to approach if starting over]

### Connections to prior work
- [How this relates to previous milestones or concepts]

### Open questions
- [Things still unclear or worth revisiting]
```

## Example Entry Structure

```markdown
## Entry: YYYY-MM-DD — M2 Neural Network from Scratch

### What I learned
- (Placeholder for insights)
- (Placeholder for insights)

### What was difficult
- (Placeholder for challenges)

### What I would do differently
- (Placeholder for retrospective)

### Connections to prior work
- (Placeholder for synthesis)

### Open questions
- (Placeholder for unresolved items)
```

---

## Journal Entries

## Entry: 2026-02-13 — M1 Logistic Regression from Scratch

### What I learned

- **Classification** — Binary classification: predict class ∈ {0, 1} from features. Logistic regression learns a linear decision boundary in feature space.
- **Logits and activation** — Raw score \(z = w^T x + b\) (logit) → sigmoid \(\sigma(z) = 1/(1+e^{-z})\) maps to probability \(\hat{y} \in (0,1)\). Step function gives same boundary but no gradients; sigmoid enables smooth optimization.
- **BCE vs MSE** — Binary cross-entropy \(L = -\big[y\log\hat{y} + (1-y)\log(1-\hat{y})\big]\) penalizes confident wrong predictions more than MSE → stronger gradient signal for learning.
- **Probability model** — Bernoulli: \(P(y|x) = \hat{y}^y(1-\hat{y})^{1-y}\). BCE is the negative log-likelihood.
- **Gradient derivation** — Chain rule yields \(\frac{\partial L}{\partial w} = (\hat{y} - y)x\), \(\frac{\partial L}{\partial b} = \hat{y} - y\). Elegant cancellation.
- **Update rules** — \(w := w - \eta \cdot \frac{1}{N}\sum_i (\hat{y}^{(i)} - y^{(i)}) x^{(i)}\); same for \(b\) without \(x\).
- **Linear separability** — Logistic regression learns only linear boundaries. Fails on XOR; needs feature engineering or nonlinear models.
- **L2 regularization** — Adds \(\frac{\lambda}{2}\|w\|^2\) to loss; shrinks weights. Bayesian view: L2 = Gaussian prior centered at 0 on \(w\).
- **Feature scaling** — Standardizing features (zero mean, unit variance) improves convergence and makes learning rate less sensitive.
- **Evaluation** — ROC, AUC, confusion matrix, precision, recall. Probability heatmaps show model confidence across feature space.

### What was difficult

- Connecting the analytic gradient derivation to the actual update rule (averaging over batch, minus sign for descent).
- Picking datasets that clearly demonstrate L2 regularization effects (e.g., outliers vs no outliers).

### What I would do differently

- Start with the probability interpretation (Bernoulli) before diving into loss; it makes BCE intuitive.
- Try L2 comparison on outlier-heavy data earlier to see the visual difference in decision boundaries.

### Connections to prior work

- Calculus (chain rule for gradients), linear algebra (\(w^T x\)), optimization (gradient descent). This is the base building block before neural networks.

### Open questions

- How does L1 (Laplace prior) differ from L2 in practice? When would one prefer L1?
- Multiclass extension: softmax + categorical cross-entropy.
