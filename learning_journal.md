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

---

## Entry: 2026-02-15 — M1 Optimization Geometry & Learning Rate Dynamics

### What I learned

- **Loss as a surface** — For 2D weights (\(w_1, w_2\)), the BCE loss defines a surface \(L(w_1, w_2)\). Contour lines represent constant-loss regions. Logistic regression produces a convex "bowl"-shaped surface.
- **Gradient as normal vector** — The gradient \(\nabla L(\mathbf{w})\) is perpendicular to contour lines. This is because contour lines represent directions where loss does not change, so the gradient (direction of maximum increase) must be orthogonal to them.
- **Descent direction** — Gradient descent updates: \(\mathbf{w} := \mathbf{w} - \eta \nabla L(\mathbf{w})\). The negative gradient moves directly toward steepest decrease in loss.
- **Circular vs elliptical contours** —

  - Circular contours \(\implies\) equal curvature in all directions \(\implies\) straight-line convergence.
  - Elliptical contours \(\implies\) unequal curvature \(\implies\) zig-zag trajectory due to steep and shallow directions.
- **Conditioning** — Elongated ellipses indicate high condition number (ratio of largest to smallest curvature/eigenvalues). Poor conditioning causes slow convergence and oscillation.
- **Learning rate dynamics** —

  - Small \(\eta\): slow but stable convergence.
  - Optimal \(\eta\): smooth and fast convergence.
  - Large \(\eta\): overshooting \(\rightarrow\) oscillation \(\rightarrow\) divergence.
- **Why overshooting happens** — If learning rate exceeds curvature scale (roughly related to largest eigenvalue of the Hessian), steps become unstable and amplify error instead of reducing it.
- **Gradient norm meaning** — \(|\nabla L(\mathbf{w})| \to 0\) indicates approach to a stationary point. In convex logistic regression, this implies convergence to the global minimum.
- **Feature scaling & geometry** —

  - Large feature magnitude stretches the loss surface into narrow valleys.
  - Gradient descent zig-zags across steep directions.
  - Scaling reduces condition number, making contours more circular and optimization faster.
- **Scaling invariance of model** — If feature \(x_j\) is scaled by \(c\), optimal weight scales by \(1/c\). Prediction \(w_j x_j\) remains invariant. Scaling changes optimization path, not model expressiveness.
- **Curvature-aware updates** — Scaling updates differently per direction (e.g., using Hessian inverse) corrects zig-zag behavior. This is the intuition behind Newton’s method and adaptive optimizers (Adam, RMSProp).

---

### What was difficult

- Fully internalizing why gradient is perpendicular to contour lines (needed geometric reasoning, not just algebra).
- Understanding that convexity guarantees no bad local minima but does NOT guarantee fast convergence.
- Separating feature scaling (conditioning problem) from L2 regularization (overfitting + stability problem).

---

### What I would do differently

- Visualize loss surface earlier in the learning process instead of focusing only on algebra.
- Think in terms of geometry (contours and curvature) before jumping to optimization formulas.
- Explicitly compute and visualize gradient vectors before implementing full training loops.

---

### Connections to prior work

- Linear algebra: transpose (\(X^T\)) projects error into feature space.
- Calculus: gradient as direction of steepest ascent; Hessian encodes curvature.
- Systems thinking: conditioning is analogous to poorly scaled metrics in distributed systems affecting optimization and stability.
- Optimization theory: learning rate interacts with curvature; connects to eigenvalues and stability conditions.

---

### Open questions

- How exactly does the Hessian matrix relate to curvature in multiple dimensions?
- What is the formal relationship between learning rate stability and eigenvalues?
- How do adaptive optimizers approximate curvature without explicitly computing Hessian?
- How does this extend to non-convex deep networks (where saddle points exist)?

---
