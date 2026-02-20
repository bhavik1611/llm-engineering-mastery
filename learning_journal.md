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

## Entry: 2026-02-15 — M1 Vanishing Gradient & Activation Analysis

### What I learned

- **Why sigmoid derivative causes vanishing gradient** — Sigmoid asymptotically approaches 0 and 1 (does not "clip"); σ′(z) = σ(z)(1 − σ(z)) → 0 as |z| → ∞. When weights are initialized very large, outputs saturate and gradients become tiny everywhere except near the decision boundary (z ≈ 0, where σ′ ≈ 0.25).
- **Why logistic regression was trainable despite sigmoid** — BCE–sigmoid cancellation: the gradient ∂L/∂w ∝ (ŷ − y)x does not include σ′. The sigmoid derivative cancels out in the BCE gradient, so we get a clean signal (ŷ − y) and no vanishing. Single-layer convexity also helps.
- **What changes when stacking multiple sigmoid layers** — Chain rule multiplies derivatives across layers. Each layer contributes a factor ≤ 0.25; the product shrinks exponentially with depth → vanishing gradient.
  - If each derivative ≤ 0.25, then after L layers: \((0.25)^L\).
  - For \(L=10\): \((0.25)^{10} \approx 10^{-6}\).
- **How activation choice affects optimization geometry** — Sigmoid: curved landscape, gradient strongest near decision boundary (output ≈ 0.5), weak when saturated. ReLU: flat (zero gradient) for z < 0, constant derivative 1 for z > 0 → no vanishing in positive region, different loss surface that lets gradients flow farther in deep networks. ReLU avoids vanishing in positive region, but introduces "Dead neuron problem (z < 0 permanently)". That’s a good open question to note.
- **Conceptual shift (Day 1 → Day 3)** — Day 1: logistic regression as single-layer classifier with sigmoid and BCE. Day 3: activation functions as a design choice that shapes training dynamics—saturation, vanishing gradients, and why sigmoid fails in deep stacks while ReLU helps. Shift from "sigmoid works for classification" to "activation choice determines whether gradients flow through deep networks."

### What was difficult

- Internalizing that sigmoid itself is not the problem in logistic regression; the issue emerges only when stacking layers due to repeated derivative multiplication.
- Understanding that vanishing gradient is not a failure of gradient descent but a structural property of activation functions.
- Separating optimization geometry issues (conditioning, learning rate) from activation-induced gradient collapse.

### What I would do differently

- (To be filled as you reflect)

### Connections to prior work

- Builds on M1 logistic regression (sigmoid, BCE, gradients) and optimization geometry (gradient flow, loss surface). Sets up motivation for neural network activations (ReLU) in later milestones.

### Open questions

- How does weight initialization interact with activation saturation?
- Why do Xavier/He initialization methods help mitigate vanishing gradients?
- How does softmax + cross-entropy behave in multi-class settings?
- Why does ReLU change loss surface curvature?
- How do modern architectures (e.g., residual connections) mitigate vanishing gradients?

---

## Entry: 2026-02-15 — M2 Neural Network from Scratch

### What I learned

- **Backprop through 2 layers** — Chain rule: dL/dz₂ = ŷ−y (BCE+sigmoid cancel); dL/da₁ = dL/dz₂ @ W₂ᵀ; dL/dz₁ = dL/da₁ ⊙ σ'(z₁); weight grads = (input)ᵀ @ (upstream) / n.
- **XOR and depth** — XOR is linearly inseparable. 1-layer (logistic regression) ≈ 50% (random). 2-layer NN with hidden units learns non-linear boundary → ~100%.
- **Why depth helps** — Hidden layer computes non-linear features (e.g. AND, OR); output layer combines them. Single linear layer cannot approximate XOR.
- **Vanishing gradient and depth** — In logistic regression, σ′ cancels in the BCE gradient. In NNs, gradients flow layer-by-layer; each layer multiplies by σ′(z) ≤ 0.25. Product ≤ (0.25)^L → exponentially smaller with depth. Demo: plot gradient magnitude vs L on log scale.
- **ReLU vs sigmoid for gradient flow** — Sigmoid: σ′(z) → 0 as |z| → ∞ (saturation). ReLU: f′(z) = 1 for z > 0 → no shrinking in the positive region. Gradients can flow through many ReLU layers without vanishing.
- **Initialization and dead neurons** — ReLU is “dead” when z < 0 always → output 0, gradient 0. Bad init (small W, large −b) keeps most z < 0 → 100% dead. He init (W scaled by √(2/n_in)) spreads z around 0 so many neurons fire.

### Gradient check: what and why

**Purpose:** Verify that analytical (backprop) gradients match the true derivative. If backprop has a bug (wrong formula, shape error), training will fail. Gradient check catches this before wasting compute.

**Formula (central difference):**
$$
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}
$$

**Why add ε, subtract 2ε, add ε?** We perturb a single parameter in place:

1. θ += ε → compute L(θ+ε)
2. θ -= 2ε → now at θ−ε, compute L(θ−ε)  
   (from θ+ε down to θ−ε = step of 2ε)
3. θ += ε → restore original θ

**Validation:** Compare analytical (from backprop) vs numerical (finite diff). Max relative error < 1e−5 ⇒ pass.

### Gradient check: numerical example (linear layer, MSE)

**Setup:** Single weight \(w = 0.5\), input \(x = 1\), target \(y = 1\). Loss \(L = (z - y)^2\) with \(z = wx\). ε = 1e−5.

**Analytical:**
$$
\frac{\partial L}{\partial w} = 2(z - y) \cdot x = 2(0.5 - 1) \times 1 = -1.0
$$

| Step | Action | w | z = wx | L = (z−1)² |
|------|--------|---|--------|------------|
| Start | — | 0.50000 | 0.50 | 0.25 |
| 1 | w += 1e−5 | 0.50001 | 0.50001 | **0.24999** |
| 2 | w -= 2e−5 | 0.49999 | 0.49999 | **0.25001** |
| 3 | w += 1e−5 | 0.50000 | (restored) | — |

**Numerical gradient:**
$$
\frac{L(w{+}\varepsilon) - L(w{-}\varepsilon)}{2\varepsilon} = \frac{0.24999 - 0.25001}{2 \times 10^{-5}} = \frac{-0.00002}{0.00002} = -1.0
$$

**Match.** Analytical = −1.0, numerical = −1.0. For a full NN layer, we repeat this for every element of W and b, compare each, and report the max relative error across all parameters.

### What was difficult

- **Gradient check mechanics** — The add ε / subtract 2ε / add ε sequence was initially opaque. Why 2ε? The jump: we need L(θ−ε) but are at θ+ε; to reach θ−ε we must step back 2ε. Once visualized as “perturb up, perturb down, restore,” it clicked.

### What I would do differently

- (To be filled as you reflect)

### Connections to prior work

- M1: sigmoid, BCE, gradient descent. M2 stacks two such blocks and chains gradients. Vanishing gradient notebook (M1 closure) foreshadows why ReLU is preferred in deeper nets.

### Open questions

- He vs Xavier: when to use which? (He for ReLU, Xavier for sigmoid/tanh.)
- How do skip connections (ResNet) change gradient flow in very deep nets?

---
