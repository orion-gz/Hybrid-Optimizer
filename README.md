# Hybrid Optimizer: AdamW to SAM Transition Strategy

> **Finding optimal switching point for AdamW to SAM optimizer in deep learning**

## Overview
This repository investigates **Hybrid Optimizer** strategies that combine AdamW's fast convergence with SAM's flat minima exploration. 

> **The core question: when and how should we switch from AdamW to SAM to maximize generalization?**

Through systematic experimentation on `WRN-28-10` across **CIFAR-10** and **CIFAR-100**, we discover that:
1. **Cosine decay implicitly performs flat minima exploration**: mechanistically overlapping with SAM, which explains why SAM provides no benefits when cosine decay is already present. 
2. **A late-phase switch point exists** where AdamW to SAM consistently outperforms AdamW + cosine alone.
3. **Architecture-dependent sharpness dynamics**: determine whether SAM is beneficial or harmful 

---

## Key Results 

### Cosine Decay Ablation (CIFAR-100)

| AdamW Schedule | SAM Schedule | Switch Epoch | Test Acc |
|:---:|:---:|:---:|:---:|
| Cosine | — | No switching (AdamW Only) | 81.45% |
| Cosine | Cosine | 151 | 81.41% |
| **Constant** | **Cosine** | **151** | **81.24%** |
| Constant | — | No switching (AdamW Only)  | 77.93% |

> Cosine decay alone (AdamW + cosine, no SAM) provides +3.52%p. Switching to SAM with constant LR AdamW (AdamW constant → SAM + cosine, switch@151) provides +3.31%p. When cosine decay is already present in the AdamW phase, adding SAM provides **no additional benefit — the two mechanisms are functionally overlapping**.

### Fixed-Ratio Switching (AdamW Constant LR → SAM Cosine)

| Switch Epoch | Ratio | SAM Epochs | CIFAR-100 | CIFAR-10 |
|:---:|:---:|:---:|:---:|:---:|
| 50 | 17% | 250 | 80.86% | — |
| 100 | 33% | 200 | 80.97% | 97.05% |
| 150 | 50% | 150 | 81.58% | 97.08% |
| 200 | 67% | 100 | 81.93% | 97.29% |
| **225** | **75%** | **75** | **82.30%** | **97.34%** |
| 250 | 83% | 50 | 81.71% | 97.30% |
| 275 | 92% | 25 | 81.63% | — |

> Both datasets peak at switch@225 (75% of total epochs), forming an inverted-U curve and surpassing the cosine decay baseline.

### Architecture-Dependent Sharpness
| Architecture | Sharpness Start | Sharpness End | Trend | SAM Effect |
|:---:|:---:|:---:|:---:|:---:|
| WRN-28-10 | 0.06 | 0.60 | ↗ Increasing | Beneficial |
| ViT-S/8 | 1.89 | 0.17 | ↘ Decreasing | Harmful (−21%p) |
| ViT-S/16 | 0.37 | 0.04 | ↘ Decreasing | Harmful (−4.8%p) |

---

## Method
### Hybrid Training Framework
```
Phase 1 - AdamW (constant LR, no cosine decay)
    Warmup: 10 epochs (LR 0 → 0.001)
    LR: 0.001 constant
    Goal: learn representation without implicit flatness bias

Phase 2 - SAM (AdamW base, cosine decay)
    Restart LR: 0.0003 (= 0.001 x 0.3, restart ratio is a value determined by experimental results)
    Scheduler: CosineAnnealingLR (T_max = remaining epochs)
    Rho warmup: 0.02 → 0.15 over 20 epochs
    Momentum state transferred from AdamW
```

### Sharpness Measurement
We measure loss landscape sharpness throughout training as:

$$ sharpness = \mathcal{L}(\mathcal{w} + \rho \cdot \frac{g}{\|g\|}) - \mathcal{L}(\mathcal{w}) $$

where $g = \nabla\mathcal{L}(\mathcal{w})$. This requires 1 forward + 1 backward + 1 forward pass on a probe set (20% of validation data), approximately 10 $\times$ cheaper than simulation-based approaches.

### Why $\lambda_{max}(H)$ Define Sharpness
From Taylor expansion at a minimum  $\theta^*$ where  $\nabla\mathcal{L}(\theta^*) = 0$ :

$$ \mathcal{L}(\theta^* + \epsilon) - \mathcal{L}(\theta^*) \approx \frac{1}{2}\epsilon^TH\epsilon $$

Maximizing over $\|\epsilon\| \le \rho$ yield $\frac{1}{2}\lambda_{max}(H)\rho^2$, where $\lambda_{max}(H)$ is the largest Hessian eigenvalue. The optimal perturbation direction is the corresponding eigenvector - exactly what SAM's first step approximates.

## Version of Switcher
The path to the final approach wetn through several failed attempts - each failure was informative.

### Multi-signal based switcher (ver01 - ver03) - Hard to triggered and setting Hyperparameters
Combining 4 signals (plateau, generalization gap, gradient norm, oscillation) into a weighted sum. Too many parameters, conditions never met in practice.

### Nesterov Simulation based switcher (ver04 - ver06) - SAM lost every comparison
Copy weights, simulate SAM vs AdamW for N steps, switch if SAM accuracy gain > threshold. In 15 fair comparisons, SAM never outperformed AdamW in short-horizon accuracy.

> Key Insight: SAM intentionally sacrifices short-term progress for long-term landscape geometry.

### Sharpness-based switcher (ver07) - Triggered but missed optimal
Directly measure sharpness instead of simulating. Successfully triggered transitions (epoch 151-181), but consistently missed the optimal switch point (epoch 225).

> Key Insight: Sharpness EMA fluctuates rather than monotonically increasing - impossible to target a specific epoch with a threshold alone.

### Fixed-Ratio switching - Final
Systematic sweep across switch points revealed an inverted-U curve peaking at 75% of total epochs, reproducible across CIFAR-10 and CIFAR-100.

---

## Experimental Setup

```python
Model:        WRN-28-10
Datasets:     CIFAR-10, CIFAR-100
Epochs:       300
Batch size:   256
Seed:         42

AdamW:        lr=0.001, weight_decay=0.05, warmup=10ep
SAM:          base=AdamW, restart_lr=0.0003
              rho: 0.02 → 0.15 (warmup 20ep)
              cosine decay (T_max=remaining epochs)

Augmentation: AutoAugment + Mixup(α=0.8) + CutMix(α=1.0)
Label smooth: 0.1
```

---

## Related Work
- SAM — Foret et al., ICLR 2021
- Late-phase SAM (SGD→SAM) — Zhang et al., ICLR 2025
- Edge of Stability — Cohen et al., ICLR 2022
- ASAM — Kwon et al., ICML 2021
- ESAM — Du et al., ICLR 2022

---

## Limitations
- Architecture: WRN-28-10 only (CNN). ViT results show SAM is harmful, but the 75% rule is not tested on other CNNs.
- Scale: CIFAR-10/100 only. ImageNet validation is needed.
- The 75% rule holds across CIFAR-10 and CIFAR-100 at 300 epochs, but epoch-budget variation experiments suggest the optimal ratio shifts with total epoch count — the rule may be epoch-budget-dependent.
- Seed variance is not reported; single-run results throughout.

---

## Appendix: Mathematical Background

### A. Why sharpness = $\lambda_{max}(H)$
The intuition behind sharpness is _how much does the loss increase when parameters are perturbed slightly?_

Formally:

$$\text{sharpness} = \max_{\|\epsilon\| \leq \rho} \left[ \mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta) \right]$$

This is exactly the objective SAM minimizes. To solve it, apply a second-order Taylor expansion around $\theta$:

$$\mathcal{L}(\theta^* + \epsilon) - \mathcal{L}(\theta^*) \approx \frac{1}{2}\epsilon^TH\epsilon$$


**Why does the second-order term take this form?** Parameterize the perturbation path as $g(t) = \mathcal{L}(\theta + t\epsilon) and expand in t:

$$ g(t) = g(0) + g'(0)t + \frac{1}{2}g''(0)t^2 + \dots $$

Computing derivatives via the chain rule:

$$\begin{align} 
g'(0)&= \nabla\mathcal{L}(\theta)^T\epsilon \\
g''(0)&=\epsilon^TH\epsilon
\end{align}$$

where $H_{ij} = \frac{\partial^2\mathcal{L}}{\partial \theta_i \partial \theta_j}$. Setting $t = 1$ gives the Taylor expansion above. The $\frac{1}{2}$ comes from the $\frac{1}{2!}$ coefficient - identical to $\frac{1}{2}f''(x)\epsilon^2$ in the scalar case.

**Maximizing over the constraint ball** $\|\epsilon\| \leq \rho$. Since $H$ is symmetric, it admits an eigen decomposition $H = V\Lambda V^T$. Substituting $\epsilon = V\alpha$ (so $\|\alpha\| = \|\epsilon\|$):

$$ \epsilon^T H \epsilon = \alpha^T \Lambda \alpha = \sum_i \lambda_i \alpha_i^2$$

Under $\|\epsilon\| \leq \rho^2$, this is maximized by placing all weight on the largest eigenvalue:

$$\max_{\|\epsilon\| \leq \rho} \frac{1}{2} \epsilon^T H \epsilon = \frac{1}{2} \lambda_{\max}(H) \cdot \rho^2$$
 
The optimal perturbation direction is the eigenvector $\mathcal{v}_1$ corresponding to $\lambda_{max}$ - exactly the direction SAM's first step approximates. Therefore **sharpness is proportional to** $\lambda_{max}(H)$: a larger maximum eigenvalue means the loss rises more steeply in the most sensitive direction.

### B. Why the GD Stability Condition is $\eta < \frac{2}{\lambda_{max}(H)}$
Consider gradient descent near a minimum $\theta^*$ where $\nabla\mathcal{L}(\theta^*) = 0$:
$$\theta_{t + 1} = \theta_t - \eta \nabla\mathcal{L}(\theta_t)$$

Using the first-order Taylor approximation $\nabla\mathcal{L}(\theta_t) \approx H(\theta_t - \theta^*)$, the error $e_t = \theta_t - \theta^*$ satisfies:

$$e_{t+1} = (I - \eta H)e_t$$

This is a linear recurrence. For convergence we need $\|e_\| → 0$, which requires all eigenvalues of $(I - \eta H)$ to lie strictly inside the unit circle.

For each eigenvalue $\lambda_i$ of H, the corresponding eigenvalue of $(I - \eta H)$ is $(1 - \eta \lambda_i)$. The stability condition is:

$$|1 - \eta \lambda_i| < 1 \quad \Longleftrightarrow \quad 0 < \eta \lambda_i < 2 \quad \Longleftrightarrow \quad 0 < \eta < \frac{2}{\lambda_i}$$
 
The binding constraint comes from the largest eigenvalue:
 
$$\boxed{\eta < \frac{2}{\lambda_{\max}(H)}}$$
 
**Geometric interpretation:** If $\eta$ is too large, the update overshoots the minimum along the sharpest direction, bouncing back and forth with growing amplitude. The sharper the minimum (larger $\lambda_{\max}$), the smaller the learning rate must be to stay stable.
 
This has a direct implication for sharpness filtering: **a flat minimum with small $\lambda_{\max}$ is reachable with a large $\eta$, while a sharp minimum with large $\lambda_{\max}$ is only reachable with a small $\eta$**. Large learning rates therefore act as an implicit filter that excludes sharp minima.

### C. Cosine Decay as Implicit Flat-Minima Exploration
This section explains the mechanism behind the empirical observation that cosine decay and SAM provide overlapping benefits (Section 3.2 results).
 
**The key identity from Section B:** gradient descent with learning rate $\eta$ can only stably converge to minima satisfying $\lambda_{\max}(H) < 2/\eta$. Equivalently:
 
$$\text{reachable sharpness} < \frac{2}{\eta}$$
 
Cosine decay schedules the learning rate as:
 
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$
 
This creates a **dynamic sharpness ceiling** $2/\eta_t$ that evolves over training:
 
| Training phase | $\eta_t$ | Sharpness ceiling $2/\eta_t$ | Effect |
|:---:|:---:|:---:|:---:|
| Early ($t \approx 0$) | $\eta_{\max}$ (large) | Low | Only flat minima are stable |
| Late ($t \approx T$) | $\eta_{\min}$ (small) | High | Sharp minima become reachable |
 
At the start of training, the large learning rate destabilizes any trajectory that enters a sharp minimum — the optimizer is automatically ejected and continues exploring. Only regions where $\lambda_{\max}(H) < 2/\eta_{\max}$ are stable. As training progresses and $\eta_t$ decreases, the optimizer settles more finely within the already-flat region it has entered.
 
**Connection to Cohen et al. (2021) — "Edge of Stability":** Empirically, gradient descent tends to operate near $\lambda_{\max}(H) \approx 2/\eta$. Under cosine decay, this means the optimizer actively tracks the moving sharpness ceiling — early in training, it inhabits flat regions because only flat regions are stable; late in training, it refines within those regions as $\eta \to 0$.
 
**Why this mimics SAM:** SAM explicitly seeks flat minima by minimizing the loss at a perturbed point $w + \rho \cdot g / \|g\|$, penalizing regions where small perturbations cause large loss increases. The cosine decay mechanism achieves a similar outcome through an entirely different route: by making sharp regions dynamically unstable during the high-$\eta$ phase.
 
**Quantitative comparison from our experiments (CIFAR-100):**
 
| Mechanism | Effect on Test Acc | Route |
|:---|:---:|:---|
| Cosine decay alone | +3.52%p over constant LR | Implicit: LR-based sharpness filter |
| AdamW→SAM (no cosine in AdamW phase) | +3.31%p over constant LR | Explicit: perturbation-based |
| Both together | ≈ +3.48%p (no additive gain) | Redundant overlap |
 
The near-identical magnitudes (+3.52%p vs +3.31%p) and the absence of additive benefit when combined provide strong empirical support for the mechanistic overlap hypothesis.
 
> [!IMPORTANT] A formal proof that cosine decay is equivalent to SAM does not exist in the literature. The argument above is a theoretical interpretation supported by (1) the Edge of Stability finding (Cohen et al., 2021), (2) the implicit bias of large learning rates toward flat minima (Li et al., 2019; Damian et al., 2023), and (3) the empirical results in this repository. It should be treated as a well-supported hypothesis, not an established theorem.
