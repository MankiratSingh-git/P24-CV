# Technical Methodology: Semi-Supervised Action Recognition with FixMatch & $F^2$-FixMatch

## 1. Executive Summary & Problem Formulation

Video action recognition requires extracting dynamic spatial and temporal features across sequential video frames. While collecting unlabeled video is inexpensive, annotating temporal boundaries and action labels is prohibitively labor-intensive. 

> **Problem Statement (P24)**: Given the UCF-101 benchmark with 101 human action categories, learn a high-accuracy video classifier using strictly **10% labeled training videos** (953 labeled videos, ~9.4 videos/class) and **90% unlabeled videos** (8,584 unlabeled videos), closing the performance gap toward full supervision (9,537 labeled videos).

To systematically evaluate the benefit of unlabeled video data, this repository formulates and reproduces three core tiers alongside an oracle upper bound:

1. **Supervised Baseline (10% Labels Only)**: Trained exclusively on the 953 labeled videos with standard cross-entropy loss, establishing the lower-bound reference.
2. **FixMatch Baseline (Single-Stream RGB)**: Adapts consistency-based pseudo-labeling (Sohn et al., NeurIPS 2020) to 3D video recognition using weak-to-strong spatial augmentations (RandAugment + Cutout) and confidence thresholding ($\tau = 0.3$).
3. **$F^2$-FixMatch (Full Framework)**: State-of-the-art semi-supervised video framework (Tu et al., ACM TOMM 2025) integrating bi-modal representations (RGB + dynamic Temporal Gradient), **Frame-Level FixMatch**, and a unified **Feature-Level Consistency Module (P-Aug + MPL)** with decoupled classification heads.
4. **Supervised Upper Bound (Oracle, 100% Labels)**: Full-supervision ceiling trained across all 9,537 annotated videos.

---

## 2. Theoretical Formulation & System Architecture

The $F^2$-FixMatch architecture operates across three distinct hierarchical levels:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT & MODALITY LEVEL                                        │
│  - RGB Video Stream: x^V in R^{3 x T x H x W}                                                   │
│  - Temporal Gradient Stream: x^G = (x_t^V - x_{t+1}^V + 255) / 2                                │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FRAME-LEVEL FIXMATCH (Dual Stream)                              │
│  - Weak RGB + Weak TG -> Shared Backbone -> Ensemble Pseudo-Label:                              │
│      p_w = 0.5 * (p_w^{RGB} + p_w^{TG}),  y_hat = argmax(p_w),  mask = 1[max(p_w) >= tau]       │
│  - Strong RGB + Strong TG -> Shared Backbone -> 3D Head -> Cross-Entropy with y_hat             │
│  - Objective: alpha * L_u^s                                                                     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED FEATURE-LEVEL CONSISTENCY MODULE (P-Aug + MPL)                      │
│  - P-Aug (Feature-Level Perturbations):                                                         │
│      * Z_w -> Z_{w_p}: Detached weak feature + TAP + Spatial Dropout + CutMix                   │
│      * Z_s -> Z_{s_p}: Strong feature + TAP + Spatial Dropout + CutMix                          │
│  - MPL (Multi-branch Pseudo-Label Loss via Decoupled Head C_TAP):                               │
│      * Objective: beta * L_u^{w_p} + gamma * L_u^{s_p}                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Feature-Level Component Coupling
P-Aug and the auxiliary feature-level consistency losses form the feature-level component of our implementation. The perturbations only affect the representation through the corresponding consistency losses ($\beta$ and $\gamma$). While they are structurally separable by setting $\beta$ and $\gamma$ independently, Run D ($\beta=0, \gamma=0$) disables both to isolate the aggregate effect of the feature-level consistency module from the RGB+TG dual-stream configuration.

---

## 3. Mathematical Formulation

### 3.1 Input Modalities & Sampling
Let an input video clip consist of $T=8$ frames sampled with temporal interval $s=8$ (spanning 64 raw video frames):

1. **RGB Appearance Stream ($x^V$):**
   $$x^V = \{x_1^V, x_2^V, \dots, x_T^V\} \in \mathbb{R}^{3 \times T \times H \times W}$$

2. **Temporal Gradient Dynamic Stream ($x^G$):**
   Temporal gradients capture instantaneous velocity without the multi-gigabyte storage cost of precomputed optical flow:
   $$x_t^G = \frac{x_t^V - x_{t+1}^V + 255}{2}, \quad t \in \{1, 2, \dots, T\}$$
   Both modalities share spatial dimensions ($224 \times 224$) and are processed by a single shared 3D-ResNet-18 backbone.

---

### 3.2 Progressive Augmentation (P-Aug)

Progressive Augmentation applies perturbations in two cascading stages:

#### Stage 1: Frame-Level Augmentation
* **Weak Augmentation ($A_w$):** Scale-preserving RandomResizedCrop (scale $[0.8, 1.0]$, size $224 \times 224$) and horizontal flip ($p=0.5$).
* **Strong Augmentation ($A_s$):** Weak augmentation followed by **Video RandAugment** (sampled across 13 temporal operations with magnitude $M=10$) and **Video Cutout** ($128 \times 128$ rectangular spatial masking applied consistently across all $T$ frames).

#### Stage 2: Feature-Level Perturbations
Let $Z = E(x) \in \mathbb{R}^{B \times 512 \times T \times H \times W}$ denote the feature representation extracted by the shared 3D encoder before global pooling. The perturbator applies:
1. **Temporal Average Pooling (TAP):**
   $$\text{TAP}(Z) = \frac{1}{T} \sum_{t=1}^T Z_{:, :, t, :, :} \in \mathbb{R}^{B \times 512 \times H \times W}$$
2. **Spatial Feature Dropout:** Randomly zeroes 2D feature channels with probability $p=0.5$.
3. **Feature CutMix:** Swaps random rectangular feature patches between batch samples with probability $p=0.5$:
   $$\widetilde{Z} = M_{\text{feat}} \odot Z_i + (1 - M_{\text{feat}}) \odot Z_j$$
4. **Feature Horizontal Flip:** Horizontally flips 2D feature maps with probability $p=0.5$ as an implementation choice.

Applying these operations yields four representations for each unlabeled clip:
* $Z_w$: Frame-weak feature (unperturbed)
* $Z_s$: Frame-strong feature (unperturbed at feature stage)
* $Z_{w_p}$: Frame-weak feature perturbed via P-Aug
* $Z_{s_p}$: Frame-strong feature perturbed via P-Aug

---

### 3.3 Decoupled Classification Heads & Pseudo-Label Generation

To decouple spatio-temporal dynamics from frame-collapsed features, $F^2$-FixMatch employs two distinct classifier heads:
* **3D Spatio-Temporal Head ($C_{3D}$):** `AdaptiveAvgPool3d((1,1,1)) -> Dropout(0.5) -> Linear(512, 101)`. Evaluates $Z_w$ and $Z_s$.
* **TAP 2D Head ($C_{TAP}$):** `AdaptiveAvgPool2d((1,1)) -> Dropout(0.5) -> Linear(512, 101)`. Evaluates $Z_{w_p}$ and $Z_{s_p}$.

#### Dual-Stream Pseudo-Labeling
Pseudo-labels are generated by ensembling the weakly augmented RGB and TG predictions through the 3D head:
$$p_w = \frac{1}{2} \left( \operatorname{softmax}\left(C_{3D}(Z_w^{\text{RGB}})\right) + \operatorname{softmax}\left(C_{3D}(Z_w^{\text{TG}})\right) \right)$$
$$\hat{y} = \arg\max_c p_w(c), \qquad q = \max_c p_w(c)$$
$$m = \mathbb{1}[q \geq \tau] \quad (\tau = 0.3)$$

---

### 3.4 Multi-Task Objective Function & $Z_w$ Detachment

The joint optimization objective is:
$$\mathcal{L} = \mathcal{L}_s + \lambda_{\text{ramp}}(t) \cdot \left( \alpha \mathcal{L}_u^s + \beta \mathcal{L}_u^{w_p} + \gamma \mathcal{L}_u^{s_p} \right)$$

1. **Supervised Loss ($\mathcal{L}_s$):**
   $$\mathcal{L}_s = \frac{1}{|\mathcal{B}_l|} \sum_{i \in \mathcal{B}_l} \text{CE}\left(C_{3D}(E(x_i)), y_i\right)$$

2. **Frame-Level Strong Consistency Loss ($\mathcal{L}_u^s$):**
   $$\mathcal{L}_u^s = \frac{1}{|\mathcal{B}_u|} \sum_{j \in \mathcal{B}_u} m_j \cdot \frac{1}{2} \left[ \text{CE}\left(C_{3D}(Z_{s, j}^{\text{RGB}}), \hat{y}_j\right) + \text{CE}\left(C_{3D}(Z_{s, j}^{\text{TG}}), \hat{y}_j\right) \right]$$

3. **Weak Feature-Level Consistency Loss ($\mathcal{L}_u^{w_p}$):**
   Supervises perturbed weak features through the TAP head $C_{TAP}$:
   $$\mathcal{L}_u^{w_p} = \frac{1}{|\mathcal{B}_u|} \sum_{j \in \mathcal{B}_u} m_j \cdot \frac{1}{2} \left[ \text{CE}\left(C_{TAP}(Z_{w_p, j}^{\text{RGB}}), \hat{y}_j\right) + \text{CE}\left(C_{TAP}(Z_{w_p, j}^{\text{TG}}), \hat{y}_j\right) \right]$$

4. **Strong Feature-Level Consistency Loss ($\mathcal{L}_u^{s_p}$):**
   Supervises perturbed strong features through the TAP head $C_{TAP}$:
   $$\mathcal{L}_u^{s_p} = \frac{1}{|\mathcal{B}_u|} \sum_{j \in \mathcal{B}_u} m_j \cdot \frac{1}{2} \left[ \text{CE}\left(C_{TAP}(Z_{s_p, j}^{\text{RGB}}), \hat{y}_j\right) + \text{CE}\left(C_{TAP}(Z_{s_p, j}^{\text{TG}}), \hat{y}_j\right) \right]$$

> **Implementation Choice — $Z_w$ Feature Detach**:  
> In `train_f2_fixmatch.py`, the weak latent representation is detached prior to P-Aug:
> `_, logits_wp = model.forward_paug(z_w.detach())`.  
> The weak feature branch is detached in the current implementation, so its consistency loss ($\mathcal{L}_u^{w_p}$) does not backpropagate through the weak backbone features. This directs gradients from $\mathcal{L}_u^{w_p}$ exclusively into the TAP classification head ($C_{TAP}$) rather than into the backbone encoder. This is an implementation choice and is documented as such, rather than an empirically proven claim of mode collapse prevention. Backbone gradient updates from unlabeled data are maintained through the strongly augmented student pathways ($Z_s$ and $Z_{s_p}$).

Hyperparameters follow official specifications: $\alpha = 2.0$, $\beta = 2.0$, $\gamma = 1.0$, and $\tau = 0.3$.

---

## 4. Network Architecture: 3D-ResNet-18

The backbone is an inflated 3D-ResNet-18 (31.92M parameters) adhering strictly to Table 2 of Tu et al. (2025):

| Stage | Layer Configuration | Output Spatial Dim | Output Channels | Temporal Dim ($T$) |
| :--- | :--- | :---: | :---: | :---: |
| **Input** | Raw video clip / Temporal Gradient | $224 \times 224$ | 3 | 8 |
| **conv1** | $1 \times 7 \times 7$, spatial stride 2 | $112 \times 112$ | 64 | 8 |
| **pool1** | $1 \times 3 \times 3$ max pool, spatial stride 2 | $56 \times 56$ | 64 | 8 |
| **res2** | 2 blocks: $[1 \times 3 \times 3, 64]$ | $56 \times 56$ | 64 | 8 |
| **res3** | 2 blocks: $[1 \times 3 \times 3, 128]$, spatial stride 2 | $28 \times 28$ | 128 | 8 |
| **res4** | 2 blocks: $[3 \times 3 \times 3, 256]$ (inflated), spatial stride 2 | $14 \times 14$ | 256 | 8 |
| **res5** | 2 blocks: $[3 \times 3 \times 3, 512]$ (inflated), spatial stride 2 | $7 \times 7$ | 512 | 8 |
| **Head $C_{3D}$** | `AdaptiveAvgPool3d((1,1,1)) -> Dropout(0.5) -> Linear(512, 101)` | $1 \times 1$ | 101 | 1 |
| **Head $C_{TAP}$** | `AdaptiveAvgPool2d((1,1)) -> Dropout(0.5) -> Linear(512, 101)` | $1 \times 1$ | 101 | 1 |

*Crucial Design Feature*: No temporal pooling or temporal stride is applied anywhere in the backbone; temporal length $T=8$ is preserved throughout all residual stages.

---

## 5. Training Schedules & Progressive Ramp-Up

Training is partitioned into three distinct phases across 220 epochs:

1. **Supervised Warmup (Epochs 1–40):**
   * Unsupervised weights set to zero: $\alpha = 0, \beta = 0, \gamma = 0$.
   * Backbone and $C_{3D}$ head train strictly on the 10% labeled data ($\mathcal{L}_s$).
   * Enables the encoder to learn stable preliminary class cluster centers before unlabeled pseudo-labeling begins.
2. **Progressive Ramp-Up (Epochs 41–70):**
   * Unsupervised loss weight $\lambda_{\text{ramp}}(t) = (t - 40) / 30$ scales linearly from 0.0 to 1.0.
   * Pseudo-label mask coverage expands smoothly from 13.6% to 75.2%.
   * Detached $Z_w$ ensures weak branch consistency updates the TAP head directly without updating weak backbone features.
3. **Joint Semi-Supervised Convergence (Epochs 71–220):**
   * All loss terms operate at full weight ($\lambda_{\text{ramp}} = 1.0$).
   * Cosine annealing learning rate schedule decays $\eta$ from 0.2 down to $10^{-4}$.

---

## 6. Official Evaluation Protocol (30-View Multi-Crop)

To prevent evaluation bias, testing strictly follows Section 4.2 of Tu et al. (2025):
* **Temporal Sampling:** 10 uniform clips sampled along the entire duration of the video.
* **Spatial Cropping:** 3 crops per clip (left/top, center, right/bottom) scaled to shorter side 256 and cropped to $256 \times 256$.
* **Total Views per Video:** $10 \text{ temporal} \times 3 \text{ spatial} = \mathbf{30 \text{ views}}$.
* **Video Prediction:** Softmax probabilities are averaged across all 30 views:
  $$P(y = c \mid V) = \frac{1}{30} \sum_{k=1}^{30} \operatorname{softmax}\left(f_\theta(v_k)\right)_c$$
* Total validation clips processed: $3,783 \text{ videos} \times 30 = \mathbf{113,490 \text{ clips}}$ per evaluated checkpoint.
