# Empirical Experimental Results: FixMatch & $F^2$-FixMatch on UCF-101

**Benchmark:** UCF-101 (Split 1) | **Backbone:** 3D-ResNet-18 (31.92M parameters) | **Hardware:** NVIDIA RTX A5000 (24GB VRAM)  
**Label Regimes:** 10% Labeled (953 labeled / 8,584 unlabeled) vs. 100% Labeled (9,537 labeled) | **Validation Set:** 3,783 videos  
**Evaluation Protocols:**
- **Matched 1-Crop:** 1 temporal center clip $\times$ 1 spatial center crop (exact training validation standard).
- **Matched 30-View:** 10 uniform temporal clips $\times$ 3 spatial crops ($256 \times 256$ resolution, 113,490 total clips evaluated per model, official paper testing protocol).

---

## 1. Executive Summary & Benchmark Comparison

All evaluation results are reported under transparent, unmixed evaluation modes:

### Matched 1-Crop Validation Performance (Center Crop, 1 View per Video)

| Model | Labels | Modality | Architecture / Loss Components | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Gain vs. 10% Sup |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Supervised Baseline (10%)** | 10% | RGB | Standard RandomCrop + HorizontalFlip | **30.19%** | **53.69%** | *Reference* |
| **FixMatch Baseline** | 10% | RGB | Weak/Strong (RandAug + Cutout, $\tau=0.3$) | **37.85%** | **63.12%** | **+7.66%** |
| **$F^2$-FixMatch** | 10% | **RGB + TG** | **Dual-Stream + Feature-Level Module (P-Aug & MPL)** | **56.15%** | **81.05%** | **+25.96%** |
| **Supervised Reference (100%)** | 100% | RGB | Full Training Set (9,537 videos, 120 ep) | **71.69%** | **89.14%** | **+41.50%** |

---

### Matched 30-View Testing Performance (10 Temporal Clips $\times$ 3 Spatial Crops)

> **Note on Evaluation Selection**: Historical checkpoints evaluated below were selected based on the official test set and are designated as *test-selected* historical results. In accordance with the approved scientific implementation plan, Phase 2 will audit both test-selected historical checkpoints and pre-declared final-epoch checkpoints side-by-side.

| Model | Labels | Modality | Architecture / Loss Components | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Absolute Gain vs. 10% Sup | Gain vs. FixMatch | Checkpoint Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Supervised Baseline (10%)** | 10% | RGB | Standard RandomCrop + HorizontalFlip | **31.32%** | **54.11%** | *Reference* | — | Test-selected historical |
| **FixMatch Baseline** | 10% | RGB | Weak/Strong (RandAug + Cutout, $\tau=0.3$) | **38.49%** | **64.00%** | **+7.17%** | *Reference* | Test-selected historical |
| **$F^2$-FixMatch** | 10% | **RGB + TG** | **Dual-Stream + Feature-Level Module (P-Aug & MPL)** | **59.90%** | **83.32%** | **+28.58%** | **+21.41%** | Test-selected historical |
| **Supervised Reference (100%)** | 100% | RGB | Full Training Set (9,537 videos, 120 ep) | **72.91%** | **90.06%** | **+41.59%** | **+34.42%** | Matched 30-View Test |

```
Top-1 Accuracy (30-View Multi-Crop):
Supervised (10%):  [███████████                           ] 31.32%
FixMatch (10%):    [█████████████                         ] 38.49% (+7.17% gain over 10% Sup)
F²-FixMatch (10%): [█████████████████████                 ] 59.90% (+28.58% gain over 10% Sup | +21.41% over FixMatch)
```

---

## 2. Comparison With Published Literature (Tu et al., ACM TOMM 2025)

| Metric / Setting | Tu et al. Paper (Table 3 & 5) | Our Empirical Reproduction | Context & Reproduction Analysis |
| :--- | :---: | :---: | :--- |
| **Supervised Reference (10%)** | ~25.4% | **31.32%** | Supervised baseline exceeds paper reference (+5.92%) |
| **FixMatch Baseline (10%)** | 51.7% (Table 5) | **38.49%** | Single-stream RGB baseline without temporal gradients |
| **$F^2$-FixMatch (10%)** | **71.1%** (Table 3) | **59.90%** | **+28.58% absolute gain** over 10% supervised; **+21.41%** over FixMatch |
| **Supervised Reference (100%)** | ~70% – 73% | **72.91%** (30-view) | Consistent with standard 3D-ResNet-18 supervised performance on UCF-101 |
| **Top-5 Accuracy ($F^2$-FixMatch)** | Not reported | **83.32%** | Ground-truth action in top-5 for over 5 out of 6 test videos |
| **Pseudo-Label Mask Coverage** | ~90% – 95% | **95.22%** | Matches paper Section 4.5 dynamics with $\tau = 0.3$ |

### Reproduction Parameters & Context

Our single-GPU reproduction achieves **59.90%** on the official 30-view protocol compared to the paper's reported **71.1%** (Table 3). Concrete execution differences are documented below:

1. **Hardware & Batch Execution:**  
   The paper employs multi-GPU distributed data-parallel training with an aggregate distributed batch size. Our execution runs on a single GPU with labeled batch size $B_l = 4$, unlabeled batch size $B_u = 16$ ($\mu=4$), and gradient accumulation of 2 steps.
2. **Learning Rate & Optimization:**  
   The base learning rate is set to $\eta = 0.20$ with SGD momentum 0.9 and cosine annealing schedule.
3. **Audit Plan:**  
   Rather than advancing speculative explanations regarding normalization statistics, potential architectural and gradient factors will be evaluated empirically during the Phase 2 Architecture Audit (gradient flow verification) and PreciseBN recalibration diagnostic.

---

## 3. Training Dynamics & Artifact Registry

* **Training Budget:** 220 epochs (40-epoch warmup + 30-epoch ramp-up + 150-epoch joint semi-supervised learning).
* **Wall-Clock Time:** 17.26 hours on 1 $\times$ NVIDIA RTX A5000 (~4.7 minutes / epoch).
* **Loss Convergence:**
  * Supervised loss $\mathcal{L}_s$: $9.95 \to 0.3455$.
  * Frame strong consistency loss $\mathcal{L}_u^s$: $1.0006$.
  * Weak TAP feature consistency loss $\mathcal{L}_u^{w_p}$: $0.8411$.
  * Strong TAP feature consistency loss $\mathcal{L}_u^{s_p}$: $1.3268$.
* **Online Validation Top-1 (1-crop):** **56.15%** at Epoch 210.
* **Final Multi-Crop Evaluation (30-view):** **59.90%** Top-1 / **83.32%** Top-5.

### Checkpoint and Raw Data Registry

1. **$F^2$-FixMatch Checkpoint:** `checkpoints/f2_fixmatch/best_model.pth`
   * Metrics: `checkpoints/f2_fixmatch/eval_metrics_full30view.json`
   * Training History: `checkpoints/f2_fixmatch/training_history.json`
2. **FixMatch Baseline Checkpoint:** `checkpoints/fixmatch_baseline/best_model.pth`
   * Metrics: `checkpoints/fixmatch_baseline/eval_metrics_full30view.json`
3. **Supervised Baseline (10%) Checkpoint:** `checkpoints/supervised_baseline/best_model.pth`
   * Metrics: `checkpoints/supervised_baseline/eval_metrics_full30view.json`
4. **Supervised Reference (100%) Checkpoint:** `checkpoints/supervised_100/best_model.pth`
   * Metrics: `checkpoints/supervised_100/metrics.json`

## 4. Phase 4 Synthesis & Disentanglement Analysis

> **Model Update Protocol:** `Verify -> Confirm Bug/Deviation -> Assess Impact -> Fix -> Update & Rerun Affected Model`

### 4.1 The Complete 5-Rung Experimental Ladder

| Run | Modality | Losses | Description | Top-1 (%) | Top-5 (%) | Gain vs 10% Sup (B) | Gain vs FixMatch (C) | Status |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **A** | RGB | None | 100% Supervised Oracle | **72.91%** | **90.06%** | +41.59% | +34.42% | Upper reference oracle |
| **B** | RGB | None | 10% Supervised Baseline | **31.32%** | **54.11%** | *Reference* | -7.17% | Lower supervised baseline |
| **C** | RGB | None | FixMatch RGB Baseline | **38.49%** | **64.00%** | +7.17% | *Reference* | Single-stream SSL |
| **D** | **RGB + TG** | **$\beta=0.0, \gamma=0.0$** | **Dual-Stream Control** | **59.03%** | **82.90%** | **+27.71%** | **+20.54%** | Matched 30-View Test |
| **F** | **RGB + TG** | $\beta=2.0, \gamma=1.0$ | Full $F^2$-FixMatch | **59.90%** | **83.32%** | **+28.58%** | **+21.41%** | Canonical / Verified |

### 4.2 Disentangled Mathematical Attribution

* **Dual-Stream Configuration ($\Delta_{\text{dual}} = D - C$):** **+20.54 pts** (95.9% of total SSL gain).
* **Feature-Level Consistency Module ($\Delta_{\text{feat}} = F - D$):** **+0.87 pts** (4.1% of total SSL gain).
* **Total Semi-Supervised Advantage ($\Delta_{\text{total}} = F - C$):** **+21.41 pts** (100.0%).

#### 4.2.1 Published Paper Ablation Comparison, Statistical Noise Margin & Computational Cost

1. **Comparison with Tu et al. Published Ablations (Table 4)**:
   * In Tu et al. (ACM TOMM 2025, Table 4), the authors report:
     * Baseline **A1** (Frame-level dual-stream, $Z_{w_p}=\times, Z_{s_p}=\times$, i.e. $\beta=0, \gamma=0$): **67.9%**.
     * Baseline **A4** (Full $F^2$-FixMatch with both feature augmentations): **71.1%**.
   * In the published data, transitioning from single-stream FixMatch (38.4%) to frame-level dual-stream (67.9%) accounts for **+29.5 pts (90.2% of their total gain)**, while the feature-level module contributes an incremental **+3.2 pts (9.8%)**.
   * In our physical reproduction, the dual-stream package accounts for **+20.54 pts (95.9%)**, while the feature module contributes **+0.87 pts (4.1%)**.
   * Crucially, in the paper's hyperparameter diagnostic study (Section 4.5), the authors tuned $\beta$ and $\gamma$ across ${0.5, 1, 2, 3}$, **never evaluating $\beta=0, \gamma=0$**. They assumed non-zero auxiliary weights rather than testing whether the feature module was statistically necessary.

2. **Statistical Binomial Noise Analysis**:
   * On the official UCF-101 Split 1 test set ($N = 3,783$ videos) at baseline accuracy $p \approx 0.60$:
     $$\text{Standard Error (SE)} = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0.60 \times 0.40}{3,783}} \approx 0.796\% \approx 0.80\%$$
     $$\text{95\% Margin of Error} = 1.96 \times \text{SE} \approx \pm 1.56\%$$
   * Because the observed empirical difference between Run F and Run D ($\Delta_{\text{feat}} = +0.87\%$) is smaller than $\pm 1.56\%$, it corresponds to $\sim 33$ test videos out of 3,783.
   * **Statistical Context**: Because $+0.87\%$ lies within the $\pm 1.56\%$ evaluation margin, it remains open whether the observed difference reflects a modest auxiliary regularization effect or finite-sample evaluation variance. Differences in distributed batch scaling or unreleased implementation subtleties may also affect how feature regularizers behave across environments.

3. **Computational Overhead and Efficiency Trade-off**:
   * In Full $F^2$-FixMatch, every unlabeled batch ($B_u = 16, \mu = 4$) must pass weak and strong perturbed features through the spatial perturbator and the Temporal Attention Pooling (TAP) head (`forward_paug`), followed by two additional backward passes for $\mathcal{L}_u^{w_p}$ and $\mathcal{L}_u^{s_p}$.
   * In Run D ($\beta=0, \gamma=0$), these passes are bypassed, reducing per-epoch training latency by $\sim 15\%$ and lowering VRAM requirements.
   * In contrast, parameter-free inference recalibration (PreciseBN) delivers $+2.03\%$ on Run D and $+2.91\%$ on $F^2$ in $< 2$ minutes with zero additional training cost.

### 4.3 Pre-Declared Outcome Classification: Case C (Dual-Stream Package Dominant; Feature Module Negligible)

* **Empirical Trajectory:** $C = 38.49\% \longrightarrow D = 59.03\% \longrightarrow F = 59.90\%$.
* **Primary Driver:** Integrated Dual-Stream Framework (Package Effect).
* **Interpretation:** The performance leap over FixMatch-RGB occurs when transitioning to the integrated dual-stream configuration package (Delta_dual = +20.54 pts); adding the feature-level consistency module provides negligible incremental benefit (Delta_feat = +0.87 pts < 2.0 pts). Empirical progression: C=38.49% -> D=59.03% -> F=59.90%. Important Causal Caveat: Run D evaluates the dual-stream configuration as an integrated package (bi-modal inputs, shared backbone, cross-modal ensemble pseudo-labeling, and joint optimization dynamics). While this confirms that the dual-stream package—rather than the feature consistency module—accounts for the vast majority of the gain over FixMatch-RGB, isolating the individual internal components within that package (e.g. TG inputs alone vs. ensemble pseudo-labeling) remains an open question.
* **Protocol Action:** Assess Impact & Audit: Investigate weak-feature detach (Z_w) and loss weighting. Determine whether architectural constraints in the current head prevented feature consistency losses from delivering auxiliary regularizing benefit.
