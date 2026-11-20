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
| **Supervised Reference (100%)** | 100% | RGB | Full Training Set (9,537 videos, 120 ep) | *Pending 30-view audit* | *Pending 30-view audit* | — | — | Phase 2 Audit |

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
| **Supervised Reference (100%)** | ~70% – 73% | **71.69%** (1-crop) | Consistent with standard 3D-ResNet-18 supervised performance on UCF-101 |
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

### 4.3 Pre-Declared Outcome Classification: Case C (Dual-Stream Configuration Dominant; Feature Module Negligible)

* **Empirical Trajectory:** $C = 38.49\% \longrightarrow D = 59.03\% \longrightarrow F = 59.90\%$.
* **Primary Driver:** RGB+TG Dual-Stream Configuration.
* **Interpretation:** Most of the observed gain is attributable to the dual-stream configuration (Delta_dual = +20.54 pts); the feature-level module adds negligible benefit (Delta_feat = +0.87 pts < 2.0 pts). Empirical progression: C=38.49% -> D=59.03% -> F=59.90%.
* **Protocol Action:** Assess Impact & Audit: Investigate weak-feature detach (Z_w) and loss weighting. Determine whether architectural constraints in the current head prevented feature consistency losses from delivering auxiliary regularizing benefit.
