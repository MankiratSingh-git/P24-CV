# Presentation Slide Outline (9 Slides — Concise, Table & Bullet Format)

> **Design Note**: Keep slides uncluttered ($\ge 24\text{pt}$ body text, max 4–5 short bullets or 1 compact table per slide). All numbers match [`docs/report.pdf`](../docs/report.pdf) and [`README.md`](../README.md).

---

## Slide 1: Title Slide
* **Title**: Disentangling & Auditing Semi-Supervised Action Recognition ($F^2$-FixMatch on UCF-101)
* **Course**: CS-UCS532: Computer Vision | **Problem ID**: P24
* **Team Members**:
  * **Mankirat Singh** (`1024160002`)
  * **Vanshika Garg** (`1024030092`)
* **Repository**: `github.com/MankiratSingh-git/P24-CV` (`bash run_all.sh` / `run_all.bat`)

---

## Slide 2: Problem Statement & Strategic Pivot
| Parameter | Specification |
| :--- | :--- |
| **Dataset** | UCF-101 Split 1 (101 action classes) |
| **Data Split** | 953 Labeled (10%) / 8,584 Unlabeled (90%) / 3,783 Test |
| **Backbone & Input** | 3D-ResNet-18 (31.92M params), 8 frames (stride 8, $224\times 224$) |
| **Evaluation Protocol** | Matched 30-view multi-crop ($10\text{ clips} \times 3\text{ crops}$, $256\times 256$) |

* **Why We Pivoted**: Initial proposal targeted standard FixMatch (38.49% Top-1); we pivoted to $F^2$-FixMatch (claimed 71.1%) to audit state-of-the-art claims.

---

## Slide 3: From-Scratch Implementation & Architecture
* **Why Authored From Scratch**: Official $F^2$-FixMatch repo was empty; 2D FixMatch codebases are incompatible with 5D video tensors ($B \times C \times T \times H \times W$).

| Authored From Scratch (PyTorch) | Reused / Adapted Concepts |
| :--- | :--- |
| Video loaders, 30-view evaluator, online TG extractor | Weak/strong augmentation definitions (VideoSSL) |
| Shared 3D-ResNet-18 + $C_{3D}$ & $C_{\text{TAP}}$ heads | Confidence-masked pseudo-labeling logic (FixMatch) |
| P-Aug perturbators, multi-term loss, Run D control, PreciseBN | Official UCF-101 Split 1 train/val lists |

* **Pipeline**: `RGB + TG` $\to$ `Shared 3D-ResNet-18` $\to$ `Ensemble Pseudo-Label (tau=0.30)` $\to$ `Frame Loss` $+$ `TAP Feature Loss (Bypassed in Run D)`.

---

## Slide 4: The 5-Rung Experimental Ladder (30-View Protocol)
| Run | Modality | Weights ($\beta,\gamma$) | Model Description | Best Top-1 | Final Top-1 | Top-5 | $\Delta$ vs. FixMatch |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **A** | RGB | — | 100% Supervised Oracle | **72.91%** | — | 90.06% | +34.42% |
| **B** | RGB | — | 10% Supervised Baseline | 31.32% | 30.80% | 54.11% | -7.17% |
| **C** | RGB | $\tau=0.30$ | FixMatch RGB Baseline | 38.49% | 38.88% | 64.00% | *Reference* |
| **D** | **RGB+TG** | **$\beta=0, \gamma=0$** | **Dual-Stream Control** | **59.03%** | **59.37%** | **82.90%** | **+20.54%** |
| **F** | **RGB+TG** | $\beta=2, \gamma=1$ | Full $F^2$-FixMatch | **59.90%** | **60.35%** | **83.32%** | **+21.41%** |

* **No Overfitting**: Final-epoch checkpoints match or exceed best-epoch checkpoints ($|\Delta| < 0.5\%$).

---

## Slide 5: Core Ablation — Empirical Comparison (Run D vs. Full $F^2$)
| Configuration / Metric | Our Reproduction | Tu et al. Published (Table 4) |
| :--- | :---: | :---: |
| Single-Stream Baseline | 38.49% (Run C) | 38.40% |
| Frame Dual-Stream ($\beta=0, \gamma=0$) | 59.03% (Run D) | 67.90% (A1) |
| Full $F^2$-FixMatch ($\beta=2, \gamma=1$) | 59.90% (Run F) | 71.10% (A4) |
| **Dual-Stream Gain ($\Delta_{\text{dual}}$)** | **+20.54 pts (95.9%)** | **+29.50 pts (90.2%)** |
| **Feature Module Delta ($\Delta_{\text{feat}}$)** | **+0.87 pts (4.1%)** | **+3.20 pts (9.8%)** |

* **Statistical Context**: On $N=3{,}783$ test videos, 95% binomial margin is $\pm 1.56\%$ ($+0.87\% \approx 33$ videos).
* **Compute Trade-Off**: Bypassing auxiliary TAP passes in Run D cuts training time by $\sim 15\%$ and lowers VRAM.

---

## Slide 6: Novel Extension 1 — Confidence Threshold ($\tau = 0.95$ vs. $0.30$)
| Setting | Precision | Unlabeled Coverage ($N=8{,}584$) | Top-1 (30-View) | Top-5 |
| :--- | :---: | :---: | :---: | :---: |
| **FixMatch RGB ($\tau = 0.95$)** | **94.28%** | 41.54% (3,566 videos) | **36.93%** (Final: 37.09%) | 63.81% |
| **FixMatch RGB ($\tau = 0.30$)** | 74.54% | **94.98%** (8,153 videos) | **38.49%** (Final: 38.88%) | 64.00% |
| **Supervised 10% Ref** | — | — | 31.32% (Final: 30.80%) | 54.11% |

* **Data Starvation**: Strict thresholding ($\tau = 0.95$) discards 58.5% of unlabeled videos, degrading Top-1 by **-1.56 pts**.

---

## Slide 7: Novel Extension 2 — PreciseBN Recalibration
* **Issue**: Shared `BatchNorm3d` layers drift across RGB and near-zero-mean Temporal Gradient passes.
* **Method**: Freeze weights and re-estimate BN statistics on clean RGB clips (`recalibrate_bn.py`, $< 2\text{ min}$).

| Model | Pre-Recalibration | Post-PreciseBN | Gain | Weight Changes |
| :--- | :---: | :---: | :---: | :---: |
| **Full $F^2$-FixMatch (Run F)** | 59.90% | **62.81%** | **+2.91 pts** | 0 (64/64 identical) |
| **Dual-Stream Control (Run D)** | 59.03% | **61.06%** | **+2.03 pts** | 0 (64/64 identical) |

---

## Slide 8: Limitations
1. **Residual Performance Gap**: An ~8-pt gap remains to the paper's 71.1% (multi-GPU distributed batch scaling & unreleased training details).
2. **Single-Seed Budget**: Executed under a single random seed (`seed=42`) due to ~100 GPU-hour compute budget on RTX A5000.
3. **Feature Module Margin vs. Noise**: Measured $+0.87\%$ gain sits within the $\pm 1.56\%$ binomial margin, leaving open whether it reflects modest regularization or finite-sample variance.

---

## Slide 9: Deliverables & Declaration
* **One-Command Reproduction**: `bash run_all.sh` / `run_all.bat` ($< 3\text{ sec}$, 57/57 unit tests passing).
* **Report & Code**: `docs/report.pdf` | `github.com/MankiratSingh-git/P24-CV`
* **Authors**: **Mankirat Singh (1024160002)** & **Vanshika Garg (1024030092)**.
* **AI Declaration**: Antigravity AI (DeepMind) assisted with diagnostic scripts, unit tests, formatting, and linting; all GPU runs and analyses were executed by the authors.
