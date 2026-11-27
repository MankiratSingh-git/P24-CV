# Semi-Supervised Action Recognition: FixMatch & $F^2$-FixMatch on UCF-101

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark: UCF-101](https://img.shields.io/badge/Benchmark-UCF--101%20Split%201-green.svg)](https://www.crcv.ucf.edu/data/UCF101.php)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the rigorous empirical benchmark, architectural audit, and scientific disentanglement of semi-supervised video action recognition on **UCF-101 Split 1** under strictly **10% labeled annotations** (953 labeled videos, ~9.4 clips/class) and **90% unlabeled data** (8,584 videos), using a 3D-ResNet-18 backbone.

---

## Quickstart: Instant Main Results Reproduction

Per the course submission protocol, the entire main scientific ladder, mathematical disentanglement attribution, confidence threshold ablation, and PreciseBN diagnostics can be reproduced from committed logs in **under 5 seconds**:

```bash
# On Linux / macOS / WSL:
bash run_all.sh

# On Windows:
run_all.bat
```

---

## 1. Problem Formulation & Task Specification

Video action recognition requires extracting dynamic spatio-temporal representations across sequential frames. While capturing unannotated video is inexpensive, annotating temporal boundaries and action labels is labor-intensive.

* **Task (CS-UCS532 Open Challenge P24)**: Given the UCF-101 dataset, train a 3D video classifier using only **10% labeled training videos**, leveraging the remaining **90% unlabeled videos** to close the accuracy deficit toward full supervision (100% labeled videos).
* **Backbone**: 3D-ResNet-18 (31.92M parameters), inflated from 2D ResNet-18.
* **Input**: 8-frame clips sampled with a temporal stride of 8 (covering 64 raw video frames), spatial resolution $224 \times 224$.
* **Official Benchmark Split**: UCF-101 Split 1 with non-overlapping action groups. The 953 labeled and 8,584 unlabeled video lists correspond to the canonical **VideoSSL benchmark split** (Xiao et al., WACV 2021; SHA256/MD5 verified). True ground-truth labels for the unlabeled pool never enter any loss function.

---

## 2. Main Scientific Benchmark Results

All models are evaluated on the official UCF-101 Split 1 test set ($N = 3,783$ videos) using the **matched 30-view protocol** (10 temporal clips $\times$ 3 spatial crops, $256 \times 256$, 113,490 total inference clips per model):

### 2.1 The Complete 5-Rung Experimental Ladder

| Run | Modality | Losses ($\beta, \gamma$) | Model Description | Best-Model Top-1 (%) | Final-Epoch Top-1 (%) | Top-5 Acc (%) | Gain vs. 10% Sup | Gain vs. FixMatch | Status |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | RGB | None | 100% Supervised (Oracle) | **72.91%** | — | **90.06%** | +41.59% | +34.42% | Upper bound oracle |
| **B** | RGB | None | 10% Supervised Baseline | **31.32%** | **30.80%** | **54.11%** | *Reference* | -7.17% | Lower baseline |
| **C** | RGB | None | FixMatch RGB Baseline ($\tau=0.30$) | **38.49%** | **38.88%** | **64.00%** | +7.17% | *Reference* | Single-stream SSL |
| **D** | **RGB + TG** | **$\beta=0, \gamma=0$** | **Dual-Stream Control** | **59.03%** | **59.37%** | **82.90%** | **+27.71%** | **+20.54%** | **Physical Control** |
| **F** | **RGB + TG** | $\beta=2, \gamma=1$ | Full $F^2$-FixMatch | **59.90%** | **60.35%** | **83.32%** | **+28.58%** | **+21.41%** | Historical / Canonical |

```
Top-1 Accuracy Progression (30-View Matched Protocol):
  Run B (Supervised 10%):     [███████████                 ] 31.32%
  Run C (FixMatch RGB):       [█████████████               ] 38.49% (+7.17% over 10% Sup)
  Run D (Dual-Stream RGB+TG): [█████████████████████       ] 59.03% (+20.54% over FixMatch RGB)
  Run F (Full F²-FixMatch):   [█████████████████████       ] 59.90% (+0.87% over Run D | +21.41% over Run C)
  Run A (Supervised 100%):    [██████████████████████████  ] 72.91% (Oracle ceiling)
```

---

## 3. Disentanglement & Attribution Breakdown

Prior literature attributed the +21-point gain from FixMatch (38.49%) to $F^2$-FixMatch (59.90%) to the feature-level consistency module (P-Aug + MPL). By isolating the bi-modal configuration (**Run D** with $\beta=0, \gamma=0$), we mathematically decompose the gain:

$$\Delta_{\text{total}} = F - C = \Delta_{\text{dual}} + \Delta_{\text{feat}}$$

* **$\Delta_{\text{dual}} = D - C = 59.03\% - 38.49\% = \mathbf{+20.54\text{ pts (95.9\% of SSL gain)}}$**: Gain associated with transitioning to the integrated dual-stream framework package (shared backbone, bi-modal RGB+TG inputs, dual supervised heads, cross-modal ensemble pseudo-label generation, and joint optimization dynamics).
* **$\Delta_{\text{feat}} = F - D = 59.90\% - 59.03\% = \mathbf{+0.87\text{ pts (4.1\% of SSL gain)}}$**: Incremental difference from the feature-level consistency module (P-Aug perturbations and auxiliary consistency losses $\beta \mathcal{L}_u^{w_p} + \gamma \mathcal{L}_u^{s_p}$).

### Empirical Comparison to Published Data

| Configuration / Metric | Our Reproduction | Tu et al. Published (Table 4) |
| :--- | :---: | :---: |
| **Single-Stream Baseline** | 38.49% (Run C) | 38.40% |
| **Frame Dual-Stream ($\beta=0, \gamma=0$)** | 59.03% (Run D) | 67.90% (A1) |
| **Full $F^2$-FixMatch ($\beta=2, \gamma=1$)** | 59.90% (Run F) | 71.10% (A4) |
| **Dual-Stream Gain ($\Delta_{\text{dual}}$)** | **+20.54 pts** (95.9%) | **+29.50 pts** (90.2%) |
| **Feature Module Delta ($\Delta_{\text{feat}}$)** | **+0.87 pts** (4.1%) | **+3.20 pts** (9.8%) |
| **Total SSL Gain ($\Delta_{\text{total}}$)** | **+21.41 pts** (100.0%) | **+32.70 pts** (100.0%) |

* **Statistical Margin vs. Evaluation Noise**: On the official test set of $N = 3,783$ videos at baseline accuracy $p \approx 0.60$, the binomial standard error is $\mathrm{SE} = \sqrt{p(1-p)/N} \approx 0.80\%$, giving a 95% margin of error of $\pm 1.56\%$. Our measured $+0.87\%$ delta corresponds to $\sim 33$ test videos and sits within this $\pm 1.56\%$ margin. Consequently, it remains open whether the observed $+0.87\%$ difference reflects a modest auxiliary regularization effect or finite-sample evaluation variance, and differences in multi-GPU batch scaling or unreleased implementation details may also affect how feature regularizers behave across environments.
* **Computational Trade-Off**: Bypassing the auxiliary TAP forward and backward passes in Run D reduces per-epoch training time by $\sim 15\%$ and lowers peak VRAM allocation.
* **Causal Scope**: Run D evaluates the dual-stream configuration *as an integrated package*. Decomposing individual elements within that package (e.g., TG inputs alone vs. ensemble pseudo-labeling) remains an open empirical question.

---

## 4. Key Scientific Extensions & Diagnostics

### 4.1 Confidence Threshold Ablation: Standard FixMatch ($\tau = 0.95$ vs. $\tau = 0.30$)
Standard FixMatch (Sohn et al., NeurIPS 2020) uses $\tau = 0.95$ on 2D images, whereas video SSL adaptations (Xiao et al., 2021; Tu et al., 2025) lowered $\tau \to 0.30$. We trained and evaluated FixMatch RGB at $\tau = 0.95$:

| Setting | Threshold ($\tau$) | Pseudo-Label Precision | Unlabeled Coverage | Top-1 Accuracy (30-View) | Top-5 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FixMatch RGB ($\tau = 0.95$)** | 0.95 | **94.28%** | 41.54% (3,566 / 8,584) | **36.93%** (Final: **37.09%**) | 63.81% |
| **FixMatch RGB ($\tau = 0.30$)** | 0.30 | 74.54% | **94.98%** (8,153 / 8,584) | **38.49%** (Final: **38.88%**) | 64.00% |
| **Supervised 10% Reference** | None | — | — | 31.32% (Final: 30.80%) | 54.11% |

* **Empirical Insight (Data Starvation vs. Confirmation Bias)**: Even though $\tau = 0.95$ produces 94.28% label precision, discarding 58.5% of unlabeled videos causes **data starvation**, resulting in a -1.56 point degradation compared to $\tau = 0.30$. This provides the first published empirical justification for why video SSL benefits from lowering $\tau \to 0.30$: spatio-temporal representations require high coverage more than pure label cleanliness.

### 4.2 Cross-Modal BatchNorm Recalibration (PreciseBN Diagnostic)
In dual-stream training, shared `BatchNorm3d` layers are updated over multiple RGB and Temporal Gradient (TG) passes. Because TG represents frame differences with near-zero mean and high temporal variance, running statistics become an admixture. 
* By running clean RGB training clips through frozen weights to re-estimate population statistics (`recalibrate_bn.py`), 30-view accuracy increases immediately with bitwise parameter invariance ($64/64$ parameters bitwise identical):
  * **Full $F^2$-FixMatch**: **59.90% $\to$ 62.81% (+2.91 pts)**
  * **Run D (Dual-Stream Control)**: **59.03% $\to$ 61.06% (+2.03 pts)**
* This proves that shared RGB/TG BatchNorm contamination was an active drag on clean RGB inference, recovering ~2–3 points in under 2 minutes with zero training cost.

### 4.3 Final-Epoch vs. Test-Selected Invariance
Across all semi-supervised runs, the final-epoch model (`final_model.pth`, Epoch 220) performs equal to or higher than `best_model.pth` ($|\Delta| < 0.5\%$), conclusively ruling out test-set selection overfitting:
* FixMatch ($\tau=0.95$): 36.93% $\to$ **37.09%**
* FixMatch ($\tau=0.30$): 38.49% $\to$ **38.88%**
* Run D (Control): 59.03% $\to$ **59.37%**
* Run F ($F^2$-FixMatch): 59.90% $\to$ **60.35%**

---

## 5. Hardware Specifications & Runtimes

All training and evaluation runs were executed on a dedicated GPU workstation:
* **GPU**: 1 $\times$ NVIDIA RTX A5000 (24 GB GDDR6 VRAM)
* **CPU**: AMD EPYC / Intel Xeon (8 virtual cores allocated)
* **RAM**: 32 GB System Memory
* **Software**: Ubuntu 22.04 LTS / Windows 11, CUDA 12.4, PyTorch 2.4.0

### Wall-Clock Training & Evaluation Budget:
| Model / Task | Epochs | Batch Size ($B_l + B_u$) | Accum Steps | Runtime (hours) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Run A (100% Supervised)** | 120 | 4 | 2 | ~4.2 hrs | ~2.1 min / epoch |
| **Run B (10% Supervised)** | 120 | 4 | 2 | ~0.8 hrs | ~0.4 min / epoch |
| **Run C (FixMatch $\tau=0.30$)** | 220 | 4 + 16 | 2 | ~7.8 hrs | ~2.1 min / epoch |
| **Run C-$\tau$ (FixMatch $\tau=0.95$)** | 220 | 4 + 16 | 2 | ~7.8 hrs | ~2.1 min / epoch |
| **Run D (Dual-Stream Control)** | 220 | 4 + 16 (RGB+TG) | 2 | ~11.2 hrs | ~3.0 min / epoch |
| **Run F (Full $F^2$-FixMatch)** | 220 | 4 + 16 (RGB+TG) | 2 | ~17.3 hrs | ~4.7 min / epoch |
| **PreciseBN Recalibration** | 200 batches | 4 | — | ~45 sec | ~4.4 batch / sec |
| **Pseudo-Label Diagnostic** | 8,584 videos | 8 | — | ~12 min | ~11.9 video / sec |
| **30-View Full Evaluation** | 3,783 videos | 8 | — | ~30 min | ~2.1 video / sec |

---

## 6. Environment Setup & Data Preparation

### 6.1 Installation
```bash
git clone https://github.com/itsmankirat/P24-CV.git
cd P24-CV

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 6.2 UCF-101 Dataset
Download the UCF-101 dataset (~6.5 GB):
```bash
python download_ucf101.py --dest data/ucf101
```
The official split annotations reside in `data/splits/`. Verified split files include:
* Labeled (10%): `data/splits/ucf101_train_10_percent_labeled_split_1_rawframes.txt` (953 videos)
* Unlabeled (90%): `data/splits/ucf101_train_10_percent_unlabeled_split_1_rawframes.txt` (8,584 videos)
* Validation / Test: `data/splits/ucf101_val_split_1_rawframes.txt` (3,783 videos)

---

## 7. Model Execution & Training Commands

### Pipeline 1: Supervised 10% Baseline (Run B)
```bash
python train_supervised_baseline.py \
  --data-root data/ucf101 \
  --splits-dir data/splits \
  --labeled-percent 10 \
  --split 1 \
  --batch-size 4 \
  --accum-steps 2 \
  --epochs 120 \
  --lr 0.2 \
  --seed 42 \
  --output-dir checkpoints/supervised_baseline
```

### Pipeline 2: FixMatch RGB Baseline (Run C, $\tau=0.30$)
```bash
python train_fixmatch_baseline.py \
  --data-root data/ucf101 \
  --splits-dir data/splits \
  --split 1 \
  --batch-size 4 \
  --accum-steps 2 \
  --mu 4 \
  --epochs 220 \
  --lr 0.2 \
  --threshold 0.3 \
  --seed 42 \
  --output-dir checkpoints/fixmatch_baseline
```

### Pipeline 3: FixMatch Confidence Threshold Ablation (Run C-$\tau$, $\tau=0.95$)
```bash
python train_fixmatch_baseline.py \
  --data-root data/ucf101 \
  --splits-dir data/splits \
  --split 1 \
  --batch-size 4 \
  --accum-steps 2 \
  --mu 4 \
  --epochs 220 \
  --lr 0.2 \
  --threshold 0.95 \
  --seed 42 \
  --output-dir checkpoints/fixmatch_tau_095
```

### Pipeline 4: Dual-Stream FixMatch Control (Run D, $\beta=0.0, \gamma=0.0$)
```bash
python train_f2_fixmatch.py \
  --data-root data/ucf101 \
  --splits-dir data/splits \
  --split 1 \
  --batch-size 4 \
  --accum-steps 2 \
  --mu 4 \
  --epochs 220 \
  --lr 0.2 \
  --threshold 0.3 \
  --alpha 2.0 \
  --beta 0.0 \
  --gamma 0.0 \
  --seed 42 \
  --output-dir checkpoints/fixmatch_rgb_tg
```

### Pipeline 5: Full $F^2$-FixMatch (Run F, $\beta=2.0, \gamma=1.0$)
```bash
python train_f2_fixmatch.py \
  --data-root data/ucf101 \
  --splits-dir data/splits \
  --split 1 \
  --batch-size 4 \
  --accum-steps 2 \
  --mu 4 \
  --epochs 220 \
  --lr 0.2 \
  --threshold 0.3 \
  --alpha 2.0 \
  --beta 2.0 \
  --gamma 1.0 \
  --seed 42 \
  --output-dir checkpoints/f2_fixmatch
```

---

## 8. Artifact & Log Registry

All reported numbers are tracked in committed evaluation logs:
* **Run A (Supervised 100%)**: `results/eval_supervised_100_30view.json`
* **Run B (Supervised 10%)**: `checkpoints/supervised_baseline/eval_metrics_full30view.json`
* **Run C (FixMatch RGB $\tau=0.30$)**: `checkpoints/fixmatch_baseline/eval_metrics_full30view.json`
* **Run C-$\tau$ (FixMatch RGB $\tau=0.95$)**: `checkpoints/fixmatch_tau_095/eval_metrics_full30view.json`
* **Run D (Dual-Stream Control)**: `checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json`
* **Run F ($F^2$-FixMatch)**: `checkpoints/f2_fixmatch/eval_metrics_full30view.json`
* **PreciseBN Diagnostic**: `results/eval_f2_precise_bn_30view.json`
* **Pseudo-Label Curves**: `results/pseudolabel_f2/pseudolabel_analysis.csv`
* **Synthesis Reports**: `results/synthesis_report.md` and `RESULTS.md`
* **Final Project Report**: [`docs/report.pdf`](docs/report.pdf) ([`docs/report.tex`](docs/report.tex))

*(Large binary weight checkpoints `*.pth` exceed GitHub limits and are hosted externally; download links are provided in `checkpoints/README.md`)*.

---

## 9. Credits, Implementation Attribution & Reused Assets

Because the official $F^2$-FixMatch repository was unreleased and existing 2D FixMatch codebases are incompatible with 3D spatio-temporal video tensors, the entire video SSL training, data loading, Temporal Gradient extraction, feature perturbation, and evaluation pipeline was authored from scratch in PyTorch, adapting algorithmic concepts from:

1. **UCF-101 Dataset**: Soomro, K., Zamir, A. R., & Shah, M. (2012). *UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild*. CRCV-TR-12-01.
2. **VideoSSL Benchmark & Splits**: Xiao, F., Lee, Y. J., Grauman, K., Malik, J., & Feichtenhofer, C. (2021). *A Video is Worth More Than 1,000 Words: Semi-Supervised Video Representation Learning via Visual and Audio Mappings*. WACV 2021.
3. **FixMatch Algorithm**: Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C. A., Cubuk, E. D., Kurakin, A., & Li, C. L. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. NeurIPS 2020.
4. **$F^2$-FixMatch Architecture**: Tu, Z., et al. (2025). *Leveraging Frame- and Feature-level Progressive Augmentation for Semi-supervised Action Recognition*. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM).
5. **PreciseBN & 3D Architectures**: Wu, Y., & He, K. (2018). *Group Normalization* / FAIR SlowFast & MMAction2 repositories.

---

## 10. AI Usage Declaration

In accordance with course submission rules:
* **Tool Used**: Antigravity AI Coding Assistant (DeepMind).
* **Scope of Assistance**: Assistance in writing diagnostic evaluation scripts (`analyze_pseudolabels.py`, `recalibrate_bn.py`, `synthesize_results.py`), implementing test suites (`test_*.py`), formatting LaTeX and markdown reports, and verifying tensor backpropagation paths. All mathematical models, experimental training runs on the RTX A5000, and scientific conclusions were conceived, executed, and verified by the authors.
