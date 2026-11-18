# Semi-Supervised Action Recognition: FixMatch & $F^2$-FixMatch on UCF-101

This repository reproduces and benchmarks semi-supervised video action recognition methods on **UCF-101** using strictly **10% labeled data** (953 videos) and **90% unlabeled data** (8,584 videos). The UCF-101 10% labeled split corresponds to the canonical VideoSSL benchmark split from Xiao et al. (WACV 2021). The existing 953-video labelled split is retained unchanged for comparability; its source/provenance and per-class labelled counts will be recorded (see [data/splits/split1_10percent_class_distribution.json](file:///c:/Code/P24-CV/data/splits/split1_10percent_class_distribution.json)). It implements and evaluates:
1. **FixMatch**: *"FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"* (Sohn et al., NeurIPS 2020) adapted to 3D video recognition.
2. **$F^2$-FixMatch**: *"Leveraging Frame- and Feature-level Progressive Augmentation for Semi-supervised Action Recognition"* (Tu et al., ACM TOMM 2025).

---

## 1. Benchmark Results (UCF-101 Split 1)

Evaluations are reported under both standard validation (1-crop) and the official paper testing protocol (10 uniform clips $\times$ 3 spatial crops = **30 views per video**, $256 \times 256$ resolution across all 3,783 validation videos, 113,490 total clips per model):

### 30-View Multi-Crop Performance (Official Testing Protocol)

| Method | Labels | Modality | Backbone | Training Strategy | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Absolute Gain vs. 10% Sup | Gain vs. FixMatch |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Supervised Baseline (10%)** | 10% | RGB | 3D-ResNet-18 | 10% labeled data only (CE loss) | **31.32%** | **54.11%** | *Reference* | — |
| **FixMatch Baseline** | 10% | RGB | 3D-ResNet-18 | Pseudo-labeling ($\tau=0.3$) + RandAug + Cutout | **38.49%** | **64.00%** | **+7.17%** | *Reference* |
| **$F^2$-FixMatch (Full Framework)** | 10% | **RGB + TG** | 3D-ResNet-18 | **Dual-Stream + Feature Module (P-Aug + MPL)** | **59.90%** | **83.32%** | **+28.58%** | **+21.41%** |
| **Supervised Reference (100%)** | 100% | RGB | 3D-ResNet-18 | Full dataset (9,537 videos, 120 ep) | *Pending 30-view* | *Pending 30-view* | — | — |

### Matched 1-Crop Validation Performance & Supervision Gap Closed

| Method | Labels | Modality | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Absolute Gain | Matched Gap Closed ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline (10%)** | 10% | RGB | **30.19%** | **53.69%** | *Reference* | 0.0% |
| **FixMatch Baseline** | 10% | RGB | **37.85%** | **63.12%** | **+7.66%** | **18.5%** |
| **$F^2$-FixMatch** | 10% | **RGB + TG** | **56.15%** | **81.05%** | **+25.96%** | **62.6%** |
| **Supervised Reference (100%)** | 100% | RGB | **71.69%** | **89.14%** | **+41.50%** | *Upper Bound* |

* For detailed benchmark discussions and documented hardware factors, see [RESULTS.md](RESULTS.md).
* For mathematical formulations, audit insights ($Z_w$ detachment, pooling invariance), and architecture specifications, see [METHODOLOGY.md](METHODOLOGY.md).

---

## 2. Key Empirical Insights

* **Supervision Gap Recovery**: Under matched 1-crop protocol, $F^2$-FixMatch closes **62.6%** of the performance deficit toward full 100% supervision using only one-tenth of the annotations ($+25.96\%$ absolute gain over 10% supervised).
* **Multi-Modal & Feature-Level Contribution**: Moving from single-stream RGB FixMatch (38.49% 30-view) to $F^2$-FixMatch (59.90% 30-view) delivers an absolute improvement of **+21.41%**, demonstrating the value of combining dynamic Temporal Gradients with the unified Feature-Level Consistency Module (P-Aug & MPL).
* **Reproduction Factor Transparency**: The empirical difference between our 59.90% reproduction and the paper's 71.1% reflects concrete execution factors: single-GPU training with per-GPU batch size 4 (gradient accumulation steps 2) and base learning rate $\eta = 0.20$. Further architectural and normalization factors are pre-declared for investigation in the evaluation audit.

---

## 3. Architecture: 3D-ResNet-18 (Table 2 in Paper)

- **Backbone**: 3D-ResNet-18 (31.92M parameters), inflated from 2D ResNet-18.
- **Input**: 8 frames per clip sampled with a stride of 8 (covering 64 raw frames), resolution $224 \times 224$.
- **Temporal Gradient (TG)**: $x_t^G = (x_t^V - x_{t+1}^V + 255) / 2$. A single shared encoder extracts representations for both RGB and TG streams.
- **Temporal Resolution**: No temporal pooling or temporal stride throughout the network; $T=8$ is preserved up to global pooling.
- **Decoupled Heads**:
  - $C_{3D}$: Evaluates frame-level representations and generates all pseudo-labels.
  - $C_{TAP}$: Evaluates feature-perturbed representations after Temporal Average Pooling.

---

## 4. Execution Guide

### 4.1 Environment Setup

Activate the virtual environment (or use your preferred Python environment):
```powershell
# In PowerShell:
.\.venv\Scripts\Activate.ps1

# Install dependencies:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Verify pipeline shapes, tensor dimensions, and backpropagation:
```powershell
python verify_pipeline.py
```

---

### 4.2 Dataset Setup

#### Option A: Full UCF-101 Dataset (~6.5 GB)
```powershell
python download_ucf101.py --dest data/ucf101
```

#### Option B: Synthetic Test Dataset (Instant pipeline smoke test)
```powershell
python download_ucf101.py --dest data/ucf101 --synthetic
```

---

### 4.3 Training Pipelines

#### Pipeline 0: Fully Supervised Reference (100% Labeled Data)
```powershell
python train_supervised_baseline.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --labeled-percent 100 `
  --split 1 `
  --batch-size 4 `
  --accum-steps 2 `
  --epochs 120 `
  --lr 0.2 `
  --output-dir checkpoints/supervised_100
```

#### Pipeline 1: Supervised Baseline (10% Labeled Data Only)
```powershell
python train_supervised_baseline.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --labeled-percent 10 `
  --split 1 `
  --batch-size 4 `
  --accum-steps 2 `
  --epochs 120 `
  --lr 0.2 `
  --output-dir checkpoints/supervised_baseline
```

#### Pipeline 2: FixMatch Baseline (10% Labeled + 90% Unlabeled RGB)
```powershell
python train_fixmatch_baseline.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --split 1 `
  --batch-size 4 `
  --accum-steps 2 `
  --mu 4 `
  --epochs 220 `
  --lr 0.2 `
  --threshold 0.3 `
  --lambda-u 2.0 `
  --rampup-epochs 30 `
  --warmup-epochs 40 `
  --output-dir checkpoints/fixmatch_baseline
```

#### Pipeline 3: $F^2$-FixMatch (Full Framework: RGB + TG + P-Aug + MPL)
```powershell
python train_f2_fixmatch.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --split 1 `
  --batch-size 4 `
  --accum-steps 2 `
  --mu 4 `
  --epochs 220 `
  --lr 0.2 `
  --threshold 0.3 `
  --alpha 2.0 `
  --beta 2.0 `
  --gamma 1.0 `
  --rampup-epochs 30 `
  --warmup-epochs 40 `
  --output-dir checkpoints/f2_fixmatch
```

#### Pipeline 4: Dual-Stream FixMatch Control (Run D: beta=0.0, gamma=0.0)
```powershell
python train_f2_fixmatch.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --split 1 `
  --batch-size 4 `
  --accum-steps 2 `
  --mu 4 `
  --epochs 220 `
  --lr 0.2 `
  --threshold 0.3 `
  --alpha 2.0 `
  --beta 0.0 `
  --gamma 0.0 `
  --rampup-epochs 30 `
  --warmup-epochs 40 `
  --output-dir checkpoints/fixmatch_rgb_tg
```

---

### 4.4 Evaluation Protocol (30-View Multi-Crop)

To evaluate any saved checkpoint under the official paper protocol (10 uniform temporal clips $\times$ 3 spatial crops = 30 views):
```powershell
python evaluate.py `
  --checkpoint checkpoints/f2_fixmatch/best_model.pth `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --eval-mode full `
  --crop-size 256
```

---

## 5. Artifact Registry

| Model | Checkpoint Path | Evaluation Metrics |
| :--- | :--- | :--- |
| **$F^2$-FixMatch** | `checkpoints/f2_fixmatch/best_model.pth` | `checkpoints/f2_fixmatch/eval_metrics_full30view.json` |
| **FixMatch Baseline** | `checkpoints/fixmatch_baseline/best_model.pth` | `checkpoints/fixmatch_baseline/eval_metrics_full30view.json` |
| **Supervised (10%)** | `checkpoints/supervised_baseline/best_model.pth` | `checkpoints/supervised_baseline/eval_metrics_full30view.json` |
| **Supervised (100%)** | `checkpoints/supervised_100/best_model.pth` | `checkpoints/supervised_100/metrics.json` |
