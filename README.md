# Semi-Supervised Action Recognition: FixMatch & F2-FixMatch on UCF-101

This repository reproduces and benchmarks semi-supervised action recognition methods on the **UCF-101** dataset using **10% labeled data** and **90% unlabeled data**, as presented in:
1. **FixMatch**: *"FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"* (Sohn et al., NeurIPS 2020)
2. **F2-FixMatch**: *"Leveraging Frame- and Feature-level Progressive Augmentation for Semi-supervised Action Recognition"* (Tu et al., ACM TOMM 2025)

---

## 📊 Benchmark Targets (UCF-101, 10% Labeled Data, Split 1)

| Method | Modality | Backbone | Frames | Features / Tricks | Paper Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline** | RGB | 3D-ResNet-18 | 8 | 10% labeled data only | ~24.7% – 28.5% |
| **FixMatch Baseline** | RGB | 3D-ResNet-18 | 8 | Pseudo-labeling ($\tau=0.3$) + RandAugment + Cutout | **51.7%** (Table 5) |
| **F2-FixMatch (A1)** | RGB + TG | 3D-ResNet-18 | 8 | Bi-modal Frame-level W/S-Aug ($Z_w, Z_s$) | **67.9%** (Table 4) |
| **F2-FixMatch (Full)** | RGB + TG | 3D-ResNet-18 | 8 | P-Aug ($Z_w, Z_s, Z_{w_p}, Z_{s_p}$) + MPL | **71.1%** (Table 3) |

---

## 🛠️ Architecture: 3D-ResNet-18 (Table 2 in Paper)

- **Input**: 8 frames per clip sampled with stride of 8 (covering 64 frames), resolution $224 \times 224$.
- **conv1**: $1 \times 7 \times 7$ kernel, 64 channels, spatial stride 2 (temporal stride 1).
- **pool1**: $1 \times 3 \times 3$ max pool, spatial stride 2 (temporal stride 1).
- **res2**: 2 blocks, $1 \times 3 \times 3$ 2D spatial kernels, 64 channels.
- **res3**: 2 blocks, $1 \times 3 \times 3$ 2D spatial kernels, 128 channels, spatial stride 2.
- **res4**: 2 blocks, $3 \times 3 \times 3$ inflated 3D kernels, 256 channels, spatial stride 2.
- **res5**: 2 blocks, $3 \times 3 \times 3$ inflated 3D kernels, 512 channels, spatial stride 2.
- **Classifier Head**: `AdaptiveAvgPool3d((1, 1, 1)) -> Dropout(0.5) -> Linear(512, 101)`.
- *Key Property*: No temporal downsampling throughout the network; temporal dimension $T=8$ is preserved.
- *Temporal Gradient (TG)*: $x_t^G = (x_t^V - x_{t+1}^V + 255) / 2$. A single shared encoder extracts representations for both RGB and TG modalities.

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup

Activate the virtual environment (or use your preferred Python environment):
```powershell
# In PowerShell:
.\.venv\Scripts\Activate.ps1

# Install requirements (if not already installed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 2. Verify Pipeline & Architecture

Run the built-in verification suite to validate shapes, augmentations, and backprop:
```powershell
python verify_pipeline.py
```

### 3. Download / Prepare UCF-101 Dataset

#### Option A: Full Official Dataset (HuggingFace CDN Mirror, ~6.5 GB)
```powershell
python download_ucf101.py --dest data/ucf101
```

#### Option B: Synthetic Test Dataset (Instant, for smoke testing and pipeline validation)
```powershell
python download_ucf101.py --dest data/ucf101 --synthetic
```

---

### 4. Training Pipelines

#### Pipeline 1: Supervised Baseline (10% Labeled Data Only)
Trains the 3D-ResNet-18 backbone strictly on the 10% labeled split without unlabeled data:
```powershell
python train_supervised_baseline.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --labeled-percent 10 `
  --split 1 `
  --batch-size 8 `
  --epochs 120 `
  --lr 0.2 `
  --output-dir checkpoints/supervised_baseline
```

#### Pipeline 2: FixMatch Baseline (RGB Only)
Trains standard FixMatch with weak/strong augmentation consistency on UCF-101 (Paper Table 5 baseline: **51.7%**):
```powershell
python train_fixmatch_baseline.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --labeled-percent 10 `
  --split 1 `
  --batch-size 4 `
  --mu 4 `
  --threshold 0.3 `
  --warmup-epochs 40 `
  --epochs 220 `
  --lr 0.2 `
  --output-dir checkpoints/fixmatch_baseline
```

#### Pipeline 3: F2-FixMatch (Full Framework: P-Aug + MPL)
Trains the complete F2-FixMatch model with bi-modal inputs (RGB + TG), progressive frame- and feature-level perturbations, and multihead pseudo-labeling (Paper Table 3: **71.1%**):
```powershell
python train_f2_fixmatch.py `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --labeled-percent 10 `
  --split 1 `
  --batch-size 4 `
  --mu 4 `
  --threshold 0.3 `
  --alpha 2.0 `
  --beta 2.0 `
  --gamma 1.0 `
  --warmup-epochs 40 `
  --epochs 220 `
  --lr 0.2 `
  --output-dir checkpoints/f2_fixmatch
```

---

### 5. Evaluation Protocol

Evaluate any trained checkpoint using the official testing protocol (10 clips $\times$ 3 spatial crops = 30 views per video, RGB modality only):
```powershell
# Full 10-clip x 3-crop paper protocol:
python evaluate.py `
  --checkpoint checkpoints/fixmatch_baseline/best_model.pth `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --split 1 `
  --eval-mode full

# Fast 1-clip center-crop evaluation:
python evaluate.py `
  --checkpoint checkpoints/fixmatch_baseline/best_model.pth `
  --data-root data/ucf101 `
  --splits-dir data/splits `
  --split 1 `
  --eval-mode single
```
