# Semi-Supervised Action Recognition: FixMatch & $F^2$-FixMatch on UCF-101

This repository reproduces and benchmarks semi-supervised video action recognition methods on **UCF-101** using **10% labeled data** (953 videos) and **90% unlabeled data** (8,584 videos). It implements and evaluates:
1. **FixMatch**: *"FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"* (Sohn et al., NeurIPS 2020) adapted to 3D video recognition.
2. **$F^2$-FixMatch**: *"Leveraging Frame- and Feature-level Progressive Augmentation for Semi-supervised Action Recognition"* (Tu et al., ACM TOMM 2025).

---

## 1. Benchmark Results (UCF-101 Split 1)

Evaluated strictly under the official paper protocol (Section 4.2): 10 uniform temporal clips $\times$ 3 spatial crops = **30 views per video** ($256 \times 256$ resolution, evaluated across all 3,783 validation videos, 113,490 total clips per model).

| Method | Labels | Modality | Backbone | Training Strategy | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Absolute Gain | Gap Closed ($R$) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Supervised Baseline (10%)** | 10% | RGB | 3D-ResNet-18 | 10% labeled data only (CE loss) | **31.32%** | **54.11%** | *Reference* | 0.0% |
| **FixMatch Baseline** | 10% | RGB | 3D-ResNet-18 | Pseudo-labeling ($\tau=0.3$) + RandAug + Cutout | **38.49%** | **64.00%** | **+7.17%** | **17.8%** |
| **$F^2$-FixMatch (Full Framework)** | 10% | **RGB + TG** | 3D-ResNet-18 | **P-Aug ($Z_w, Z_s, Z_{w_p}, Z_{s_p}$) + MPL** | **59.90%** | **83.32%** | **+28.58%** | **70.8%** |
| **Supervised Reference (100%)** | 100% | RGB | 3D-ResNet-18 | Full dataset (9,537 videos, 120 ep) | **71.69%** | **89.14%** | **+40.37%** | *Upper Bound* |

* For complete per-class breakdowns, failure analysis, and negative transfer cases, see [RESULTS.md](RESULTS.md).
* For detailed mathematical derivations, P-Aug formulation, and MMAction2 architecture, see [METHODOLOGY.md](METHODOLOGY.md).

---

## 2. Key Insights & Negative Analysis Summary

* **Supervision Gap Recovery**: $F^2$-FixMatch closes **70.8%** of the performance gap toward full 100% supervision using only one-tenth of the annotations.
* **Where the Model Excels ($\ge 90\%$ Top-1)**: Highly effective on categories with prominent macro-body motion and distinctive visual context (`CleanAndJerk` 100.0%, `BasketballDunk` 100.0%, `BenchPress` 97.9%, `Diving` 97.8%, `Surfing` 97.0%).
* **Where the Model Fails ($< 20\%$ Top-1)**: Struggles on subtle micro-movements inside static scenes (`Hammering` 0.0%, `Shotput` 0.0%, `Basketball` 2.9%, `BlowingCandles` 3.0%, `PizzaTossing` 6.1%).
* **Negative Transfer (13 Classes)**: In 13 categories, semi-supervised learning suffered from pseudo-label confirmation bias and underperformed 10% supervised training (e.g., `BreastStroke` dropped -35.7%, `CliffDiving` dropped -23.1%). See Section 3 of [RESULTS.md](RESULTS.md).

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
  --batch-size 8 `
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
  --batch-size 8 `
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
  --batch-size 8 `
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
  --batch-size 8 `
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
