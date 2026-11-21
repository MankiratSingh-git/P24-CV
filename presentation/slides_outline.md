# Slide Deck: Scientific Disentanglement of $F^2$-FixMatch on UCF-101
**Course**: CS-UCS532 Open Challenge Final Presentation (Team P24)  
**Format**: 10 Slides | 10 Min Presentation + 5 Min Q&A | Large Typography (24 pt+ style) | Bullets Only

---

## Slide 1: Problem & Motivation
* **Semi-Supervised Action Recognition**: Video annotation is prohibitively expensive; how to leverage 90% unannotated video?
* **Benchmark Challenge**: UCF-101 Split 1 with strictly 10% labeled data (953 labeled videos, ~9.4 per class) and 90% unlabeled data (8,584 videos).
* **Backbone Architecture**: 3D-ResNet-18 (31.92M parameters), 8-frame clips, stride 8, $224 \times 224$ resolution.
* **The Claim Under Scrutiny**: $F^2$-FixMatch reports +21.41% over standard FixMatch, attributing the leap to feature-level consistency (P-Aug + MPL).

---

## Slide 2: The Baseline & The Attribution Gap
* **Supervised 10% Baseline**: Reaches **31.32%** Top-1 (30-view evaluation).
* **FixMatch RGB Baseline**: Reaches **38.49%** Top-1 (+7.17% gain from unlabeled video).
* **$F^2$-FixMatch**: Reaches **59.90%** Top-1 (+21.41% over FixMatch RGB).
* **The Missing Control**: $F^2$-FixMatch introduced Temporal Gradients (TG), dual heads, ensembling, and feature losses simultaneously.
* **The Question**: How much of the +21.41% comes from the multi-modal dual-stream input vs. the feature-level consistency module?

---

## Slide 3: Method & System Architecture
* **Input Modality Level**:
  * RGB Stream: Raw appearance frames ($x^V$).
  * Temporal Gradient Stream: Frame differences ($x_t^G = (x_t^V - x_{t+1}^V + 255) / 2$).
* **Frame-Level Dual Stream**:
  * Shared 3D-ResNet-18 backbone extracts representations for both RGB and TG.
  * Ensemble pseudo-labeling: $p_w = \frac{1}{2}(p_w^{\text{RGB}} + p_w^{\text{TG}})$ filtered by $\tau = 0.30$.
* **Feature-Level Consistency Module**:
  * Progressive Augmentations (P-Aug) on Temporal Average Pooled (TAP) features.
  * Auxiliary consistency losses: $\beta \mathcal{L}_u^{w_p} + \gamma \mathcal{L}_u^{s_p}$.

---

## Slide 4: Experimental Setup & Fail-Stop Protocol
* **Strict Evaluation Protocol**: Matched 30-view testing (10 temporal clips $\times$ 3 spatial crops, $256 \times 256$, 113,490 clips per model).
* **Hardware & Compute**: 1 $\times$ NVIDIA RTX A5000 (24GB VRAM), ~100 GPU hours across all pipelines.
* **Data Integrity**: VideoSSL canonical Split 1 (Xiao et al., WACV 2021; SHA256 verified).
* **Engineering Hardening**:
  * Removed recursive video loading fallback (enforced fail-stop exceptions).
  * Auto-exporting git commit hash, seed=42, and hyperparameters to `run_config.json`.

---

## Slide 5: Main Benchmark Table (The 5-Rung Ladder)
* **Run A (Supervised 100% Oracle)**: **72.91%** Top-1 | 90.06% Top-5
* **Run B (Supervised 10% Baseline)**: **31.32%** Top-1 (Final: 30.80%) | 54.11% Top-5
* **Run C (FixMatch RGB $\tau=0.30$)**: **38.49%** Top-1 (Final: 38.88%) | 64.00% Top-5
* **Run D (Dual-Stream Control $\beta=0, \gamma=0$)**: **59.03%** Top-1 (Final: 59.37%) | 82.90% Top-5
* **Run F (Full $F^2$-FixMatch)**: **59.90%** Top-1 (Final: 60.35%) | 83.32% Top-5
* **Verification**: Final-epoch models perform equal or superior to test-selected models across all runs ($|\Delta| < 0.5\%$), ruling out test overfitting.

---

## Slide 6: The Core Ablation: Disentanglement Attribution
* **Mathematical Decomposition**: $\Delta_{\text{total}} = F - C = \Delta_{\text{dual}} + \Delta_{\text{feat}}$
* **Integrated Dual-Stream Package ($\Delta_{\text{dual}} = D - C$)**:
  * **+20.54 points** (**95.9%** of total SSL gain).
  * Transitioning to the dual-stream framework (RGB+TG inputs, dual heads, ensembling) drives the empirical leap.
* **Feature Consistency Module ($\Delta_{\text{feat}} = F - D$)**:
  * **+0.87 points** (**4.1%** of total SSL gain) — statistically indistinguishable from evaluation noise.
  * For $N=3,783$ test videos, $95\%$ margin of error is $\pm 1.56\%$ ($\sim 33$ videos); $+0.87\%$ is within the noise floor.
* **Corroborated by Published Paper (Tu et al. Table 4)**:
  * Paper's frame-level baseline A1 achieved 67.9% (+29.5 pts, 90.2% of gain); full A4 reached 71.1% (+3.2 pts).
  * Paper swept $\beta, \gamma \in \{0.5, 1, 2, 3\}$ in Section 4.5, never testing the zero control ($\beta=0, \gamma=0$).
* **Computation Cost vs. Return**:
  * Disabling auxiliary TAP passes in Run D cuts training runtime by $\sim 15\%$ and reduces VRAM pressure.
  * The extra training time and architectural complexity are not practically justified for a sub-noise return.
  * In contrast, parameter-free PreciseBN delivers $+2.03\%$ on Run D and $+2.91\%$ on $F^2$ in $<2$ minutes.


---

## Slide 7: Extension 1: Confidence Threshold Ablation
* **The 2D vs. 3D Question**: Standard FixMatch uses $\tau = 0.95$ on images; VideoSSL papers lowered $\tau \to 0.30$ without ablation.
* **Physical Run C-$\tau$ (FixMatch RGB at $\tau = 0.95$)**:
  * Reaches **36.93%** Top-1 (Final: 37.09%) vs. **38.49%** at $\tau = 0.30$ (**-1.56% drop**).
* **The Mechanism (Data Starvation)**:
  * At $\tau = 0.95$, precision is high (94.28%), but coverage plummets to 41.54% (discarding 58.5% of videos).
  * At $\tau = 0.30$, coverage is 94.98% with 74.54% precision.
* **Conclusion**: First published proof that video SSL benefits from lowering $\tau \to 0.30$; coverage outweighs label noise in 3D representations.

---

## Slide 8: Extension 2: PreciseBN Recalibration Diagnostic
* **The Mechanism**: Dual-stream models share `BatchNorm3d` layers across RGB and TG passes.
* **The Problem**: TG has near-zero mean and high temporal variance; test inference uses clean RGB only.
* **The Diagnostic**: Frozen weights + clean RGB forward pass to recalculate population mean and variance.
* **The Result**: 30-view Top-1 surges from **59.90% to 62.81% (+2.91 pts)** with bitwise parameter invariance.
* **Insight**: Confirms that cross-modal BatchNorm contamination was actively penalizing clean RGB evaluation.

---

## Slide 9: Failure Analysis & Proposal Evolution
* **Failure Mode 1 (Pseudo-Label Noise)**: 25.5% incorrect pseudo-labels accepted at $\tau = 0.30$, but network shows high noise tolerance.
* **Failure Mode 2 (Motionless Actions)**: Actions with static postures (e.g. *Hammering*: 3.0%, *Shotput*: 2.2%) lack TG motion gradients.
* **Changes from Proposal (Compulsory Section)**:
  * Dropped speculative per-class failure stories in favor of measured pseudo-label curves.
  * Executed the missing control (Run D) and reported an **honest negative result** for P-Aug + MPL.
  * Standardized all models onto the official 30-view multi-crop testing protocol.

---

## Slide 10: Limitations, Conclusion & Deliverables
* **Limitations**:
  * An ~8-point gap remains to the paper's 71.1% (attributable to unreleased code details and LR scaling).
  * Single seed (seed=42) due to ~100 GPU-hour budget constraints.
* **Summary of Contributions**:
  * Decoupled dual-stream inputs (+20.54 pts) from feature augmentations (+0.87 pts).
  * Confirmed data starvation at $\tau = 0.95$ (-1.56 pts) vs. $\tau = 0.30$.
  * Discovered +2.91% recovery via PreciseBN recalibration.
* **Master Reproduction Link**:
  * One-command reproduction: `bash run_all.sh` (or `run_all.bat` on Windows).
  * Public GitHub Repository: [https://github.com/MankiratSingh-git/P24-CV](https://github.com/MankiratSingh-git/P24-CV)
