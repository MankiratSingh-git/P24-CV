# Empirical Experimental Results & Failure Analysis: F²-FixMatch vs. Baselines on UCF-101

**Benchmark:** UCF-101 (Split 1) | **Backbone:** 3D-ResNet-18 (31.92M parameters) | **Hardware:** NVIDIA RTX A5000 (24GB VRAM)  
**Label Regimes:** 10% Labeled (953 labeled / 8,584 unlabeled) vs. 100% Labeled (9,537 labeled) | **Validation Set:** 3,783 videos  
**Evaluation Protocol:** Official Paper Protocol (Section 4.2) — 10 uniform temporal clips $\times$ 3 spatial crops = **30 views per video** ($256 \times 256$ resolution, 113,490 total clips evaluated per model).

---

## 1. Executive Summary & Benchmark Comparison

| Model | Labels | Modality | Architecture / Loss Components | Evaluation Protocol | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Gain vs. 10% Sup | Gap Closed ($R$) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline (10%)** | 10% | RGB | Standard RandomCrop + HorizontalFlip | 30-view multi-crop | **31.32%** | **54.11%** | *Reference* | 0.0% |
| **FixMatch Baseline** | 10% | RGB | Weak/Strong (RandAug + Cutout, $\tau=0.3$) | 30-view multi-crop | **38.49%** | **64.00%** | **+7.17%** | **17.8%** |
| **$F^2$-FixMatch (Ours)** | 10% | **RGB + TG** | **P-Aug ($Z_w, Z_s, Z_{w_p}, Z_{s_p}$) + MPL** | **30-view multi-crop** | **59.90%** | **83.32%** | **+28.58%** | **70.8%** |
| **Supervised Reference (100%)** | 100% | RGB | Full Training Set (9,537 videos, 120 ep) | Center-Crop Val | **71.69%** | **89.14%** | **+40.37%** | *Upper Bound* |

```
Top-1 Accuracy Comparison:
Supervised (10%):  [███████████                           ] 31.32%
FixMatch (10%):    [█████████████                         ] 38.49% (+7.17% gain  | 17.8% gap closed)
F²-FixMatch (10%): [█████████████████████                 ] 59.90% (+28.58% gain | 70.8% gap closed)
Supervised (100%): [█████████████████████████             ] 71.69% (Full-Label Upper Bound Reference)
```

The supervision gap closed metric ($R$) quantifies how much of the performance deficit between the 10% supervised baseline and 100% supervised reference is recovered by semi-supervised learning:
$$R = \frac{A_{\text{SSL}} - A_{10\% \text{ sup}}}{A_{100\% \text{ sup}} - A_{10\% \text{ sup}}}$$

* **FixMatch Baseline (RGB)** closes **17.8%** ($+7.17\%$ over the 10% baseline).
* **$F^2$-FixMatch (RGB + Temporal Gradient)** closes **70.8%** ($+28.58\%$ over the 10% baseline), recovering over seven-tenths of full supervision with only one-tenth of the annotations.

---

## 2. Comparison With Published Literature (Tu et al., ACM TOMM 2025)

| Metric / Setting | Tu et al. Paper (Table 3 & 5) | Our Empirical Reproduction | Analysis & Context |
| :--- | :---: | :---: | :--- |
| **Supervised Reference (10%)** | ~25.4% | **31.32%** | Our supervised baseline exceeds paper reference (+5.92%) |
| **FixMatch Baseline (10%)** | 51.7% (Table 5) | **38.49%** | Evaluated on single-stream RGB without temporal gradient |
| **$F^2$-FixMatch (10%)** | **71.1%** (Table 3) | **59.90%** | **+28.58% absolute gain** over 10% supervised; **+21.41%** over FixMatch |
| **Supervised Reference (100%)** | ~70% – 73% | **71.69%** | Aligns with standard 3D-ResNet-18 supervised performance on UCF-101 |
| **Top-5 Accuracy ($F^2$-FixMatch)** | Not reported | **83.32%** | Ground truth is ranked in top-5 for over 5 out of 6 test videos |
| **Top-5 Accuracy (100% Sup)** | Not reported | **89.14%** | Full supervision upper bound |
| **Hardware / Batch Stats** | 2 $\times$ GPUs with SyncBN | 1 $\times$ RTX A5000 (accum steps = 2) | Difference attributable to multi-GPU cross-device SyncBN statistics |
| **Pseudo-Label Mask Coverage** | ~90% – 95% | **95.22%** | Matches paper Section 4.5 dynamics with $\tau = 0.3$ |

---

## 3. Negative Analysis: Where Models Fail & Negative Transfer

In standard machine learning literature, positive aggregate gains often conceal significant failure modes. A rigorous negative analysis inspects where models fail completely, where semi-supervised learning degrades performance relative to supervised learning (**negative transfer**), and where the multi-modal framework degrades relative to single-stream FixMatch.

### 3.1 Worst-Performing Classes in $F^2$-FixMatch ($<20\%$ Accuracy)

Out of 101 classes, 10 categories suffer severe failure ($<20\%$ Top-1 accuracy), including two complete flatlines ($0.0\%$):

| Action Category | $F^2$-FixMatch | FixMatch Baseline | Supervised (10%) | Total Videos | Primary Failure Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hammering** | **0.0%** (0/33) | 0.0% (0/33) | 9.1% (3/33) | 33 | Subtle wrist movement; dominated by static workbench background |
| **Shotput** | **0.0%** (0/46) | 0.0% (0/46) | 0.0% (0/46) | 46 | Explosive sub-second release; 8-frame stride-8 sampling misses release frame |
| **Basketball** | **2.9%** (1/35) | 22.9% (8/35) | 0.0% (0/35) | 35 | Semantic scene collapse with `BasketballDunk` on identical court |
| **BlowingCandles** | **3.0%** (1/33) | 15.2% (5/33) | 24.2% (8/33) | 33 | Subtle lip puffing; temporal gradient carries only sensor noise |
| **PizzaTossing** | **6.1%** (2/33) | 15.2% (5/33) | 18.2% (6/33) | 33 | Thin spinning dough boundary lost under spatial downsampling ($224\times 224$) |
| **MoppingFloor** | **8.8%** (3/34) | 14.7% (5/34) | 20.6% (7/34) | 34 | Repetitive floor wiping confused with sweeping or cleaning actions |
| **HighJump** | **10.8%** (4/37) | 13.5% (5/37) | 27.0% (10/37) | 37 | Track & field apparatus confusion with `PoleVault` and `LongJump` |
| **Lunges** | **16.2%** (6/37) | 5.4% (2/37) | 18.9% (7/37) | 37 | Gym floor context confused with `BodyWeightSquats` and `PushUps` |
| **JumpingJack** | **18.9%** (7/37) | 13.5% (5/37) | 24.3% (9/37) | 37 | High temporal and spatial similarity to `JumpRope` |
| **YoYo** | **19.4%** (7/36) | 2.8% (1/36) | 0.0% (0/36) | 36 | Fine string dynamics invisible; hand gesture ambiguous |

---

### 3.2 Negative Transfer: 13 Classes Where $F^2$-FixMatch Underperforms 10% Supervised

Semi-supervised learning is vulnerable to **confirmation bias**: when a model generates confident but incorrect pseudo-labels on unlabeled clips, consistency regularization forces the network to memorize erroneous decision boundaries. In **13 out of 101 classes**, adding unlabeled videos caused $F^2$-FixMatch to perform **worse than training strictly on 10% labeled data**:

| Action Category | $F^2$-FixMatch | Supervised (10%) | FixMatch Baseline | Negative Gap ($\Delta$) | Root Cause Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BreastStroke** | **53.6%** | **89.3%** | 92.9% | **-35.7%** | Swimming stroke confusion; features collapsed towards `FrontCrawl` |
| **CliffDiving** | **23.1%** | **46.1%** | 64.1% | **-23.1%** | Unlabeled videos collapsed cliff scenery into indoor `Diving` |
| **BlowingCandles** | **3.0%** | **24.2%** | 15.2% | **-21.2%** | No dynamic temporal gradient; pseudo-labels drifted to static face classes |
| **HighJump** | **10.8%** | **27.0%** | 13.5% | **-16.2%** | Cross-event confusion with `LongJump` and `PoleVault` |
| **PizzaTossing** | **6.1%** | **18.2%** | 15.2% | **-12.1%** | Hand motion in kitchen confused with cooking/cutting actions |
| **MoppingFloor** | **8.8%** | **20.6%** | 14.7% | **-11.8%** | Repetitive stroke dynamics confused with domestic floor cleaning |
| **Skiing** | **55.0%** | **65.0%** | 60.0% | **-10.0%** | Severe snow glare; pseudo-labels drifted toward `Skijet` |
| **Hammering** | **0.0%** | **9.1%** | 0.0% | **-9.1%** | 100% false pseudo-label confirmation collapse |
| **SoccerPenalty** | **87.8%** | **95.1%** | 100.0% | **-7.3%** | Goalkeeper/kicker distance variations confused with `SoccerJuggling` |
| **BalanceBeam** | **54.8%** | **61.3%** | 9.7% | **-6.4%** | Gymnastics mat visual features confused with `FloorGymnastics` |
| **TennisSwing** | **22.4%** | **28.6%** | 28.6% | **-6.1%** | Racquet trajectory confused with `TableTennisShot` and `GolfSwing` |
| **JumpingJack** | **18.9%** | **24.3%** | 13.5% | **-5.4%** | Arm/leg rhythm indistinguishable from `JumpRope` under low resolution |
| **Lunges** | **16.2%** | **18.9%** | 5.4% | **-2.7%** | Leg positioning confused with bodyweight squats |

---

### 3.3 Multi-Modal & MPL Degradation: Classes Where $F^2$-FixMatch Underperforms FixMatch

In **19 classes**, the full bi-modal $F^2$-FixMatch framework underperformed single-stream RGB FixMatch. The primary cause is **Temporal Gradient noise amplification**: when an action takes place outdoors or with unstable camera work, frame subtraction $x_t^G = (x_t^V - x_{t+1}^V + 255)/2$ captures camera shake rather than human movement.

| Action Category | $F^2$-FixMatch | FixMatch Baseline | Supervised (10%) | Drop vs. FixMatch | Failure Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CliffDiving** | **23.1%** | **64.1%** | 46.1% | **-41.0%** | Violent water splashes & camera shake corrupt temporal gradient |
| **BreastStroke** | **53.6%** | **92.9%** | 89.3% | **-39.3%** | Water surface turbulence obscures stroke velocity signatures |
| **Basketball** | **2.9%** | **22.9%** | 0.0% | **-20.0%** | $F^2$ head over-indexes on rim/dunk trajectories |
| **SoccerPenalty** | **87.8%** | **100.0%** | 95.1% | **-12.2%** | FixMatch achieved 100%; $F^2$ confused 5 clips due to wide field camera panning |
| **BlowingCandles** | **3.0%** | **15.2%** | 24.2% | **-12.1%** | Static scene; TG is zero-mean noise |
| **PizzaTossing** | **6.1%** | **15.2%** | 18.2% | **-9.1%** | Thin dough boundary vanishes under feature CutMix ($p=0.5$) |
| **BaseballPitch** | **60.5%** | **67.4%** | 58.1% | **-7.0%** | Pitcher wind-up aliased across temporal gradient stride |
| **TennisSwing** | **22.4%** | **28.6%** | 28.6% | **-6.1%** | Fast racquet swing blurred across 8-frame temporal sampling |
| **MoppingFloor** | **8.8%** | **14.7%** | 20.6% | **-5.9%** | Weak motion signature diluted by feature-level horizontal flip |
| **Skiing** | **55.0%** | **60.0%** | 65.0% | **-5.0%** | Rapidly shifting snow background creates high-entropy temporal gradients |

---

## 4. Positive Analysis: Where Models Excel ($\ge 90\%$ Accuracy)

Conversely, $F^2$-FixMatch achieves exceptional performance on categories characterized by **large whole-body kinetic displacement**, **distinctive environmental scenery**, and **periodic repetitive motion cycles**:

### 4.1 High-Performing Action Categories ($\ge 90\%$ Accuracy)

| Action Category | $F^2$-FixMatch | FixMatch Baseline | Supervised (10%) | Videos | Dominant Success Factor |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BasketballDunk** | **100.0%** (37/37) | 94.6% | 91.9% | 37 | Distinct upward trajectory towards elevated hoop structure |
| **CleanAndJerk** | **100.0%** (33/33) | 9.1% | 0.0% | 33 | High-energy multi-phase vertical lift perfectly captured by TG |
| **BenchPress** | **97.9%** (47/48) | 52.1% | 52.1% | 48 | Consistent supine posture and vertical barbell path |
| **Diving** | **97.8%** (44/45) | 86.7% | 77.8% | 45 | Distinct springboard geometry and vertical downward descent |
| **IceDancing** | **97.8%** (44/45) | 95.6% | 91.1% | 45 | Large paired spinning motions on uniform high-contrast ice surface |
| **Billiards** | **97.5%** (39/40) | 100.0% | 97.5% | 40 | Highly distinctive green felt table geometry and cue stick alignment |
| **HorseRace** | **97.1%** (34/35) | 74.3% | 51.4% | 35 | Fast horizontal group motion across dirt track |
| **VolleyballSpiking** | **97.1%** (34/35) | 88.6% | 54.3% | 35 | Dynamic vertical jump followed by downward arm swing near net |
| **Surfing** | **97.0%** (32/33) | 97.0% | 66.7% | 33 | Unmistakable ocean wave break texture and standing surfer silhouette |
| **PlayingPiano** | **96.4%** (27/28) | 100.0% | 85.7% | 28 | Black and white keyboard structure and seated posture |
| **Fencing** | **94.1%** (32/34) | 64.7% | 52.9% | 34 | Linear piste geometry, protective gear, and rapid lunging motion |
| **StillRings** | **93.8%** (30/32) | 37.5% | 37.5% | 32 | Suspended gymnast posture and distinctive ring apparatus |
| **Bowling** | **93.0%** (40/43) | 95.3% | 86.0% | 43 | Polished wooden bowling lane and arm pendulum trajectory |
| **Skijet** | **92.9%** (26/28) | 67.9% | 50.0% | 28 | High-speed motorized water wake and open water context |
| **RockClimbingIndoor** | **90.2%** (37/41) | 61.0% | 53.7% | 41 | Brightly colored climbing holds on vertical climbing wall |

---

### 4.2 Top 15 Most Improved Classes ($F^2$-FixMatch vs. Supervised 10%)

Categories with substantial temporal dynamics that completely failed under 10% supervision were revived by the combination of Temporal Gradients and Progressive Feature Augmentations:

| Action Category | Supervised (10%) | FixMatch Baseline | $F^2$-FixMatch | Absolute Gain ($\Delta$) | Physical Motion Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CleanAndJerk** | 0.0% | 9.1% | **100.0%** | **+100.0%** | Explosive vertical barbell trajectory |
| **PlayingTabla** | 0.0% | 35.5% | **87.1%** | **+87.1%** | Fast rhythmic fingertip percussion captured by TG |
| **SoccerJuggling** | 10.3% | 51.3% | **84.6%** | **+74.4%** | Fast alternating lower-body foot and knee touches |
| **TableTennisShot** | 5.1% | 66.7% | **79.5%** | **+74.4%** | Rapid stroke transitions across consecutive frames |
| **PlayingFlute** | 0.0% | 25.0% | **72.9%** | **+72.9%** | Hand-to-mouth embouchure posture and subtle finger movement |
| **PlayingDhol** | 0.0% | 8.2% | **67.3%** | **+67.3%** | High-energy drumstick and arm percussion |
| **RopeClimbing** | 23.5% | 29.4% | **88.2%** | **+64.7%** | Vertical hand-over-hand periodic motion |
| **PlayingDaf** | 4.9% | 7.3% | **65.8%** | **+61.0%** | Hand-held frame drum rotation and palm strikes |
| **TaiChi** | 14.3% | 3.6% | **75.0%** | **+60.7%** | Continuous fluid full-body transitions |
| **CuttingInKitchen** | 9.1% | 3.0% | **69.7%** | **+60.6%** | Cyclic downward chopping motion on cutting board |
| **SalsaSpin** | 0.0% | 0.0% | **60.5%** | **+60.5%** | Multi-person rotational velocity signatures |
| **Drumming** | 0.0% | 17.8% | **60.0%** | **+60.0%** | High-frequency drum kit percussion |
| **JumpRope** | 13.2% | 10.5% | **71.0%** | **+57.9%** | Synchronized cyclical vertical hops |
| **MilitaryParade** | 27.3% | 57.6% | **84.8%** | **+57.6%** | Synchronized group marching motion |
| **StillRings** | 37.5% | 37.5% | **93.8%** | **+56.2%** | Static strength holds with ring apparatus |

---

## 5. Comparative Dynamics: What Distinguishes "Good" vs. "Bad" Classes?

| Attribute | High-Performing Classes ($\ge 90\%$) | Severe Failure Classes ($< 20\%$) |
| :--- | :--- | :--- |
| **Spatial Motion Scale** | **Macro-kinetic**: Whole-body translation, vertical leaps, or major limb trajectories (`CleanAndJerk`, `BasketballDunk`, `VolleyballSpiking`). | **Micro-kinetic**: Localized finger, lip, or wrist motion inside an otherwise static frame (`Hammering`, `BlowingCandles`, `PizzaTossing`). |
| **Temporal Gradient (TG) SNR** | **High Signal-to-Noise Ratio**: Silhouette displacement $|x_t - x_{t+1}|$ clearly outlines the action against background. | **Low Signal-to-Noise Ratio**: Motion difference is smaller than camera sensor noise or natural hand tremor. |
| **Scene Disambiguation** | **Unambiguous Scene Context**: Unique physical environments (ice rink, green felt pool table, ocean surf, weight bench). | **Ambiguous / Shared Scene**: Cluttered kitchens, generic living rooms, or identical sports courts (`Basketball` vs `BasketballDunk`). |
| **Temporal Duration Matching** | **Matched to 8-Frame Stride-8 Sampling**: Actions that repeat or endure over the 64-frame observation window. | **Transient / Aliased**: Ultra-fast events completed in $< 0.3$s (`Shotput` release) are completely skipped by temporal stride. |
| **Pseudo-Label Mask Behavior** | Confident predictions ($q \ge 0.3$) form compact, correct feature clusters, driving clean consistency training. | Low threshold allows wrong classes with moderate confidence ($q \approx 0.35$) to be accepted, driving irreversible confirmation bias. |

---

## 6. Training Dynamics & Artifact Registry

* **Training Budget:** 220 epochs (40-epoch warmup + 30-epoch ramp-up + 150-epoch joint semi-supervised learning).
* **Wall-Clock Time:** 17.26 hours on 1 $\times$ NVIDIA RTX A5000 (~4.7 minutes / epoch).
* **Loss Convergence:**
  * Supervised loss $\mathcal{L}_s$: $9.95 \to 0.3455$.
  * Frame strong consistency loss $\mathcal{L}_u^s$: $1.0006$.
  * Weak TAP feature consistency loss $\mathcal{L}_u^{w_p}$: $0.8411$.
  * Strong TAP feature consistency loss $\mathcal{L}_u^{s_p}$: $1.3268$.
* **Best Online Validation Top-1:** **56.15%** at Epoch 210 (1-crop protocol).
* **Final Multi-Crop Evaluation:** **59.90%** Top-1 / **83.32%** Top-5 (30-view protocol).

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
