# Checkpoint Registry & Model Weights

This directory stores evaluation logs, run configurations, training histories, and model checkpoints.

Per course submission rules, binary checkpoint files (`*.pth`, ~122 MB each) are excluded from the git repository to keep the repository lightweight (< 50 MB) and prevent exceeding GitHub push limits.

## Checkpoint Registry

| Experiment | Model Description | Checkpoint File | Eval Metrics (Committed) | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Run A** | Supervised 100% | `supervised_100/best_model.pth` | `results/eval_supervised_100_30view.json` | 72.91% Top-1 |
| **Run B** | Supervised 10% | `supervised_baseline/best_model.pth` | `supervised_baseline/eval_metrics_full30view.json` | 31.32% Top-1 |
| **Run C** | FixMatch RGB ($\tau=0.30$) | `fixmatch_baseline/best_model.pth` | `fixmatch_baseline/eval_metrics_full30view.json` | 38.49% Top-1 |
| **Run C-$\tau$** | FixMatch RGB ($\tau=0.95$) | `fixmatch_tau_095/best_model.pth` | `fixmatch_tau_095/eval_metrics_full30view.json` | 36.93% Top-1 |
| **Run D** | Dual-Stream Control ($\beta=0, \gamma=0$) | `fixmatch_rgb_tg/best_model.pth` | `fixmatch_rgb_tg/eval_metrics_full30view.json` | 59.03% Top-1 |
| **Run F** | Full $F^2$-FixMatch | `f2_fixmatch/best_model.pth` | `f2_fixmatch/eval_metrics_full30view.json` | 59.90% Top-1 |
| **PreciseBN** | Recalibrated $F^2$ | `f2_fixmatch_recal/best_model_precise_bn.pth` | `results/eval_f2_precise_bn_30view.json` | 62.81% Top-1 |

## External Checkpoint Download
To download pre-trained weights for offline evaluation or verification, model weights are archived on Google Drive / Hugging Face:
- Download Link: `[Archive URL to be inserted upon public release]`
- Checkpoint MD5 hashes and parameter counts are verified in `results/synthesis_report.md`.
