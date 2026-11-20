#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/run_remaining_evals.sh
# Purpose: Execute all remaining 30-view evaluations, diagnostics, and final
#          mathematical synthesis after completing Run D training.
# ==============================================================================

set -e

# Resolve Python executable
if [ -n "$PYTHON" ]; then
    PY_BIN="$PYTHON"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo "[Error] Neither python3 nor python found in PATH." >&2
    exit 1
fi

echo "========================================================================"
echo "RUNNING ALL REMAINING EVALUATIONS, DIAGNOSTICS & SYNTHESIS"
echo "Python: $($PY_BIN --version 2>&1) ($PY_BIN)"
echo "Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

mkdir -p results
mkdir -p checkpoints/fixmatch_rgb_tg

# ------------------------------------------------------------------------------
# STEP 1: Evaluate Run D (Control) on 30-View Protocol
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/7] Evaluating Run D (RGB+TG Control) - Best Model (30-View)..."
$PY_BIN evaluate.py \
  --checkpoint checkpoints/fixmatch_rgb_tg/best_model.pth \
  --data-root data/ucf101 \
  --split-file data/splits/ucf101_val_split_1_rawframes.txt \
  --eval-mode full \
  --crop-size 256 \
  --batch-size 8 \
  --output-file checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json

echo ""
echo "[Step 1b/7] Evaluating Run D - Final Epoch Model (30-View)..."
if [ -f "checkpoints/fixmatch_rgb_tg/final_model.pth" ]; then
    $PY_BIN evaluate.py \
      --checkpoint checkpoints/fixmatch_rgb_tg/final_model.pth \
      --data-root data/ucf101 \
      --split-file data/splits/ucf101_val_split_1_rawframes.txt \
      --eval-mode full \
      --crop-size 256 \
      --batch-size 8 \
      --output-file checkpoints/fixmatch_rgb_tg/eval_metrics_final_30view.json
fi

# ------------------------------------------------------------------------------
# STEP 2: Evaluate Supervised 100% on Matched 30-View Protocol (Upper Oracle)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/7] Evaluating Supervised 100% on Matched 30-View Protocol..."
$PY_BIN evaluate.py \
  --checkpoint checkpoints/supervised_100/best_model.pth \
  --data-root data/ucf101 \
  --split-file data/splits/ucf101_val_split_1_rawframes.txt \
  --eval-mode full \
  --crop-size 256 \
  --batch-size 8 \
  --output-file results/eval_supervised_100_30view.json

# ------------------------------------------------------------------------------
# STEP 3: Evaluate Final Epoch Checkpoints for Baselines
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/7] Evaluating Final Epoch Checkpoints for Baselines..."

if [ -f "checkpoints/supervised_baseline/final_model.pth" ]; then
    echo "  - Evaluating Supervised 10% (Final Epoch 120)..."
    $PY_BIN evaluate.py \
      --checkpoint checkpoints/supervised_baseline/final_model.pth \
      --data-root data/ucf101 \
      --split-file data/splits/ucf101_val_split_1_rawframes.txt \
      --eval-mode full \
      --crop-size 256 \
      --batch-size 8 \
      --output-file results/eval_supervised_10_final_30view.json
fi

if [ -f "checkpoints/fixmatch_baseline/final_model.pth" ]; then
    echo "  - Evaluating FixMatch RGB (Final Epoch 220)..."
    $PY_BIN evaluate.py \
      --checkpoint checkpoints/fixmatch_baseline/final_model.pth \
      --data-root data/ucf101 \
      --split-file data/splits/ucf101_val_split_1_rawframes.txt \
      --eval-mode full \
      --crop-size 256 \
      --batch-size 8 \
      --output-file results/eval_fixmatch_final_30view.json
fi

if [ -f "checkpoints/f2_fixmatch/final_model.pth" ]; then
    echo "  - Evaluating Full F2-FixMatch (Final Epoch 220)..."
    $PY_BIN evaluate.py \
      --checkpoint checkpoints/f2_fixmatch/final_model.pth \
      --data-root data/ucf101 \
      --split-file data/splits/ucf101_val_split_1_rawframes.txt \
      --eval-mode full \
      --crop-size 256 \
      --batch-size 8 \
      --output-file results/eval_f2_final_30view.json
fi

# ------------------------------------------------------------------------------
# STEP 4: Pseudo-Label Coverage and Precision Analysis (Unlabeled Pool)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/7] Generating Pseudo-Label Precision and Coverage Curves..."
$PY_BIN analyze_pseudolabels.py \
  --checkpoint checkpoints/f2_fixmatch/best_model.pth \
  --data-root data/ucf101 \
  --split-file data/splits/ucf101_train_10_percent_unlabeled_split_1_rawframes.txt \
  --modality mpl \
  --output-dir results/pseudolabel_f2

# ------------------------------------------------------------------------------
# STEP 5: PreciseBN Recalibration Diagnostic on F2-FixMatch
# ------------------------------------------------------------------------------
echo ""
echo "[Step 5/7] Running PreciseBN Diagnostic on F2-FixMatch..."
$PY_BIN recalibrate_bn.py \
  --checkpoint checkpoints/f2_fixmatch/best_model.pth \
  --data-root data/ucf101 \
  --split-file data/splits/ucf101_train_10_percent_labeled_split_1_rawframes.txt \
  --output-dir checkpoints/f2_fixmatch_recal \
  --num-batches 200 \
  --batch-size 4

# ------------------------------------------------------------------------------
# STEP 6: 30-View Evaluation of Recalibrated F2-FixMatch
# ------------------------------------------------------------------------------
echo ""
echo "[Step 6/7] Evaluating Recalibrated F2-FixMatch (30-View)..."
$PY_BIN evaluate.py \
  --checkpoint checkpoints/f2_fixmatch_recal/best_model_precise_bn.pth \
  --data-root data/ucf101 \
  --split-file data/splits/ucf101_val_split_1_rawframes.txt \
  --eval-mode full \
  --crop-size 256 \
  --batch-size 8 \
  --output-file results/eval_f2_precise_bn_30view.json

# ------------------------------------------------------------------------------
# STEP 7: Run Final Scientific Synthesis & Disentanglement Report
# ------------------------------------------------------------------------------
echo ""
echo "[Step 7/7] Running Mathematical Synthesis & Disentanglement Analysis..."
$PY_BIN synthesize_results.py \
  --run-a results/eval_supervised_100_30view.json \
  --run-d checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json

echo ""
echo "========================================================================"
echo "ALL REMAINING TASKS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Review:"
echo "  - Report:      results/synthesis_report.md"
echo "  - Updated:     RESULTS.md (Section 4)"
echo "  - Curves:      results/pseudolabel_f2/pseudolabel_curves.png"
echo "========================================================================"
