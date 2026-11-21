#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/eval_tau_095.sh
# Purpose: Evaluate FixMatch trained with tau=0.95 on the matched 30-view protocol.
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
echo "Evaluating FixMatch (tau=0.95) on Matched 30-View Protocol"
echo "Python: $($PY_BIN --version 2>&1)"
echo "Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

mkdir -p checkpoints/fixmatch_tau_095

# 1. Best Model Evaluation
if [ -f "checkpoints/fixmatch_tau_095/best_model.pth" ]; then
    echo ""
    echo "[1/2] Evaluating FixMatch (tau=0.95) - Best Model (30-View)..."
    $PY_BIN evaluate.py \
      --checkpoint checkpoints/fixmatch_tau_095/best_model.pth \
      --data-root data/ucf101 \
      --split-file data/splits/ucf101_val_split_1_rawframes.txt \
      --eval-mode full \
      --crop-size 256 \
      --batch-size 8 \
      --output-file checkpoints/fixmatch_tau_095/eval_metrics_full30view.json
else
    echo "[Warning] checkpoints/fixmatch_tau_095/best_model.pth not found!"
fi

# 2. Final Epoch Model Evaluation
if [ -f "checkpoints/fixmatch_tau_095/final_model.pth" ]; then
    echo ""
    echo "[2/2] Evaluating FixMatch (tau=0.95) - Final Epoch Model (30-View)..."
    $PY_BIN evaluate.py \
      --checkpoint checkpoints/fixmatch_tau_095/final_model.pth \
      --data-root data/ucf101 \
      --split-file data/splits/ucf101_val_split_1_rawframes.txt \
      --eval-mode full \
      --crop-size 256 \
      --batch-size 8 \
      --output-file checkpoints/fixmatch_tau_095/eval_metrics_final_30view.json
fi

echo ""
echo "========================================================================"
echo "FixMatch (tau=0.95) Evaluation Complete!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
