#!/usr/bin/env bash
# ==============================================================================
# Run D: Dual-Stream FixMatch Control (RGB + TG, beta=0.0, gamma=0.0)
# Target platform: Linux (e.g. RTX A5000 GPU)
# Paper: "Leveraging Frame- and Feature-level Progressive Augmentation for
#         Semi-supervised Action Recognition" (Tu et al., ACM TOMM 2025)
# ==============================================================================
set -euo pipefail

# Directory of this script and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Default parameters (can be overridden via environment variables)
PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

DATA_ROOT="${DATA_ROOT:-data/ucf101}"
SPLITS_DIR="${SPLITS_DIR:-data/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/fixmatch_rgb_tg}"
SPLIT="${SPLIT:-1}"
LABELED_PERCENT="${LABELED_PERCENT:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
MU="${MU:-4}"
EPOCHS="${EPOCHS:-220}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
RAMPUP_EPOCHS="${RAMPUP_EPOCHS:-30}"
LR="${LR:-0.2}"
THRESHOLD="${THRESHOLD:-0.3}"
ALPHA="${ALPHA:-2.0}"
BETA="${BETA:-0.0}"
GAMMA="${GAMMA:-0.0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_FREQ="${EVAL_FREQ:-5}"

echo "======================================================================"
echo "Run D: Dual-Stream FixMatch Control (RGB + TG)"
echo "Repo Root:      ${REPO_ROOT}"
echo "Python:         ${PYTHON}"
echo "Data Root:      ${DATA_ROOT}"
echo "Splits Dir:     ${SPLITS_DIR}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo "Hyperparameters:"
echo "  Split: ${SPLIT} | Labeled %: ${LABELED_PERCENT}% | Seed: ${SEED}"
echo "  Batch Size: B_l=${BATCH_SIZE}, mu=${MU} (B_u=$((BATCH_SIZE * MU))) | Accum Steps: ${ACCUM_STEPS}"
echo "  Epochs: ${EPOCHS} | Warmup: ${WARMUP_EPOCHS} | Rampup: ${RAMPUP_EPOCHS}"
echo "  LR: ${LR} | Threshold tau: ${THRESHOLD}"
echo "  Loss Weights: alpha=${ALPHA}, beta=${BETA}, gamma=${GAMMA}"
echo "======================================================================"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" train_f2_fixmatch.py \
    --data-root "${DATA_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --split "${SPLIT}" \
    --labeled-percent "${LABELED_PERCENT}" \
    --batch-size "${BATCH_SIZE}" \
    --accum-steps "${ACCUM_STEPS}" \
    --mu "${MU}" \
    --epochs "${EPOCHS}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --rampup-epochs "${RAMPUP_EPOCHS}" \
    --lr "${LR}" \
    --threshold "${THRESHOLD}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
    --eval-freq "${EVAL_FREQ}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo "======================================================================"
echo "[SUCCESS] Run D training complete. Artifacts saved in ${OUTPUT_DIR}"
echo "======================================================================"
