#!/bin/bash
#SBATCH --job-name=pert_infer
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/inference_%j_%a.out
#SBATCH --error=logs/inference_%j_%a.err

# ============================================================================
# Perturbation Expert Inference Script
# ============================================================================
#
# Usage:
#   # Single fold
#   sbatch --export=FOLD=0,CKPT_DIR=./checkpoints/perturbation/fold_0_K562 scripts/inference_perturbation.sbatch
#
#   # All 4 folds as array job
#   sbatch --array=0-3 --export=CKPT_BASE=./checkpoints/perturbation scripts/inference_perturbation.sbatch
#
# ============================================================================

set -e

mkdir -p logs

# ============================================================================
# Configuration
# ============================================================================

# Stack model for genelist
STACK_MODEL_DIR="${STACK_MODEL_DIR:-./models/Stack-Large}"
STACK_GENELIST="${STACK_MODEL_DIR}/basecount_1000per_15000max.pkl"

# Data paths
DATA_PATH="${DATA_PATH:-/data/replogle_nogwps_v2/replogle_concat.h5ad}"
SPLIT_TOML_DIR="${SPLIT_TOML_DIR:-/data/replogle_nogwps_v2/}"

# Checkpoint base directory
CKPT_BASE="${CKPT_BASE:-/checkpoints/pert_expert}"

# Determine fold
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    FOLD=${SLURM_ARRAY_TASK_ID}
elif [ -z "${FOLD}" ]; then
    echo "Error: FOLD not specified"
    exit 1
fi

# Map fold to cell line
CELL_LINES=("K562" "HepG2" "Jurkat" "RPE1")
CELL_LINE=${CELL_LINES[$FOLD]}

# Checkpoint directory for this fold
CKPT_DIR="${CKPT_DIR:-${CKPT_BASE}/fold_${FOLD}_${CELL_LINE}}"

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "Perturbation Expert Inference"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Fold: ${FOLD} (${CELL_LINE})"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "=============================================="

# Activate conda
source /home/dhruvgautam/miniconda3/etc/profile.d/conda.sh
conda activate stack2

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# ============================================================================
# Find Best Checkpoint
# ============================================================================

BEST_CKPT=$(ls -t ${CKPT_DIR}/*.ckpt 2>/dev/null | grep -v "last.ckpt" | head -1)
if [ -z "${BEST_CKPT}" ]; then
    BEST_CKPT="${CKPT_DIR}/last.ckpt"
fi

if [ ! -f "${BEST_CKPT}" ]; then
    echo "Error: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Using checkpoint: ${BEST_CKPT}"

# ============================================================================
# Run Inference
# ============================================================================

PRED_OUTPUT="${CKPT_DIR}/predictions_fold_${FOLD}_${CELL_LINE}.h5ad"

python scripts/inference_perturbation.py \
    --checkpoint "${BEST_CKPT}" \
    --data-path "${DATA_PATH}" \
    --split-toml-dir "${SPLIT_TOML_DIR}" \
    --genelist-path "${STACK_GENELIST}" \
    --fold "${FOLD}" \
    --output "${PRED_OUTPUT}" \
    --mode test \
    --num-sampling-steps 100 \
    --batch-size 8 \
    --aggregated

echo ""
echo "=============================================="
echo "Inference completed!"
echo "Predictions saved to: ${PRED_OUTPUT}"
echo "=============================================="
echo ""
echo "To evaluate with cell-eval:"
echo "  cell-eval run -i ${PRED_OUTPUT} --pert-col perturbation --control-pert non-targeting"
