#!/bin/bash
#SBATCH --job-name=pert_expert
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/perturbation_%j_%a.out
#SBATCH --error=logs/perturbation_%j_%a.err

# ============================================================================
# Perturbation Expert Training Script
# ============================================================================
#
# Usage:
#   # Single fold (e.g., fold 0)
#   sbatch --export=FOLD=0 scripts/train_perturbation_expert.sbatch
#
#   # All 4 folds as array job
#   sbatch --array=0-3 scripts/train_perturbation_expert.sbatch
#
# ============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Configuration - Modify these as needed
# ============================================================================

# Stack model from HuggingFace (will be downloaded if not present)
STACK_MODEL_DIR="${STACK_MODEL_DIR:-./models/Stack-Large}"
STACK_CKPT="${STACK_MODEL_DIR}/bc_large.ckpt"
STACK_GENELIST="${STACK_MODEL_DIR}/basecount_1000per_15000max.pkl"

# Data paths
DATA_PATH="${DATA_PATH:-/data/replogle_nogwps_v2/replogle_concat.h5ad}"
SPLIT_TOML_DIR="${SPLIT_TOML_DIR:-/data/replogle_nogwps_v2/}"

# Output directory
OUTPUT_BASE="${OUTPUT_BASE:-/checkpoints/pert_expert}"

# Training config
CONFIG_FILE="${CONFIG_FILE:-configs/perturbation/replogle.yaml}"

# Determine fold from array task ID or environment variable
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    FOLD=${SLURM_ARRAY_TASK_ID}
elif [ -z "${FOLD}" ]; then
    echo "Error: FOLD not specified. Use --export=FOLD=N or --array=0-3"
    exit 1
fi

# Validate fold
if [ "${FOLD}" -lt 0 ] || [ "${FOLD}" -gt 3 ]; then
    echo "Error: FOLD must be 0-3 (got ${FOLD})"
    exit 1
fi

# Map fold to cell line name for logging
CELL_LINES=("K562" "HepG2" "Jurkat" "RPE1")
CELL_LINE=${CELL_LINES[$FOLD]}

# Output directory for this fold
OUTPUT_DIR="${OUTPUT_BASE}/fold_${FOLD}_${CELL_LINE}"
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "Perturbation Expert Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Fold: ${FOLD} (${CELL_LINE} as test set)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Activate conda environment
source /home/dhruvgautam/miniconda3/etc/profile.d/conda.sh
conda activate stack2

# Ensure dependencies are installed
pip install -q huggingface_hub tomli psutil
# Upgrade scvi-tools to be compatible with jax 0.9.x
pip install -q "scvi-tools>=1.4"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# Optimize for multi-GPU training
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ============================================================================
# Download Stack-Large from HuggingFace if not present
# ============================================================================

if [ ! -f "${STACK_CKPT}" ]; then
    echo ""
    echo "Downloading Stack-Large from HuggingFace..."
    echo ""

    mkdir -p "$(dirname ${STACK_MODEL_DIR})"

    python -c "
from huggingface_hub import snapshot_download
import os

repo_id = 'arcinstitute/Stack-Large'
target_dir = '${STACK_MODEL_DIR}'

if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type='model',
    revision='main',
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f'Downloaded to {target_dir}')
"
    echo "Download complete!"
fi

echo "Stack checkpoint: ${STACK_CKPT}"
echo "Stack genelist: ${STACK_GENELIST}"

# Verify files exist
if [ ! -f "${STACK_CKPT}" ]; then
    echo "Error: Stack checkpoint not found at ${STACK_CKPT}"
    exit 1
fi

# ============================================================================
# Run Training
# ============================================================================

echo ""
echo "Starting training..."
echo ""

python scripts/train_perturbation_expert.py \
    --config "${CONFIG_FILE}" \
    --stack-checkpoint "${STACK_CKPT}" \
    --data-path "${DATA_PATH}" \
    --split-toml-dir "${SPLIT_TOML_DIR}" \
    --genelist-path "${STACK_GENELIST}" \
    --fold "${FOLD}" \
    --save-dir "${OUTPUT_DIR}" \
    --run-name "fold_${FOLD}_${CELL_LINE}" \
    --gpus 1 \
    --strategy auto \
    --num-workers 4 \
    --batch-size 16 \
    --max-epochs 100 \
    --precision bf16-mixed

echo ""
echo "=============================================="
echo "Training completed for fold ${FOLD} (${CELL_LINE})"
echo "=============================================="

# ============================================================================
# Run Inference on Test Set
# ============================================================================

echo ""
echo "Running inference on test set..."
echo ""

# Find the best checkpoint
BEST_CKPT=$(ls -t ${OUTPUT_DIR}/*.ckpt | grep -v "last.ckpt" | head -1)
if [ -z "${BEST_CKPT}" ]; then
    BEST_CKPT="${OUTPUT_DIR}/last.ckpt"
fi

echo "Using checkpoint: ${BEST_CKPT}"

# Output h5ad file
PRED_OUTPUT="${OUTPUT_DIR}/predictions_fold_${FOLD}_${CELL_LINE}.h5ad"

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
echo ""
echo "Job completed successfully!"
