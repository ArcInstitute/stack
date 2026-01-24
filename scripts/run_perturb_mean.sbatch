#!/bin/bash
#SBATCH --job-name=perturb_mean
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/perturb_mean_%j_%a.out
#SBATCH --error=logs/perturb_mean_%j_%a.err

# ============================================================================
# PerturbMean Baseline Script
# ============================================================================
#
# Usage:
#   # Single fold (e.g., fold 0)
#   sbatch --export=FOLD=0 scripts/run_perturb_mean.sbatch
#
#   # All 4 folds as array job
#   sbatch --array=0-3 scripts/run_perturb_mean.sbatch
#
# ============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Configuration - Modify these as needed
# ============================================================================

# Stack model genelist (for gene subsetting)
STACK_MODEL_DIR="${STACK_MODEL_DIR:-./models/Stack-Large}"
STACK_GENELIST="${STACK_MODEL_DIR}/basecount_1000per_15000max.pkl"

# Data paths
DATA_PATH="${DATA_PATH:-/data/replogle_nogwps_v2/replogle_concat.h5ad}"
SPLIT_TOML_DIR="${SPLIT_TOML_DIR:-/data/replogle_nogwps_v2/}"

# Output directory
OUTPUT_BASE="${OUTPUT_BASE:-./results/perturb_mean}"

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
echo "PerturbMean Baseline"
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
pip install -q tomli psutil

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# Optimize for multi-threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ============================================================================
# Download Stack-Large genelist from HuggingFace if not present
# ============================================================================

if [ ! -f "${STACK_GENELIST}" ]; then
    echo ""
    echo "Downloading Stack-Large genelist from HuggingFace..."
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
    allow_patterns=['*.pkl'],  # Only download genelist
)
print(f'Downloaded to {target_dir}')
"
    echo "Download complete!"
fi

echo "Genelist: ${STACK_GENELIST}"

# Verify genelist exists
if [ ! -f "${STACK_GENELIST}" ]; then
    echo "Error: Genelist not found at ${STACK_GENELIST}"
    exit 1
fi

# ============================================================================
# Run PerturbMean
# ============================================================================

echo ""
echo "Starting PerturbMean baseline..."
echo ""

python scripts/run_perturb_mean.py \
    --data-path "${DATA_PATH}" \
    --split-toml-dir "${SPLIT_TOML_DIR}" \
    --genelist-path "${STACK_GENELIST}" \
    --fold "${FOLD}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 16 \
    --num-workers 4 \
    --aggregated

echo ""
echo "=============================================="
echo "PerturbMean completed for fold ${FOLD} (${CELL_LINE})"
echo "=============================================="

# Display results summary
METRICS_FILE="${OUTPUT_DIR}/metrics_fold_${FOLD}.json"
if [ -f "${METRICS_FILE}" ]; then
    echo ""
    echo "Metrics:"
    cat "${METRICS_FILE}"
    echo ""
fi

PRED_OUTPUT="${OUTPUT_DIR}/predictions_fold_${FOLD}_${CELL_LINE}.h5ad"
echo ""
echo "To evaluate with cell-eval:"
echo "  cell-eval run -i ${PRED_OUTPUT} --pert-col perturbation --control-pert non-targeting"
echo ""
echo "Job completed successfully!"
