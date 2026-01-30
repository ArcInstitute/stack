#!/bin/bash
#SBATCH --job-name=cell_eval
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cell_eval_%j_%a.out
#SBATCH --error=logs/cell_eval_%j_%a.err

# ============================================================================
# Cell-Eval Script for PerturbMean Results
# ============================================================================
#
# Usage:
#   # Single fold (e.g., fold 0)
#   sbatch --export=FOLD=0 scripts/run_cell_eval.sbatch
#
#   # All 4 folds as array job
#   sbatch --array=0-3 scripts/run_cell_eval.sbatch
#
# ============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

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

# Map fold to cell line name
CELL_LINES=("K562" "HepG2" "Jurkat" "RPE1")
CELL_LINE=${CELL_LINES[$FOLD]}

# Input/output paths
RESULTS_DIR="${RESULTS_DIR:-./results/perturb_mean}"
PRED_H5AD="${RESULTS_DIR}/fold_${FOLD}_${CELL_LINE}/predictions_fold_${FOLD}_${CELL_LINE}_pred.h5ad"
REAL_H5AD="${RESULTS_DIR}/fold_${FOLD}_${CELL_LINE}/predictions_fold_${FOLD}_${CELL_LINE}_real.h5ad"
OUTPUT_DIR="${RESULTS_DIR}/fold_${FOLD}_${CELL_LINE}/cell_eval"

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "Cell-Eval for PerturbMean"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Fold: ${FOLD} (${CELL_LINE})"
echo "Predictions: ${PRED_H5AD}"
echo "Ground Truth: ${REAL_H5AD}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Activate conda environment
source /home/dhruvgautam/miniconda3/etc/profile.d/conda.sh
conda activate stack2

# Verify input files exist
if [ ! -f "${PRED_H5AD}" ]; then
    echo "Error: Predictions file not found: ${PRED_H5AD}"
    exit 1
fi

if [ ! -f "${REAL_H5AD}" ]; then
    echo "Error: Ground truth file not found: ${REAL_H5AD}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Run Cell-Eval
# ============================================================================

echo ""
echo "Running cell-eval..."
echo ""

cell-eval run \
    -ap "${PRED_H5AD}" \
    -ar "${REAL_H5AD}" \
    --pert-col perturbation \
    --control-pert non-targeting \
    --outdir "${OUTPUT_DIR}" \
    --num-threads ${SLURM_CPUS_PER_TASK}

echo ""
echo "=============================================="
echo "Cell-eval completed for fold ${FOLD} (${CELL_LINE})"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="

# Display results if available
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo ""
    echo "Metrics:"
    cat "${OUTPUT_DIR}/metrics.json"
fi

echo ""
echo "Job completed successfully!"
