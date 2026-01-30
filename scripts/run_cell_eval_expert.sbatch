#!/bin/bash
#SBATCH --job-name=cell_eval_expert
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cell_eval_expert_%j_%a.out
#SBATCH --error=logs/cell_eval_expert_%j_%a.err

# ============================================================================
# Cell-Eval Script for PerturbationExpert Results
# ============================================================================
#
# Usage:
#   sbatch --array=0-3 scripts/run_cell_eval_expert.sbatch
#
# ============================================================================

set -e

mkdir -p logs

# Determine fold
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    FOLD=${SLURM_ARRAY_TASK_ID}
elif [ -z "${FOLD}" ]; then
    echo "Error: FOLD not specified"
    exit 1
fi

CELL_LINES=("K562" "HepG2" "Jurkat" "RPE1")
CELL_LINE=${CELL_LINES[$FOLD]}

# Paths
CKPT_DIR="${CKPT_DIR:-/checkpoints/pert_expert}"
INPUT_H5AD="${CKPT_DIR}/fold_${FOLD}_${CELL_LINE}/predictions_fold_${FOLD}_${CELL_LINE}_aggregated_mean.h5ad"
OUTPUT_DIR="${CKPT_DIR}/fold_${FOLD}_${CELL_LINE}/cell_eval"
DATA_PATH="${DATA_PATH:-/data/replogle_nogwps_v2/replogle_concat.h5ad}"
GENELIST_PATH="${GENELIST_PATH:-./models/Stack-Large/basecount_1000per_15000max.pkl}"

echo "=============================================="
echo "Cell-Eval for PerturbationExpert"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Fold: ${FOLD} (${CELL_LINE})"
echo "Input: ${INPUT_H5AD}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Activate conda
source /home/dhruvgautam/miniconda3/etc/profile.d/conda.sh
conda activate stack2

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Verify input exists
if [ ! -f "${INPUT_H5AD}" ]; then
    echo "Error: Input file not found: ${INPUT_H5AD}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Step 1: Prepare h5ad files for cell-eval
echo ""
echo "Preparing h5ad files..."
python scripts/prepare_celleval.py \
    --input "${INPUT_H5AD}" \
    --output-dir "${OUTPUT_DIR}" \
    --data-path "${DATA_PATH}" \
    --genelist-path "${GENELIST_PATH}" \
    --control-label non-targeting

# Step 2: Run cell-eval
echo ""
echo "Running cell-eval..."
cell-eval run \
    -ap "${OUTPUT_DIR}/pred.h5ad" \
    -ar "${OUTPUT_DIR}/real.h5ad" \
    --pert-col perturbation \
    --control-pert non-targeting \
    --outdir "${OUTPUT_DIR}/results" \
    --num-threads ${SLURM_CPUS_PER_TASK}

echo ""
echo "=============================================="
echo "Cell-eval completed for fold ${FOLD} (${CELL_LINE})"
echo "Results: ${OUTPUT_DIR}/results"
echo "=============================================="

# Show results
if [ -f "${OUTPUT_DIR}/results/metrics.json" ]; then
    echo ""
    echo "Metrics:"
    cat "${OUTPUT_DIR}/results/metrics.json"
fi

echo ""
echo "Done!"
