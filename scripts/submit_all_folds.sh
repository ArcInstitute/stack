#!/bin/bash
# ============================================================================
# Submit all 4 folds of perturbation expert training
# ============================================================================
#
# Usage:
#   ./scripts/submit_all_folds.sh
#
# The script will automatically download Stack-Large from HuggingFace if needed.
#
# ============================================================================

set -e

# Optional overrides
DATA_PATH="${1:-/data/replogle_nogwps_v2/replogle_concat.h5ad}"
OUTPUT_BASE="${2:-/checkpoints/pert_expert}"

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "Submitting 4-fold CV jobs"
echo "=============================================="
echo "Data path: ${DATA_PATH}"
echo "Output base: ${OUTPUT_BASE}"
echo "Stack model: Will download from HuggingFace if needed"
echo "Conda env: stack2"
echo "=============================================="

# Submit array job for all 4 folds
JOB_ID=$(sbatch --array=0-3 \
    --export=DATA_PATH="${DATA_PATH}",OUTPUT_BASE="${OUTPUT_BASE}" \
    scripts/train_perturbation_expert.sbatch | awk '{print $4}')

echo ""
echo "Submitted job array: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f logs/perturbation_${JOB_ID}_*.out"
echo ""
echo "Fold mapping:"
echo "  ${JOB_ID}_0 -> K562 (test)"
echo "  ${JOB_ID}_1 -> HepG2 (test)"
echo "  ${JOB_ID}_2 -> Jurkat (test)"
echo "  ${JOB_ID}_3 -> RPE1 (test)"
echo ""
echo "After completion, predictions will be at:"
echo "  /checkpoints/pert_expert/fold_0_K562/predictions_fold_0_K562.h5ad"
echo "  /checkpoints/pert_expert/fold_1_HepG2/predictions_fold_1_HepG2.h5ad"
echo "  /checkpoints/pert_expert/fold_2_Jurkat/predictions_fold_2_Jurkat.h5ad"
echo "  /checkpoints/pert_expert/fold_3_RPE1/predictions_fold_3_RPE1.h5ad"
