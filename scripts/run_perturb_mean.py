#!/usr/bin/env python3
"""Run PerturbMean baseline for perturbation prediction.

This script computes mean perturbation effects from training data and uses
them to generate predictions on the test set. Outputs are compatible with
cell-eval for benchmarking.

Usage:
    python scripts/run_perturb_mean.py \
        --data-path /data/replogle_concat.h5ad \
        --split-toml-dir /data/replogle_nogwps_v2/ \
        --genelist-path ./models/Stack-Large/basecount_1000per_15000max.pkl \
        --fold 0 \
        --output-dir ./results/perturb_mean/fold_0_K562
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_dataset(args, mode: str):
    """Create perturbation dataset for the specified mode."""
    from stack.data.perturbation import PerturbationDataset, PerturbationDatasetConfig

    config = PerturbationDatasetConfig(
        data_path=args.data_path,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
        cell_line_col=args.cell_line_col,
        n_cells=args.n_cells,
        genelist_path=args.genelist_path,
        split_toml_dir=args.split_toml_dir,
        fold=args.fold,
    )

    dataset = PerturbationDataset(
        config=config,
        mode=mode,
        random_state=args.seed,
    )

    return dataset


def run_inference_and_save(
    model,
    dataset,
    device: torch.device,
    gene_names: List[str],
    output_path: str,
    fold: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    create_aggregated: bool = False,
) -> Dict[str, float]:
    """Run inference and save results in a memory-efficient streaming manner.

    Instead of accumulating all predictions in memory, this function:
    1. Computes metrics incrementally using running statistics
    2. Writes h5ad file in chunks to avoid OOM

    Returns:
        Dictionary of computed metrics.
    """
    import anndata as ad
    import pandas as pd
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    n_genes = len(gene_names)
    n_cells_per_batch = dataset.config.n_cells
    control_label = dataset.config.control_label  # e.g., "non-targeting"

    # For incremental metrics computation
    # We'll compute per-perturbation mean predictions and ground truth
    pert_to_pred_sum = {}
    pert_to_gt_sum = {}
    pert_to_count = {}

    # For control expression (needed for cell-eval)
    control_sum = np.zeros(n_genes, dtype=np.float64)
    control_count = 0

    # For MSE computation
    mse_log1p_sum = 0.0
    mse_raw_sum = 0.0
    total_elements = 0

    # Collect all perturbation names for h5ad
    all_pert_names = []

    # First pass: compute metrics and collect perturbation info
    logging.info("Running inference and computing metrics...")
    model.eval()

    # We'll store aggregated predictions per perturbation for gene correlation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            control = batch["control_features"].to(device)
            perturbed = batch["perturbed_features"]
            pert_names = batch["perturbation_name"]

            # Accumulate control expression (mean across cells in batch)
            # Note: control is already log1p normalized from the dataset
            control_np = control.cpu().numpy()
            control_sum += control_np.mean(axis=(0, 1))  # Mean across batch and cells
            control_count += 1

            # Generate predictions
            result = model.sample(
                control_features=control,
                perturbation_names=list(pert_names),
            )

            # Use log1p outputs directly (data is already log1p normalized)
            pred_log1p = result["perturbed_log1p"].cpu().numpy()
            gt_log1p = perturbed.numpy()  # Already log1p from dataset

            # Compute MSE incrementally (both in log1p space)
            mse_log1p_sum += np.sum((pred_log1p - gt_log1p) ** 2)
            # Also compute approximate raw space MSE for comparison
            pred_raw = np.expm1(pred_log1p)
            gt_raw = np.expm1(gt_log1p)
            mse_raw_sum += np.sum((pred_raw - gt_raw) ** 2)
            total_elements += pred_log1p.size

            # Aggregate per perturbation (mean across cells) - keep in log1p space
            pred_mean = pred_log1p.mean(axis=1)  # (B, n_genes)
            gt_mean = gt_log1p.mean(axis=1)

            for i, pert_name in enumerate(pert_names):
                if pert_name not in pert_to_pred_sum:
                    pert_to_pred_sum[pert_name] = np.zeros(n_genes, dtype=np.float64)
                    pert_to_gt_sum[pert_name] = np.zeros(n_genes, dtype=np.float64)
                    pert_to_count[pert_name] = 0
                pert_to_pred_sum[pert_name] += pred_mean[i]
                pert_to_gt_sum[pert_name] += gt_mean[i]
                pert_to_count[pert_name] += 1

            all_pert_names.extend(pert_names)

    # Compute final metrics
    mse_log1p = mse_log1p_sum / total_elements
    mse_raw = mse_raw_sum / total_elements

    # Compute gene correlations from aggregated per-perturbation means
    unique_perts = sorted(pert_to_pred_sum.keys())
    n_perts = len(unique_perts)

    pred_matrix = np.zeros((n_perts, n_genes), dtype=np.float32)
    gt_matrix = np.zeros((n_perts, n_genes), dtype=np.float32)

    for i, pert in enumerate(unique_perts):
        pred_matrix[i] = pert_to_pred_sum[pert] / pert_to_count[pert]
        gt_matrix[i] = pert_to_gt_sum[pert] / pert_to_count[pert]

    # Compute gene-wise correlations
    gene_correlations = []
    for gene_idx in range(n_genes):
        pred_gene = pred_matrix[:, gene_idx]
        gt_gene = gt_matrix[:, gene_idx]

        if np.std(pred_gene) < 1e-8 or np.std(gt_gene) < 1e-8:
            continue

        corr, _ = stats.pearsonr(pred_gene, gt_gene)
        if not np.isnan(corr):
            gene_correlations.append(corr)

    gene_correlations = np.array(gene_correlations)

    metrics = {
        "gene_correlation_mean": float(np.mean(gene_correlations)),
        "gene_correlation_median": float(np.median(gene_correlations)),
        "gene_correlation_std": float(np.std(gene_correlations)),
        "mse_log1p": float(mse_log1p),
        "mse_raw": float(mse_raw),
        "n_genes_evaluated": len(gene_correlations),
        "n_samples": len(all_pert_names),
        "n_unique_perturbations": len(unique_perts),
    }

    # Compute mean control expression
    control_mean = (control_sum / control_count).astype(np.float32)

    # Add control to the data (needed for cell-eval)
    # Control prediction = control (no perturbation effect)
    # Control ground truth = control
    all_perts_with_ctrl = [control_label] + unique_perts
    pred_matrix_with_ctrl = np.vstack([control_mean.reshape(1, -1), pred_matrix])
    gt_matrix_with_ctrl = np.vstack([control_mean.reshape(1, -1), gt_matrix])

    # Save aggregated h5ad files (one row per perturbation - much smaller)
    # Note: All data is in log1p space (compatible with cell-eval)
    logging.info("Saving aggregated h5ad files (log1p normalized)...")
    obs = pd.DataFrame({
        "perturbation": all_perts_with_ctrl,
    })
    obs.index = all_perts_with_ctrl

    var = pd.DataFrame(index=gene_names)
    var.index.name = "gene"

    # Save predictions h5ad
    adata_pred = ad.AnnData(
        X=pred_matrix_with_ctrl,
        obs=obs.copy(),
        var=var.copy(),
    )
    adata_pred.uns["perturb_mean"] = {
        "unique_perturbations": all_perts_with_ctrl,
        "fold": fold,
        "model": "PerturbMean",
        "aggregation": "mean",
        "normalization": "log1p",
    }

    output_pred_path = output_path.replace(".h5ad", "_pred.h5ad")
    adata_pred.write_h5ad(output_pred_path)
    logging.info(f"Saved predictions to {output_pred_path} with shape {adata_pred.shape}")

    # Save ground truth h5ad (for cell-eval)
    adata_real = ad.AnnData(
        X=gt_matrix_with_ctrl,
        obs=obs.copy(),
        var=var.copy(),
    )
    adata_real.uns["perturb_mean"] = {
        "unique_perturbations": all_perts_with_ctrl,
        "fold": fold,
        "type": "ground_truth",
    }

    output_real_path = output_path.replace(".h5ad", "_real.h5ad")
    adata_real.write_h5ad(output_real_path)
    logging.info(f"Saved ground truth to {output_real_path} with shape {adata_real.shape}")

    # Clean up
    del pred_matrix, gt_matrix, pred_matrix_with_ctrl, gt_matrix_with_ctrl
    del adata_pred, adata_real

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run PerturbMean baseline for perturbation prediction"
    )

    # Required arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to h5ad file with perturbation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # Data configuration
    parser.add_argument(
        "--split-toml-dir",
        type=str,
        default=None,
        help="Directory containing TOML files for train/test splits",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for 4-fold CV (0=K562, 1=HepG2, 2=Jurkat, 3=RPE1)",
    )
    parser.add_argument(
        "--genelist-path",
        type=str,
        default=None,
        help="Path to gene list pickle file",
    )
    parser.add_argument(
        "--perturbation-col",
        type=str,
        default="gene",
        help="Column name for perturbation labels",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default="non-targeting",
        help="Label for control cells",
    )
    parser.add_argument(
        "--cell-line-col",
        type=str,
        default="cell_line",
        help="Column name for cell line labels",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=128,
        help="Number of cells per sample",
    )

    # Processing configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Output configuration
    parser.add_argument(
        "--aggregated",
        action="store_true",
        help="Also output aggregated predictions (mean per perturbation)",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    setup_logging()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Get fold information
    cell_lines = ["K562", "HepG2", "Jurkat", "RPE1"]
    if args.fold is not None:
        fold_cell_line = cell_lines[args.fold]
        logging.info(f"Running fold {args.fold} (test cell line: {fold_cell_line})")
    else:
        fold_cell_line = "combined"
        logging.info("Running with combined test set (all TOML test genes)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create datasets
    logging.info("Creating training dataset...")
    train_dataset = create_dataset(args, mode="train")
    logging.info(f"Training dataset: {len(train_dataset)} samples")

    logging.info("Creating test dataset...")
    test_dataset = create_dataset(args, mode="test")
    logging.info(f"Test dataset: {len(test_dataset)} samples")
    logging.info(f"Test perturbations: {len(test_dataset.test_perts)}")

    # Get gene names
    if train_dataset.target_genes is not None:
        gene_names = train_dataset.target_genes
        n_genes = len(gene_names)
    else:
        gene_names = list(train_dataset.gene_names)
        n_genes = len(gene_names)

    logging.info(f"Number of genes: {n_genes}")

    # Create model
    from stack.models.perturbation import PerturbMean

    model = PerturbMean(n_genes=n_genes)

    # Compute means from training data
    logging.info("Computing mean deltas from training data...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pert_counts = model.compute_means_from_dataloader(train_loader, device=device)
    logging.info(f"Computed means for {len(pert_counts)} perturbations")

    # Save model checkpoint
    checkpoint_path = os.path.join(
        args.output_dir, f"perturb_mean_fold_{args.fold}.pt"
    )
    model.save_checkpoint(checkpoint_path)

    # Run inference and save results (memory-efficient)
    h5ad_path = os.path.join(
        args.output_dir, f"predictions_fold_{args.fold}_{fold_cell_line}.h5ad"
    )

    metrics = run_inference_and_save(
        model=model,
        dataset=test_dataset,
        device=device,
        gene_names=gene_names,
        output_path=h5ad_path,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        create_aggregated=args.aggregated,
    )

    # Add fold info to metrics
    metrics["fold"] = args.fold
    metrics["fold_cell_line"] = fold_cell_line
    metrics["n_perturbations_train"] = len(pert_counts)
    metrics["n_perturbations_test"] = len(test_dataset.test_perts)

    # Log metrics
    logging.info("=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    logging.info(f"Gene correlation (mean): {metrics['gene_correlation_mean']:.4f}")
    logging.info(f"Gene correlation (median): {metrics['gene_correlation_median']:.4f}")
    logging.info(f"MSE (log1p): {metrics['mse_log1p']:.4f}")
    logging.info(f"MSE (raw): {metrics['mse_raw']:.4f}")
    logging.info(f"Genes evaluated: {metrics['n_genes_evaluated']}")
    logging.info("=" * 60)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"metrics_fold_{args.fold}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")

    logging.info("Inference complete!")

    # Print cell-eval usage hint
    h5ad_pred_path = h5ad_path.replace(".h5ad", "_pred.h5ad")
    h5ad_real_path = h5ad_path.replace(".h5ad", "_real.h5ad")
    print("\n" + "=" * 60)
    print("To evaluate with cell-eval:")
    print("=" * 60)
    print(f"cell-eval run \\")
    print(f"    -ap {h5ad_pred_path} \\")
    print(f"    -ar {h5ad_real_path} \\")
    print(f"    --pert-col perturbation \\")
    print(f"    --control-pert {args.control_label}")
    print("=" * 60)


if __name__ == "__main__":
    main()
