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


def run_inference(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Run inference on the dataset and collect predictions.

    Returns:
        Dictionary containing:
            - predictions: np.array of predicted expression (n_samples, n_cells, n_genes)
            - ground_truth: np.array of actual perturbed expression
            - control: np.array of control expression used as context
            - perturbation_names: List of perturbation names
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_predictions = []
    all_ground_truth = []
    all_control = []
    all_pert_names = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            control = batch["control_features"].to(device)
            perturbed = batch["perturbed_features"]
            pert_names = batch["perturbation_name"]

            # Generate predictions
            result = model.sample(
                control_features=control,
                perturbation_names=list(pert_names),
            )

            # Collect results
            all_predictions.append(result["perturbed"].cpu().numpy())
            all_ground_truth.append(perturbed.numpy())
            all_control.append(control.cpu().numpy())
            all_pert_names.extend(pert_names)

    return {
        "predictions": np.concatenate(all_predictions, axis=0),
        "ground_truth": np.concatenate(all_ground_truth, axis=0),
        "control": np.concatenate(all_control, axis=0),
        "perturbation_names": all_pert_names,
    }


def compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        results: Dictionary with predictions and ground_truth arrays.

    Returns:
        Dictionary of metrics including gene correlations and MSE.
    """
    predictions = results["predictions"]  # (n_batches, n_cells, n_genes)
    ground_truth = results["ground_truth"]

    # Aggregate across cells (mean per batch)
    pred_mean = predictions.mean(axis=1)  # (n_batches, n_genes)
    gt_mean = ground_truth.mean(axis=1)

    # Compute gene-wise correlations (across samples)
    n_genes = pred_mean.shape[1]
    gene_correlations = []

    for gene_idx in range(n_genes):
        pred_gene = pred_mean[:, gene_idx]
        gt_gene = gt_mean[:, gene_idx]

        # Skip genes with no variance
        if np.std(pred_gene) < 1e-8 or np.std(gt_gene) < 1e-8:
            continue

        corr, _ = stats.pearsonr(pred_gene, gt_gene)
        if not np.isnan(corr):
            gene_correlations.append(corr)

    gene_correlations = np.array(gene_correlations)

    # Compute MSE in log1p space
    pred_log1p = np.log1p(predictions)
    gt_log1p = np.log1p(ground_truth)
    mse_log1p = np.mean((pred_log1p - gt_log1p) ** 2)

    # Compute MSE in raw space
    mse_raw = np.mean((predictions - ground_truth) ** 2)

    metrics = {
        "gene_correlation_mean": float(np.mean(gene_correlations)),
        "gene_correlation_median": float(np.median(gene_correlations)),
        "gene_correlation_std": float(np.std(gene_correlations)),
        "mse_log1p": float(mse_log1p),
        "mse_raw": float(mse_raw),
        "n_genes_evaluated": len(gene_correlations),
        "n_samples": len(results["perturbation_names"]),
    }

    return metrics


def create_h5ad_output(
    results: Dict[str, Any],
    gene_names: List[str],
    output_path: str,
    fold: Optional[int] = None,
) -> None:
    """Create h5ad file compatible with cell-eval.

    Creates an AnnData object with:
        - X: Predicted expression matrix (cells x genes)
        - obs: Cell metadata including perturbation labels
        - var: Gene metadata
        - layers['ground_truth']: Actual perturbed expression
        - layers['control']: Control expression used as context
    """
    import anndata as ad
    import pandas as pd

    predictions = results["predictions"]  # (n_batches, n_cells, n_genes)
    ground_truth = results["ground_truth"]
    control = results["control"]
    pert_names = results["perturbation_names"]

    # Flatten batch dimension: (n_batches, n_cells, n_genes) -> (n_total_cells, n_genes)
    n_batches, n_cells_per_batch, n_genes = predictions.shape
    n_total_cells = n_batches * n_cells_per_batch

    pred_flat = predictions.reshape(n_total_cells, n_genes)
    gt_flat = ground_truth.reshape(n_total_cells, n_genes)
    ctrl_flat = control.reshape(n_total_cells, n_genes)

    # Create cell-level perturbation labels (same for all cells in a batch)
    cell_pert_labels = []
    cell_batch_ids = []
    for batch_idx, pert_name in enumerate(pert_names):
        cell_pert_labels.extend([pert_name] * n_cells_per_batch)
        cell_batch_ids.extend([batch_idx] * n_cells_per_batch)

    # Create observation DataFrame
    obs = pd.DataFrame({
        "perturbation": cell_pert_labels,
        "batch_id": cell_batch_ids,
        "cell_idx": list(range(n_cells_per_batch)) * n_batches,
    })
    obs.index = [f"cell_{i}" for i in range(n_total_cells)]

    # Create variable DataFrame
    var = pd.DataFrame(index=gene_names)
    var.index.name = "gene"

    # Create AnnData
    adata = ad.AnnData(
        X=pred_flat.astype(np.float32),
        obs=obs,
        var=var,
    )

    # Add layers
    adata.layers["ground_truth"] = gt_flat.astype(np.float32)
    adata.layers["control"] = ctrl_flat.astype(np.float32)

    # Add metadata
    adata.uns["perturb_mean"] = {
        "n_batches": n_batches,
        "n_cells_per_batch": n_cells_per_batch,
        "unique_perturbations": list(set(pert_names)),
        "fold": fold,
        "model": "PerturbMean",
    }

    # Save
    logging.info(f"Saving predictions to {output_path}")
    adata.write_h5ad(output_path)
    logging.info(f"Saved AnnData with shape {adata.shape}")


def create_aggregated_h5ad(
    results: Dict[str, Any],
    gene_names: List[str],
    output_path: str,
    fold: Optional[int] = None,
) -> str:
    """Create aggregated h5ad with one row per perturbation (mean across cells).

    This format is often preferred for cell-eval as it compares perturbation-level
    effects rather than individual cells.
    """
    import anndata as ad
    import pandas as pd

    predictions = results["predictions"]  # (n_batches, n_cells, n_genes)
    ground_truth = results["ground_truth"]
    pert_names = results["perturbation_names"]

    # Aggregate across cells within each batch
    pred_agg = predictions.mean(axis=1)  # (n_batches, n_genes)
    gt_agg = ground_truth.mean(axis=1)

    # Group by perturbation name and average (in case of multiple batches per perturbation)
    pert_to_pred = {}
    pert_to_gt = {}
    for i, pert_name in enumerate(pert_names):
        if pert_name not in pert_to_pred:
            pert_to_pred[pert_name] = []
            pert_to_gt[pert_name] = []
        pert_to_pred[pert_name].append(pred_agg[i])
        pert_to_gt[pert_name].append(gt_agg[i])

    # Average across batches for each perturbation
    unique_perts = sorted(pert_to_pred.keys())
    pred_final = np.array([np.mean(pert_to_pred[p], axis=0) for p in unique_perts])
    gt_final = np.array([np.mean(pert_to_gt[p], axis=0) for p in unique_perts])

    # Create observation DataFrame
    obs = pd.DataFrame({
        "perturbation": unique_perts,
    })
    obs.index = unique_perts

    # Create variable DataFrame
    var = pd.DataFrame(index=gene_names)
    var.index.name = "gene"

    # Create AnnData
    adata = ad.AnnData(
        X=pred_final.astype(np.float32),
        obs=obs,
        var=var,
    )
    adata.layers["ground_truth"] = gt_final.astype(np.float32)

    # Add metadata
    adata.uns["perturb_mean"] = {
        "unique_perturbations": unique_perts,
        "fold": fold,
        "model": "PerturbMean",
        "aggregation": "mean",
    }

    # Save
    output_agg_path = output_path.replace(".h5ad", "_aggregated.h5ad")
    logging.info(f"Saving aggregated predictions to {output_agg_path}")
    adata.write_h5ad(output_agg_path)
    logging.info(f"Saved aggregated AnnData with shape {adata.shape}")

    return output_agg_path


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

    # Run inference on test set
    logging.info("Running inference on test set...")
    results = run_inference(
        model=model,
        dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logging.info(f"Generated predictions for {len(results['perturbation_names'])} batches")

    # Compute metrics
    logging.info("Computing metrics...")
    metrics = compute_metrics(results)

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

    # Save h5ad
    h5ad_path = os.path.join(
        args.output_dir, f"predictions_fold_{args.fold}_{fold_cell_line}.h5ad"
    )
    create_h5ad_output(
        results=results,
        gene_names=gene_names,
        output_path=h5ad_path,
        fold=args.fold,
    )

    # Also create aggregated output if requested
    if args.aggregated:
        create_aggregated_h5ad(
            results=results,
            gene_names=gene_names,
            output_path=h5ad_path,
            fold=args.fold,
        )

    logging.info("Inference complete!")

    # Print cell-eval usage hint
    print("\n" + "=" * 60)
    print("To evaluate with cell-eval:")
    print("=" * 60)
    print(f"cell-eval run -i {h5ad_path} \\")
    print(f"    --pert-col perturbation \\")
    print(f"    --control-pert {args.control_label}")
    print("=" * 60)


if __name__ == "__main__":
    main()
