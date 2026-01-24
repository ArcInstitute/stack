#!/usr/bin/env python3
"""Inference script for PerturbationExpert that outputs h5ad for cell-eval.

This script generates perturbation predictions and saves them in h5ad format
compatible with cell-eval (https://github.com/ArcInstitute/cell-eval).

Usage:
    python scripts/inference_perturbation.py \
        --checkpoint /path/to/perturbation_expert.ckpt \
        --data-path /data/replogle_concat.h5ad \
        --output predictions.h5ad \
        --fold 0
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


def load_perturbation_expert(checkpoint_path: str, device: torch.device):
    """Load trained PerturbationExpert from checkpoint."""
    import pytorch_lightning as pl
    from stack.training.perturbation_lightning import LightningPerturbationExpert

    # Load checkpoint
    model = LightningPerturbationExpert.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def run_inference(
    model,
    dataset,
    device: torch.device,
    num_sampling_steps: int = 100,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run inference on the dataset and collect predictions.

    Returns:
        Dictionary containing:
            - predictions: np.array of predicted expression (n_samples, n_cells, n_genes)
            - ground_truth: np.array of actual perturbed expression
            - control: np.array of control expression used as context
            - perturbation_names: List of perturbation names
            - perturbation_ids: np.array of perturbation IDs
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_predictions = []
    all_ground_truth = []
    all_control = []
    all_pert_names = []
    all_pert_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            control = batch["control_features"].to(device)
            perturbed = batch["perturbed_features"].to(device)
            pert_ids = batch["perturbation_id"].to(device)
            pert_names = batch["perturbation_name"]

            # Generate predictions
            result = model.model.sample(
                control_features=control,
                perturbation_ids=pert_ids,
                num_steps=num_sampling_steps,
            )

            # Collect results
            all_predictions.append(result["perturbed"].cpu().numpy())
            all_ground_truth.append(perturbed.cpu().numpy())
            all_control.append(control.cpu().numpy())
            all_pert_names.extend(pert_names)
            all_pert_ids.append(pert_ids.cpu().numpy())

    return {
        "predictions": np.concatenate(all_predictions, axis=0),
        "ground_truth": np.concatenate(all_ground_truth, axis=0),
        "control": np.concatenate(all_control, axis=0),
        "perturbation_names": all_pert_names,
        "perturbation_ids": np.concatenate(all_pert_ids, axis=0),
    }


def create_h5ad_output(
    results: Dict[str, Any],
    gene_names: List[str],
    output_path: str,
    include_ground_truth: bool = True,
) -> None:
    """Create h5ad file compatible with cell-eval.

    Creates an AnnData object with:
        - X: Predicted expression matrix (cells x genes)
        - obs: Cell metadata including perturbation labels
        - var: Gene metadata
        - layers['ground_truth']: Actual perturbed expression (optional)
        - layers['control']: Control expression used as context (optional)
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
    if include_ground_truth:
        adata.layers["ground_truth"] = gt_flat.astype(np.float32)
        adata.layers["control"] = ctrl_flat.astype(np.float32)

    # Add metadata
    adata.uns["perturbation_expert"] = {
        "n_batches": n_batches,
        "n_cells_per_batch": n_cells_per_batch,
        "unique_perturbations": list(set(pert_names)),
    }

    # Save
    logging.info(f"Saving predictions to {output_path}")
    adata.write_h5ad(output_path)
    logging.info(f"Saved AnnData with shape {adata.shape}")


def create_aggregated_h5ad(
    results: Dict[str, Any],
    gene_names: List[str],
    output_path: str,
    aggregation: str = "mean",
) -> None:
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
    if aggregation == "mean":
        pred_agg = predictions.mean(axis=1)  # (n_batches, n_genes)
        gt_agg = ground_truth.mean(axis=1)
    elif aggregation == "median":
        pred_agg = np.median(predictions, axis=1)
        gt_agg = np.median(ground_truth, axis=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

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

    # Save
    output_agg_path = output_path.replace(".h5ad", f"_aggregated_{aggregation}.h5ad")
    logging.info(f"Saving aggregated predictions to {output_agg_path}")
    adata.write_h5ad(output_agg_path)
    logging.info(f"Saved aggregated AnnData with shape {adata.shape}")

    return output_agg_path


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with PerturbationExpert and output h5ad for cell-eval"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PerturbationExpert checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to h5ad file with perturbation data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for predictions h5ad file",
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
        "--n-cells",
        type=int,
        default=128,
        help="Number of cells per sample",
    )

    # Inference configuration
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=100,
        help="Number of Euler integration steps for sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "val", "train"],
        default="test",
        help="Which split to run inference on",
    )

    # Output configuration
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Don't include ground truth in output",
    )
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
        help="Device to run inference on",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = load_perturbation_expert(args.checkpoint, device)

    # Setup dataset
    logging.info(f"Loading dataset from {args.data_path}")
    from stack.data.perturbation import PerturbationDataset, PerturbationDatasetConfig

    config = PerturbationDatasetConfig(
        data_path=args.data_path,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
        n_cells=args.n_cells,
        genelist_path=args.genelist_path,
        split_toml_dir=args.split_toml_dir,
        fold=args.fold,
    )

    dataset = PerturbationDataset(
        config=config,
        mode=args.mode,
    )

    logging.info(f"Dataset has {len(dataset)} samples")
    logging.info(f"Number of test perturbations: {len(dataset.test_perts)}")

    # Get gene names
    if dataset.target_genes is not None:
        gene_names = dataset.target_genes
    else:
        gene_names = list(dataset.gene_names)

    # Run inference
    logging.info("Running inference...")
    results = run_inference(
        model=model,
        dataset=dataset,
        device=device,
        num_sampling_steps=args.num_sampling_steps,
        batch_size=args.batch_size,
    )

    logging.info(f"Generated predictions for {len(results['perturbation_names'])} batches")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save h5ad
    create_h5ad_output(
        results=results,
        gene_names=gene_names,
        output_path=args.output,
        include_ground_truth=not args.no_ground_truth,
    )

    # Also create aggregated output if requested
    if args.aggregated:
        create_aggregated_h5ad(
            results=results,
            gene_names=gene_names,
            output_path=args.output,
            aggregation="mean",
        )

    logging.info("Inference complete!")

    # Print cell-eval usage hint
    print("\n" + "=" * 60)
    print("To evaluate with cell-eval:")
    print("=" * 60)
    print(f"cell-eval run -i {args.output} \\")
    print(f"    --pert-col perturbation \\")
    print(f"    --control-pert {args.control_label}")
    print("=" * 60)


if __name__ == "__main__":
    main()
