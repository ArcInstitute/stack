#!/usr/bin/env python3
"""Prepare h5ad files for cell-eval from PerturbationExpert predictions.

Splits predictions h5ad into separate pred and real files with control row.
"""

import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Prepare h5ad for cell-eval")
    parser.add_argument("--input", "-i", required=True, help="Input h5ad with X=pred, layers['ground_truth']=real")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--control-label", default="non-targeting", help="Control perturbation label")
    parser.add_argument("--data-path", help="Original data h5ad to get control expression (optional)")
    parser.add_argument("--genelist-path", help="Gene list pickle for subsetting (optional)")
    args = parser.parse_args()

    import anndata as ad
    import pandas as pd
    import os
    import pickle

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions
    logging.info(f"Loading {args.input}")
    adata = ad.read_h5ad(args.input)

    pred_matrix = adata.X
    gt_matrix = adata.layers["ground_truth"]
    perts = adata.obs["perturbation"].tolist()
    gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)

    logging.info(f"Loaded {len(perts)} perturbations, {n_genes} genes")

    # Convert to log1p if data appears to be raw counts
    if pred_matrix.max() > 15.0:
        logging.info("Converting predictions to log1p space (detected raw counts)")
        pred_matrix = np.log1p(pred_matrix)
    if gt_matrix.max() > 15.0:
        logging.info("Converting ground truth to log1p space (detected raw counts)")
        gt_matrix = np.log1p(gt_matrix)

    # Clamp negative values (log1p cannot be negative)
    if pred_matrix.min() < 0:
        logging.warning(f"Clamping {(pred_matrix < 0).sum()} negative prediction values to 0")
        pred_matrix = np.maximum(pred_matrix, 0)
    if gt_matrix.min() < 0:
        logging.warning(f"Clamping {(gt_matrix < 0).sum()} negative ground truth values to 0")
        gt_matrix = np.maximum(gt_matrix, 0)

    # Get control expression
    if args.data_path:
        logging.info(f"Loading control from {args.data_path}")
        adata_full = ad.read_h5ad(args.data_path, backed="r")

        # Find control cells
        if "gene" in adata_full.obs.columns:
            control_mask = adata_full.obs["gene"] == args.control_label
        else:
            control_mask = adata_full.obs["perturbation"] == args.control_label

        control_indices = np.where(control_mask)[0]
        logging.info(f"Found {len(control_indices)} control cells")

        # Load control expression
        control_expr = adata_full.X[control_indices[:1000], :]  # Sample up to 1000
        if hasattr(control_expr, "toarray"):
            control_expr = control_expr.toarray()

        # Apply gene mapping if genelist provided
        if args.genelist_path:
            with open(args.genelist_path, "rb") as f:
                target_genes = pickle.load(f)

            gene_to_idx = {g.upper(): i for i, g in enumerate(adata_full.var_names)}
            mapped_control = np.zeros((control_expr.shape[0], len(target_genes)), dtype=np.float32)

            for tgt_idx, gene in enumerate(target_genes):
                if gene.upper() in gene_to_idx:
                    mapped_control[:, tgt_idx] = control_expr[:, gene_to_idx[gene.upper()]]

            control_expr = mapped_control

        control_mean = control_expr.mean(axis=0).astype(np.float32)
        # Convert to log1p if needed
        if control_mean.max() > 15.0:
            logging.info("Converting control to log1p space")
            control_mean = np.log1p(control_mean)
        adata_full.file.close()
    else:
        # Use zeros if no data path (not ideal but works)
        logging.warning("No data-path provided, using zeros for control")
        control_mean = np.zeros(n_genes, dtype=np.float32)

    # Add control row
    all_perts = [args.control_label] + perts
    pred_with_ctrl = np.vstack([control_mean.reshape(1, -1), pred_matrix])
    gt_with_ctrl = np.vstack([control_mean.reshape(1, -1), gt_matrix])

    # Create obs
    obs = pd.DataFrame({"perturbation": all_perts})
    obs.index = all_perts

    var = pd.DataFrame(index=gene_names)
    var.index.name = "gene"

    # Save predictions
    pred_path = os.path.join(args.output_dir, "pred.h5ad")
    adata_pred = ad.AnnData(X=pred_with_ctrl.astype(np.float32), obs=obs.copy(), var=var.copy())
    adata_pred.write_h5ad(pred_path)
    logging.info(f"Saved predictions to {pred_path} with shape {adata_pred.shape}")

    # Save ground truth
    real_path = os.path.join(args.output_dir, "real.h5ad")
    adata_real = ad.AnnData(X=gt_with_ctrl.astype(np.float32), obs=obs.copy(), var=var.copy())
    adata_real.write_h5ad(real_path)
    logging.info(f"Saved ground truth to {real_path} with shape {adata_real.shape}")

    logging.info("Done!")


if __name__ == "__main__":
    main()
