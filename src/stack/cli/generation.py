#!/usr/bin/env python3
"""Generate in-context predictions for a test AnnData using donor-specific base data."""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from scipy.sparse import issparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

LOGGER = logging.getLogger("stack.generation")
_DEPS: dict | None = None


def _ensure_deps():
    """Lazy import heavy dependencies to keep ``--help`` responsive."""

    global _DEPS
    if _DEPS is not None:
        return _DEPS

    import anndata as ad  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import torch  # type: ignore
    from scipy.sparse import issparse  # type: ignore

    from stack.data.training.datasets import load_gene_list
    from ..model_loading import load_model_from_checkpoint

    _DEPS = {
        "ad": ad,
        "np": np,
        "pd": pd,
        "torch": torch,
        "issparse": issparse,
        "load_gene_list": load_gene_list,
        "load_model_from_checkpoint": load_model_from_checkpoint,
    }
    return _DEPS


def _load_adata(
    source: str | Path | ad.AnnData,
    *,
    target_genes: Optional[Sequence[str]] = None,
    gene_name_col: Optional[str] = None,
    backed: Optional[str] = None,
    align_to_target: bool = True,
) -> ad.AnnData:
    deps = _ensure_deps()
    ad = deps["ad"]
    if isinstance(source, ad.AnnData):
        adata = source
        if adata.isbacked:
            adata = adata.to_memory()
        else:
            adata = adata.copy()
    else:
        LOGGER.info("Loading AnnData from %s", source)
        adata = ad.read_h5ad(str(source), backed=backed)

    if align_to_target and target_genes:
        adata = _align_genes_to_target_list(adata, target_genes, gene_name_col)
    return adata


def _extract_gene_names(adata: ad.AnnData, gene_name_col: Optional[str]) -> np.ndarray:
    var_df = adata.raw.var if adata.raw is not None else adata.var
    if gene_name_col and gene_name_col in var_df.columns:
        names = var_df[gene_name_col].astype(str).str.upper().to_numpy()
    else:
        names = var_df.index.astype(str).str.upper().to_numpy()
    return names


def _align_genes_to_target_list(
    adata: ad.AnnData,
    target_genes: Optional[Sequence[str]],
    gene_name_col: Optional[str],
) -> ad.AnnData:
    deps = _ensure_deps()
    ad = deps["ad"]
    pd = deps["pd"]
    import scipy.sparse as sp
    
    if not target_genes:
        return adata.copy()
    
    # Convert target genes to list and create uppercase index for matching
    target_genes_list = list(target_genes)
    target_genes_upper = pd.Index([str(gene).upper() for gene in target_genes_list])
    
    # Extract source gene names using your existing function
    source_gene_names = _extract_gene_names(adata, gene_name_col)
    source_genes_upper = pd.Index([str(gene).upper() for gene in source_gene_names])
    
    # Find intersection and missing genes
    intersection = source_genes_upper.intersection(target_genes_upper)
    found_genes = len(intersection)
    
    if found_genes == 0:
        raise ValueError(f"No target genes found in the dataset. "
                       f"Dataset has {len(source_genes_upper)} genes, "
                       f"target list has {len(target_genes_list)} genes.")
    
    ratio = found_genes / len(target_genes_list)
    LOGGER.info(f"Gene alignment: {found_genes}/{len(target_genes_list)} ({ratio:.1%}) target genes found in dataset")
    
    # Determine which matrix to use (raw or regular)
    if adata.raw is not None:
        source_matrix = adata.raw.X
        source_var = adata.raw.var.copy()
    else:
        source_matrix = adata.X
        source_var = adata.var.copy()
    
    # Find genes that need to be added (missing in source data)
    genes_to_add = target_genes_upper.difference(source_genes_upper)
    needs_padding = len(genes_to_add) > 0
    
    if needs_padding:
        LOGGER.info(f"Adding {len(genes_to_add)} missing genes with zero values")
        
        # Create efficient empty sparse matrix for padding
        padding_mtx = sp.csr_matrix((adata.n_obs, len(genes_to_add)), 
                                   dtype=source_matrix.dtype)
        
        # Create padding AnnData
        adata_padding = ad.AnnData(
            X=padding_mtx,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=genes_to_add)
        )
        # Create temporary adata with uppercase var_names for concat
        adata_temp = ad.AnnData(
            X=source_matrix,
            obs=adata.obs,
            var=pd.DataFrame(index=source_genes_upper)
        )
        # Concatenate original data with padding
        adata_combined = ad.concat(
            [adata_temp, adata_padding],
            axis=1,  # Concatenate along gene axis
            join="outer",
            merge="unique"
        )
    else:
        # No padding needed, just create adata with uppercase var_names
        adata_combined = ad.AnnData(
            X=source_matrix,
            obs=adata.obs,
            var=pd.DataFrame(index=source_genes_upper)
        )
    
    # Reorder genes to match target gene order (this also removes extra genes)
    if not target_genes_upper.equals(adata_combined.var_names):
        adata_combined = adata_combined[:, target_genes_upper].copy()
    
    # Create final var dataframe with original target gene names (not uppercased)
    aligned_var = pd.DataFrame(index=target_genes_list)
    aligned_var.index.name = source_var.index.name
    
    # If gene_name_col exists, add it to the var dataframe
    if gene_name_col:
        aligned_var[gene_name_col] = target_genes_list
    
    # Create final aligned adata
    aligned_adata = ad.AnnData(
        X=adata_combined.X,
        obs=adata.obs.copy(),
        var=aligned_var
    )
    return aligned_adata


def _sanitize_split_value(split_value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(split_value))
    return safe or "split"


def _select_split_values(
    base_adata: ad.AnnData,
    split_column: str,
    requested_splits: Optional[Iterable[str]],
) -> List[str]:
    deps = _ensure_deps()
    np = deps["np"]
    if split_column not in base_adata.obs.columns:
        raise ValueError(
            f"Split column '{split_column}' not found in base AnnData. Available columns: {list(base_adata.obs.columns)}"
        )

    available = base_adata.obs[split_column].astype(str)
    if requested_splits:
        splits = []
        for value in requested_splits:
            mask = available == str(value)
            if not np.any(mask):
                LOGGER.warning("Split value '%s' not found in column '%s'", value, split_column)
                continue
            splits.append(str(value))
        if not splits:
            raise ValueError("None of the requested split values were found in the base AnnData.")
        return splits

    return sorted(available.unique())


def _prepare_base_subset(
    base_adata: ad.AnnData,
    split_column: str,
    split_value: str,
    *,
    target_genes: Optional[Sequence[str]],
    gene_name_col: Optional[str],
) -> ad.AnnData:
    deps = _ensure_deps()
    np = deps["np"]
    mask = base_adata.obs[split_column].astype(str) == split_value
    if not np.any(mask):
        raise ValueError(f"No cells found for split value '{split_value}'.")

    subset_view = base_adata[mask]
    subset = _load_adata(
        subset_view,
        target_genes=target_genes,
        gene_name_col=gene_name_col,
        align_to_target=bool(target_genes),
    )
    return subset


def _run_incontext_generation(
    model: torch.nn.Module,
    base_adata: ad.AnnData,
    test_adata: ad.AnnData,
    *,
    genelist_path: str,
    gene_name_col: Optional[str],
    prompt_ratio: float,
    context_ratio: float,
    context_ratio_min: float,
    mask_rate: float,
    mode: str,
    num_steps: Optional[int],
    batch_size: int,
    num_workers: int,
    random_seed: Optional[int],
    show_progress: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    outputs = model.get_incontext_generation(
        base_adata_or_path=base_adata,
        test_adata_or_path=test_adata,
        genelist_path=genelist_path,
        prompt_ratio=prompt_ratio,
        context_ratio=context_ratio,
        context_ratio_min=context_ratio_min,
        mask_rate=mask_rate,
        mode=mode,
        num_steps=num_steps,
        gene_name_col=gene_name_col,
        batch_size=batch_size,
        show_progress=show_progress,
        num_workers=num_workers,
        random_seed=random_seed,
    )

    if isinstance(outputs, tuple):
        predictions, test_logit = outputs
    else:
        predictions, test_logit = outputs, None
    return predictions, test_logit


def generate(
    checkpoint_path: str,
    base_adata_path: str,
    test_adata_path: str,
    genelist_path: str,
    *,
    split_column: str,
    split_values: Optional[Sequence[str]] = None,
    gene_name_col: Optional[str] = None,
    prompt_ratio: float = 0.25,
    context_ratio: float = 0.4,
    context_ratio_min: float = 0.2,
    mask_rate: float = 1.0,
    num_steps: Optional[int] = None,
    mode: str = "vanilla",
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: Optional[int] = None,
    device: str = "auto",
    show_progress: bool = False,
    on_generation: Optional[Callable[[str, ad.AnnData], None]] = None,
) -> Dict[str, ad.AnnData]:
    """Run in-context generation for the specified donor splits.

    When ``on_generation`` is provided each split is passed to the callback as soon
    as it is computed, avoiding the need to hold all generated AnnData objects in
    memory simultaneously. The returned dictionary will remain empty in that
    streaming mode. Without a callback the results are accumulated and returned.
    """

    deps = _ensure_deps()
    ad = deps["ad"]
    np = deps["np"]
    load_gene_list = deps["load_gene_list"]
    load_model_from_checkpoint = deps["load_model_from_checkpoint"]

    model = load_model_from_checkpoint(checkpoint_path, model_class="ICL_FinetunedModel")
    target_genes: Optional[List[str]] = load_gene_list(genelist_path) if genelist_path else None
    LOGGER.info("Loading base AnnData from %s in backed mode", base_adata_path)
    base_adata = ad.read_h5ad(base_adata_path, backed="r")
    test_adata = _load_adata(
        test_adata_path,
        target_genes=target_genes,
        gene_name_col=gene_name_col,
    )

    splits = _select_split_values(base_adata, split_column, split_values)
    LOGGER.info("Running in-context generation for %d donor(s)", len(splits))

    generations: Dict[str, ad.AnnData] = {}
    for split_value in splits:
        LOGGER.info("Processing split '%s'", split_value)
        base_subset = _prepare_base_subset(
            base_adata,
            split_column,
            split_value,
            target_genes=target_genes,
            gene_name_col=gene_name_col,
        )

        predictions, test_logit = _run_incontext_generation(
            model,
            base_subset,
            test_adata,
            genelist_path=genelist_path,
            gene_name_col=gene_name_col,
            prompt_ratio=prompt_ratio,
            context_ratio=context_ratio,
            context_ratio_min=context_ratio_min,
            mask_rate=mask_rate,
            mode=mode,
            num_steps=num_steps,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
            show_progress=show_progress,
        )


        pred_adata = ad.AnnData(
            X=predictions,
            obs=test_adata.obs.copy(),
            var=test_adata.var.copy(),
        )
        if test_logit is not None:
            pred_adata.obs["gen_logit"] = np.asarray(test_logit)

        if on_generation is not None:
            on_generation(split_value, pred_adata)
        else:
            generations[split_value] = pred_adata

    return generations


def save_generations(
    generations: Dict[str, ad.AnnData],
    output_dir: Path,
    *,
    concatenate: bool = False,
) -> None:
    deps = _ensure_deps()
    ad = deps["ad"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not generations:
        LOGGER.warning("No generations to save.")
        return

    if concatenate:
        annotated_adatas = []
        for split_value, adata in generations.items():
            adata_copy = adata.copy()
            adata_copy.obs = adata_copy.obs.copy()
            adata_copy.obs["split_value"] = split_value
            annotated_adatas.append(adata_copy)

        concatenated = ad.concat(annotated_adatas, axis=0, join="outer")
        output_path = output_dir / "concatenated.h5ad"
        LOGGER.info("Writing concatenated generations to %s", output_path)
        concatenated.write_h5ad(output_path)
        return

    for split_value, adata in generations.items():
        _write_generation(output_dir, split_value, adata)


def _write_generation(output_dir: Path, split_value: str, adata: ad.AnnData) -> None:
    _ensure_deps()
    safe_name = _sanitize_split_value(split_value)
    output_path = output_dir / f"{safe_name}.h5ad"
    LOGGER.info("Writing generation for split '%s' to %s", split_value, output_path)
    adata.write_h5ad(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run in-context generation using a StateICL checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained Lightning checkpoint (.ckpt)")
    parser.add_argument("--base-adata", required=True, help="Path to the base AnnData (.h5ad) containing donor cells")
    parser.add_argument("--test-adata", required=True, help="Path to the test AnnData (.h5ad) to generate for")
    parser.add_argument("--genelist", required=True, help="Path to the pickled gene list used during training")
    parser.add_argument("--output-dir", required=True, help="Directory where generated AnnData files will be saved")
    parser.add_argument("--split-column", required=True, help="Column in base AnnData.obs specifying donor identifiers")
    parser.add_argument(
        "--split-values",
        nargs="*",
        default=None,
        help="Optional subset of donor identifiers to process (defaults to all unique values)",
    )
    parser.add_argument(
        "--gene-name-col",
        default=None,
        help="Optional column in adata.var/raw.var containing gene symbols for alignment",
    )
    parser.add_argument("--prompt-ratio", type=float, default=0.25, help="Prompt ratio passed to in-context generation")
    parser.add_argument("--context-ratio", type=float, default=0.4, help="Context ratio passed to in-context generation")
    parser.add_argument("--context-ratio-min", type=float, default=0.2, help="Min value of context ratio")
    parser.add_argument("--mask-rate", type=float, default=1.0, help="Mask rate used during in-context generation")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of iterative generation steps (defaults to model heuristic if omitted)",
    )
    parser.add_argument(
        "--mode",
        default="mdm",
        help="Generation mode to use when calling the model",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--random-seed", type=int, default=0, help="Optional seed for reproducible sampling")
    parser.add_argument("--device", default="auto", help="Target device (cuda, cpu, or auto)")
    parser.add_argument("--show-progress", action="store_true", help="Display a tqdm progress bar")
    parser.add_argument(
        "--concatenate",
        action="store_true",
        help="Concatenate generations across splits into a single file",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    _ensure_deps()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if parsed.concatenate:
        generations = generate(
            checkpoint_path=parsed.checkpoint,
            base_adata_path=parsed.base_adata,
            test_adata_path=parsed.test_adata,
            genelist_path=parsed.genelist,
            split_column=parsed.split_column,
            split_values=parsed.split_values,
            gene_name_col=parsed.gene_name_col,
            prompt_ratio=parsed.prompt_ratio,
            context_ratio=parsed.context_ratio,
            context_ratio_min=parsed.context_ratio_min,
            mask_rate=parsed.mask_rate,
            num_steps=parsed.num_steps,
            mode=parsed.mode,
            batch_size=parsed.batch_size,
            num_workers=parsed.num_workers,
            random_seed=parsed.random_seed,
            device=parsed.device,
            show_progress=parsed.show_progress,
        )
        save_generations(
            generations,
            output_dir,
            concatenate=True,
        )
        return

    def _stream_save(split_value: str, adata: ad.AnnData) -> None:
        _write_generation(output_dir, split_value, adata)

    generate(
        checkpoint_path=parsed.checkpoint,
        base_adata_path=parsed.base_adata,
        test_adata_path=parsed.test_adata,
        genelist_path=parsed.genelist,
        split_column=parsed.split_column,
        split_values=parsed.split_values,
        gene_name_col=parsed.gene_name_col,
        prompt_ratio=parsed.prompt_ratio,
        context_ratio=parsed.context_ratio,
        context_ratio_min=parsed.context_ratio_min,
        mask_rate=parsed.mask_rate,
        num_steps=parsed.num_steps,
        mode=parsed.mode,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        random_seed=parsed.random_seed,
        device=parsed.device,
        show_progress=parsed.show_progress,
        on_generation=_stream_save,
    )


if __name__ == "__main__":
    main()
