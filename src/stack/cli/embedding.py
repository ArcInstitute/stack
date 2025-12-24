#!/usr/bin/env python3
"""Utility script to extract embeddings from a trained StateICL checkpoint."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch


LOGGER = logging.getLogger("stack.embedding")
_DEPS: dict | None = None


def _ensure_deps():
    """Lazy import heavy runtime dependencies to keep ``--help`` fast."""

    global _DEPS
    if _DEPS is not None:
        return _DEPS

    import anndata as ad  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import torch  # type: ignore

    from ..model_loading import load_model_from_checkpoint

    _DEPS = {
        "ad": ad,
        "np": np,
        "pd": pd,
        "torch": torch,
        "load_model_from_checkpoint": load_model_from_checkpoint,
    }
    return _DEPS


def _resolve_device(device_arg: str) -> torch.device:
    deps = _ensure_deps()
    torch = deps["torch"]
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    deps = _ensure_deps()
    load_model_from_checkpoint = deps["load_model_from_checkpoint"]
    return load_model_from_checkpoint(checkpoint_path, device=device)


def extract_embeddings(
    checkpoint_path: str,
    adata_path: str,
    genelist_path: str,
    *,
    gene_name_col: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: Optional[int] = None,
    max_samples: Optional[int] = None,
    device: str = "auto",
    show_progress: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract embeddings for all cells in ``adata_path`` using the trained model."""

    deps = _ensure_deps()
    np = deps["np"]
    torch = deps["torch"]
    model = _load_model(checkpoint_path, _resolve_device(device))

    dataloader_kwargs = {}
    if max_samples is not None:
        dataloader_kwargs["max_samples"] = max_samples
    if random_seed is not None:
        dataloader_kwargs["random_state"] = random_seed

    embeddings, dataset_embeddings = model.get_latent_representation(
        adata_path=adata_path,
        genelist_path=genelist_path,
        gene_name_col=gene_name_col,
        batch_size=batch_size,
        show_progress=show_progress,
        num_workers=num_workers,
        **dataloader_kwargs,
    )
    return embeddings, dataset_embeddings


def _build_obs_dataframe(obs_source: Optional[str], n_cells: int) -> pd.DataFrame:
    deps = _ensure_deps()
    ad = deps["ad"]
    pd = deps["pd"]
    if obs_source is None:
        index = [f"cell_{idx}" for idx in range(n_cells)]
        return pd.DataFrame(index=index)

    LOGGER.info("Loading observation metadata from %s", obs_source)
    source_adata = ad.read_h5ad(obs_source)
    if source_adata.n_obs < n_cells:
        raise ValueError(
            f"Observation source contains {source_adata.n_obs} cells but embeddings cover {n_cells}."
        )
    return source_adata.obs.iloc[:n_cells].copy()


def save_embeddings(
    embeddings: np.ndarray,
    output_path: Path,
    *,
    obs_source: Optional[str] = None,
) -> None:
    """Persist embeddings to disk in either NumPy or AnnData format."""

    deps = _ensure_deps()
    ad = deps["ad"]
    np = deps["np"]
    pd = deps["pd"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".h5ad":
        obs = _build_obs_dataframe(obs_source, embeddings.shape[0])
        var_index = [f"latent_{idx}" for idx in range(embeddings.shape[1])]
        embed_adata = ad.AnnData(X=embeddings, obs=obs, var=pd.DataFrame(index=var_index))
        LOGGER.info("Writing embeddings to %s", output_path)
        embed_adata.write_h5ad(output_path)
    else:
        LOGGER.info("Saving embeddings array to %s", output_path)
        np.save(output_path, embeddings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract latent embeddings from a StateICL checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained Lightning checkpoint (.ckpt)")
    parser.add_argument("--adata", required=True, help="Path to the AnnData file (.h5ad) to embed")
    parser.add_argument("--genelist", required=True, help="Path to the pickled gene list used during training")
    parser.add_argument("--output", required=True, help="Destination file (.npy or .h5ad) for embeddings")
    parser.add_argument(
        "--gene-name-col",
        default=None,
        help="Optional column in adata.var/raw.var containing gene symbols for alignment",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--random-seed", type=int, default=0, help="Optional seed for deterministic sampling")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of sampled batches")
    parser.add_argument("--device", default="auto", help="Target device (cuda, cpu, or auto)")
    parser.add_argument("--show-progress", action="store_true", help="Display a tqdm progress bar")
    parser.add_argument(
        "--obs-source",
        type=str,
        default=None,
        help="Optional .h5ad file with observation metadata to attach to the embeddings",
    )
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    embeddings, dataset_embeddings = extract_embeddings(
        checkpoint_path=parsed.checkpoint,
        adata_path=parsed.adata,
        genelist_path=parsed.genelist,
        gene_name_col=parsed.gene_name_col,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        random_seed=parsed.random_seed,
        max_samples=parsed.max_samples,
        device=parsed.device,
        show_progress=parsed.show_progress,
    )

    output_path = Path(parsed.output)
    save_embeddings(
        embeddings,
        output_path,
        obs_source=parsed.obs_source,
    )


if __name__ == "__main__":
    main()
