"""Data loading and preprocessing helpers for StateICL."""
from .datasets import *  # noqa: F401,F403
from .gene_processing import filter_gene_names, get_gene_names_from_h5, safe_decode_array
from .h5_manager import (
    H5HandleManager,
    get_h5_handle,
    reset_h5_handle_pool,
    worker_init_fn,
)
from .hvg import compute_analytic_pearson_residuals, compute_hvg_union
from . import training as training_data
from . import finetuning as finetuning_data
from . import perturbation as perturbation_data

__all__ = sorted(
    {
        "filter_gene_names",
        "get_gene_names_from_h5",
        "safe_decode_array",
        "H5HandleManager",
        "get_h5_handle",
        "reset_h5_handle_pool",
        "worker_init_fn",
        "compute_analytic_pearson_residuals",
        "compute_hvg_union",
        "training_data",
        "finetuning_data",
        "perturbation_data",
    }
    | {name for name in globals().keys() if not name.startswith("_")}
)
