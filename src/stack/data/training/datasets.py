from __future__ import annotations

import gc
import logging
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import psutil
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from ..gene_processing import filter_gene_names, get_gene_names_from_h5, safe_decode_array
from ..h5_manager import get_h5_handle, reset_h5_handle_pool, worker_init_fn
from ..hvg import compute_analytic_pearson_residuals, compute_hvg_union

log = logging.getLogger(__name__)

__all__ = [
    "DatasetConfig",
    "SimplifiedDatasetCache",
    "SimplifiedMultiDataset",
    "TestSamplerDataset",
    "compute_analytic_pearson_residuals",
    "compute_hvg_union",
    "create_train_val_test_datasets",
    "create_example_configs",
    "compute_and_save_hvg_union",
    "create_datasets_from_gene_list",
    "load_gene_list",
    "worker_init_fn",
    "reset_h5_handle_pool",
]


@dataclass
class DatasetConfig:
    """Simplified configuration for a single dataset."""

    path: str
    filter_organism: bool = True
    gene_name_col: Optional[str] = None
    
def load_gene_list(genelist_path: str) -> List[str]:
    """Load previously computed gene list from pickle file."""
    with open(genelist_path, 'rb') as f:
        return pickle.load(f)

class SimplifiedDatasetCache:
    """Simplified metadata cache for file-level dataset management."""
    _instances: dict[str, "SimplifiedDatasetCache"] = {}

    @classmethod
    def get_singleton(cls, cache_key: str, **kwargs) -> "SimplifiedDatasetCache":
        """Return one shared instance per cache key."""
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(**kwargs)
        return cls._instances[cache_key]
    
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        target_genes: List[str],
        cache_file: Optional[str] = None
    ):
        """Initialize the simplified cache."""
        self.dataset_configs = dataset_configs
        self.target_genes = target_genes
        self.n_genes = len(target_genes)
        
        log.info("Initializing Simplified Dataset Cache...")
        start_time = time.time()
        
        if cache_file and os.path.exists(cache_file):
            try:
                self._load_from_cache(cache_file)
                log.info(f"Loaded metadata from cache: {cache_file}")
            except Exception as e:
                log.warning(f"Failed to load cache, rebuilding: {e}")
                self._build_metadata()
                if cache_file:
                    self._save_to_cache(cache_file)
        else:
            self._build_metadata()
            if cache_file:
                self._save_to_cache(cache_file)
        
        load_time = time.time() - start_time
        log.info(f"Cache initialization completed in {load_time:.2f}s. Total files: {len(self.file_info)}")
    
    def _get_memory_usage(self):
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    def _save_to_cache(self, cache_file: str):
        """Save metadata to cache file"""
        try:
            cache_data = {
                'file_info': self.file_info,
                'target_genes': self.target_genes,
                'dataset_configs': self.dataset_configs,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            log.info(f"Saved metadata cache to {cache_file}")
        except Exception as e:
            log.exception(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_file: str):
        """Load metadata from cache file"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.file_info = cache_data['file_info']
        
        if cache_data['target_genes'] != self.target_genes:
            raise ValueError("Target genes in cache don't match current target genes")
        
        if len(cache_data['dataset_configs']) != len(self.dataset_configs):
            raise ValueError("Dataset configurations don't match cache")
    
    def _build_metadata(self):
        """Load metadata from all H5AD files"""
        self.file_info = []
        target_gene_set = set(self.target_genes)
        
        for config_idx, config in enumerate(self.dataset_configs):
            log.info(f"Loading metadata from dataset {config_idx + 1}/{len(self.dataset_configs)}: {config.path}")
            
            h5ad_files = list(Path(config.path).glob("*.h5ad")) + list(Path(config.path).glob("*.h5"))
            h5ad_files = sorted(list(set(h5ad_files)))
            
            for h5ad_file in h5ad_files:
                try:
                    log.info(f"Processing {h5ad_file.name}...")
                    
                    with h5py.File(h5ad_file, "r") as f:
                        # Apply organism filtering
                        if config.filter_organism:
                            obs = f["obs"]
                            if "organism" not in obs:
                                log.warning(f"'organism' column not found, skipping")
                                continue
                            
                            organism_ds = obs["organism"]
                            if "categories" in organism_ds:
                                organism_categories = safe_decode_array(organism_ds["categories"][:])
                                organism_codes = organism_ds["codes"][:]
                                organism_values = organism_categories[organism_codes]
                            else:
                                organism_values = safe_decode_array(organism_ds[:])
                            
                            human_mask = organism_values == 'Homo sapiens'
                            if not human_mask.any():
                                log.warning(f"No Homo sapiens cells found, skipping")
                                continue
                            
                            n_human_cells = human_mask.sum()
                        else:
                            if hasattr(f["X"], 'shape'):
                                n_cells = f["X"].shape[0]
                            else:
                                n_cells = f["X"].attrs['shape'][0]
                            human_mask = np.ones(n_cells, dtype=bool)
                            n_human_cells = n_cells
                        
                        # Get gene information
                        use_raw = "raw" in f and "var" in f["raw"]
                        gene_names = get_gene_names_from_h5(f, config.gene_name_col, use_raw=use_raw)
                        
                        if gene_names is None:
                            log.warning(f"Could not find gene names, skipping")
                            continue
                        
                        # Create gene mapping
                        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
                        gene_mapping = {}
                        found_genes = 0
                        for target_idx, gene in enumerate(self.target_genes):
                            if gene in gene_to_idx:
                                gene_mapping[target_idx] = gene_to_idx[gene]
                                found_genes += 1
                        
                        if found_genes == 0:
                            log.warning(f"No target genes found in {h5ad_file.name}, skipping")
                            continue
                        
                        # Get matrix info
                        if use_raw:
                            X_group = f["raw"]["X"]
                        else:
                            X_group = f["X"]
                        
                        is_sparse = isinstance(X_group, h5py.Group) and 'data' in X_group and 'indices' in X_group
                        
                        # Store file information
                        file_info = {
                            'path': str(h5ad_file),
                            'config_idx': config_idx,
                            'n_cells': n_human_cells,
                            'n_genes': len(gene_names),
                            'use_raw': use_raw,
                            'gene_mapping': gene_mapping,
                            'organism_mask': human_mask,
                            'is_sparse': is_sparse,
                            'found_genes': found_genes
                        }
                        self.file_info.append(file_info)
                        
                        log.info(f"Added file with {n_human_cells} cells, {found_genes}/{len(self.target_genes)} target genes")
                    
                except Exception as e:
                    log.exception(f"Error processing {h5ad_file.name}: {e}")
                    continue
        
        log.info(f"Loaded metadata for {len(self.file_info)} files")
    
    def load_expression_data_from_file(self, file_idx: int, local_indices: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED IO FUNCTION: Efficiently load specified rows from a single file using format-aware reading strategy.
        This version supports both CSR and CSC sparse formats with automatic detection.
        """
        file_info = self.file_info[file_idx]
        
        # Sort indices for better h5py reading efficiency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]

        try:
            f = get_h5_handle(file_info['path'])
            X_group = f["raw"]["X"] if file_info['use_raw'] else f["X"]
                
            # Get absolute row indices in the original HDF5 file
            human_indices = np.where(file_info['organism_mask'])[0]
            absolute_indices_to_load = human_indices[sorted_local_indices]

            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes), dtype=np.float32)
            gene_mapping = file_info['gene_mapping']
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))

            # Dynamic format detection based on encoding-type
            if file_info['is_sparse']:
                # Read encoding-type directly from HDF5 file attributes
                attrs = dict(X_group.attrs)
                encoding_type = attrs.get("encoding-type", "unknown")
                log.debug(f"Processing sparse matrix with encoding: {encoding_type}")
                
                if encoding_type == "csr_matrix":
                    # CSR format: efficient row slicing
                    log.debug(f"Using optimized CSR row reading for {len(absolute_indices_to_load)} rows")
                    
                    # Your existing correct CSR reading logic
                    indptr_h5 = X_group["indptr"]
                    data_h5 = X_group["data"]
                    indices_h5 = X_group["indices"]

                    # 1) Get all required indptr start and end points
                    starts = indptr_h5[absolute_indices_to_load]
                    ends = indptr_h5[absolute_indices_to_load + 1]
                    
                    # 2) Detect physical data block boundaries
                    block_breaks = np.where(ends[:-1] != starts[1:])[0] + 1
                    row_blocks = np.split(np.arange(len(absolute_indices_to_load)), block_breaks)

                    # 3) Read data block by block to minimize I/O operations
                    data_slices = []
                    indices_slices = []
                    
                    row_order = np.concatenate(row_blocks) if row_blocks else np.array([], dtype=int)
                    
                    for block in row_blocks:
                        if len(block) == 0: 
                            continue
                        
                        first_row_in_block_idx = block[0]
                        last_row_in_block_idx = block[-1]
                        
                        data_start = starts[first_row_in_block_idx]
                        data_end = ends[last_row_in_block_idx]
                        
                        # Single I/O operation to read entire contiguous block
                        data_block = data_h5[data_start:data_end]
                        indices_block = indices_h5[data_start:data_end]
                        
                        # Split large block back into individual rows
                        row_lengths = ends[block] - starts[block]
                        cuts = np.cumsum(row_lengths)
                        
                        if len(cuts) > 0:
                            data_slices.extend(np.split(data_block, cuts[:-1]))
                            indices_slices.extend(np.split(indices_block, cuts[:-1]))

                    # 4) Construct sparse matrix from collected slices
                    if not data_slices:
                        sparse_subset = sp.csr_matrix((len(local_indices), file_info['n_genes']), dtype=np.float32)
                    else:
                        all_data = np.concatenate(data_slices)
                        all_indices = np.concatenate(indices_slices)

                        new_indptr = np.zeros(len(local_indices) + 1, dtype=indptr_h5.dtype)
                        row_lengths_in_order = ends[row_order] - starts[row_order]
                        new_indptr[1:] = np.cumsum(row_lengths_in_order)
                        
                        sparse_subset = sp.csr_matrix(
                            (all_data, all_indices, new_indptr),
                            shape=(len(local_indices), file_info['n_genes']),
                            dtype=np.float32
                        )
                        
                    # 5) Select target gene columns and convert to dense
                    expr_subset = sparse_subset[:, source_indices_in_file].toarray()
                    mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)

                elif encoding_type == "csc_matrix":
                    # CSC format: column-oriented reading strategy
                    log.debug(f"Using CSC column-oriented reading for {len(absolute_indices_to_load)} rows")
                    
                    data_h5 = X_group["data"]
                    indices_h5 = X_group["indices"] 
                    indptr_h5 = X_group["indptr"]
                    
                    # Get actual matrix shape
                    # For CSC, indptr length is n_cols + 1, i.e., number of genes + 1
                    n_genes_in_file = len(indptr_h5) - 1
                    n_cells_in_file = file_info['n_cells']
                    
                    # Adaptive strategy selection
                    load_threshold = min(1000, n_cells_in_file * 0.1)  # 10% of cells or 1000, whichever is smaller
                    
                    if len(absolute_indices_to_load) <= load_threshold:
                        # Strategy A: Row-by-row reading (suitable for few rows)
                        log.debug(f"Using row-by-row CSC reading for {len(absolute_indices_to_load)} rows")
                        
                        # FIXED: Use np.isin for efficient and correct matching
                        result_matrix = np.zeros((len(absolute_indices_to_load), n_genes_in_file), dtype=np.float32)
                        
                        # Create mapping from absolute_index to result_matrix row number
                        abs_to_res_map = {abs_idx: i for i, abs_idx in enumerate(absolute_indices_to_load)}
                        
                        # Process each target gene (column)
                        for gene_idx in range(n_genes_in_file):
                            col_start = indptr_h5[gene_idx]
                            col_end = indptr_h5[gene_idx + 1]
                            
                            if col_end > col_start:
                                # Read all non-zero cell indices and values for this gene (column)
                                col_cell_indices = indices_h5[col_start:col_end]
                                col_values = data_h5[col_start:col_end]
                                
                                # Use np.isin to find which cells we need are present in this column
                                # isin_mask has the same length as col_cell_indices
                                isin_mask = np.isin(col_cell_indices, absolute_indices_to_load, assume_unique=True)
                                
                                if np.any(isin_mask):
                                    # Get absolute indices of cells that have values
                                    matched_abs_indices = col_cell_indices[isin_mask]
                                    # Get corresponding values
                                    matched_values = col_values[isin_mask]
                                    
                                    # Fill result matrix using the mapping we created earlier
                                    for abs_idx, value in zip(matched_abs_indices, matched_values):
                                        result_matrix_row = abs_to_res_map[abs_idx]
                                        result_matrix[result_matrix_row, gene_idx] = value
                    
                    else:
                        # Strategy B: Load full matrix then slice (suitable for many rows)
                        log.debug(f"Using full matrix load for {len(absolute_indices_to_load)} rows (threshold: {load_threshold})")
                        
                        # Load entire CSC matrix
                        data = data_h5[:]
                        indices = indices_h5[:]
                        indptr = indptr_h5[:]
                        
                        # Construct scipy CSC matrix (Note: this is transposed, genes x cells)
                        csc_matrix_transposed = sp.csc_matrix(
                            (data, indices, indptr), 
                            shape=(n_genes_in_file, n_cells_in_file)
                        )
                        
                        # Transpose to get normal cells x genes CSR matrix
                        csr_matrix = csc_matrix_transposed.T
                        
                        # Now we can efficiently perform row slicing
                        subset_sparse = csr_matrix[absolute_indices_to_load, :]
                        result_matrix = subset_sparse.toarray()
                    
                    # Map to target genes
                    mapped_matrix[:, target_indices_in_result] = result_matrix[:, source_indices_in_file].astype(np.float32)
                    
                else:
                    # Unknown or unsupported sparse format
                    log.warning(f"Unknown sparse encoding type '{encoding_type}' in {file_info['path']}")
                    log.warning("Attempting to infer format from indptr length")
                    
                    # Try generic sparse matrix reading
                    try:
                        if 'data' in X_group and 'indices' in X_group and 'indptr' in X_group:
                            # Looks like CSR/CSC format, but encoding type is unknown
                            # Try to infer format based on indptr length
                            indptr_len = len(X_group['indptr'])
                            
                            if indptr_len == file_info['n_cells'] + 1:
                                # Likely CSR format
                                log.debug("Guessing CSR format based on indptr length")
                                encoding_type = "csr_matrix"
                                # Process as CSR (reuse CSR logic above)
                                # For simplicity, we'll use the fallback approach
                                raise NotImplementedError("CSR format inference - please use explicit encoding-type")
                                
                            elif indptr_len == file_info['n_genes'] + 1:
                                # Likely CSC format  
                                log.debug("Guessing CSC format based on indptr length")
                                encoding_type = "csc_matrix"
                                # Process as CSC (reuse CSC logic above)
                                # For simplicity, we'll use the fallback approach
                                raise NotImplementedError("CSC format inference - please use explicit encoding-type")
                            else:
                                raise ValueError(f"Cannot determine sparse matrix format for {file_info['path']}")
                        else:
                            raise ValueError(f"Unknown sparse matrix structure in {file_info['path']}")
                            
                    except Exception as e:
                        log.exception(f"Failed to load unknown sparse format: {e}")
                        raise
            else:
                # Dense matrix: direct slicing
                log.debug(f"Using dense matrix reading for {len(absolute_indices_to_load)} rows")
                expr_subset = X_group[sorted(absolute_indices_to_load)][:, source_indices_in_file]
                mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)

        except Exception as e:
            log.exception(f"Error loading specific rows from {file_info['path']}: {e}")
            raise

        # Restore original request order
        reverse_sort_order = np.empty_like(sort_order)
        reverse_sort_order[sort_order] = np.arange(len(sort_order))
        
        return mapped_matrix[reverse_sort_order]


class SimplifiedMultiDataset(Dataset):
    """Simplified dataset class with file-level splits and locality-aware sampling."""
    
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        target_genes: Optional[List[str]] = None,
        genelist_path: Optional[str] = None,
        sample_size: int = 128,
        mode: str = 'train',
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        random_state: Optional[int] = 42,
        cache_file: Optional[str] = None,
        train_files: Optional[List[str]] = None,
        val_files: Optional[List[str]] = None,
        test_files: Optional[List[str]] = None
    ):
        """Initialize the simplified dataset."""
        self.dataset_configs = dataset_configs
        self.sample_size = sample_size
        self.mode = mode
        
        self.initial_random_state = random_state
        self.rng = None
        
        # Load target genes
        if target_genes is not None:
            self.target_genes = target_genes
        elif genelist_path is not None:
            self.target_genes = load_gene_list(genelist_path)
        else:
            raise ValueError("Either target_genes or genelist_path must be provided")
        
        self.n_genes = len(self.target_genes)
        
        # Create cache key for singleton
        config_paths = sorted([config.path for config in dataset_configs])
        cache_key = f"{'_'.join(config_paths)}_{hash(tuple(self.target_genes))}"
        
        # Load metadata with simplified cache
        self.metadata_cache = SimplifiedDatasetCache.get_singleton(
            cache_key,
            dataset_configs=dataset_configs,
            target_genes=self.target_genes,
            cache_file=cache_file
        )
        
        # Use provided file splits or create new ones
        if train_files is not None and val_files is not None and test_files is not None:
            self.train_files = train_files
            self.val_files = val_files
            self.test_files = test_files
        else:
            self._split_files(test_ratio, val_ratio)
        
        # Set active files based on mode
        if mode == 'train':
            self.active_files = self.train_files
        elif mode == 'val':
            self.active_files = self.val_files
        elif mode == 'test':
            self.active_files = self.test_files
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Generate samples using locality-aware strategy
        self._generate_locality_samples()
    
    def _initialize_rng(self):
        """Initialize RNG for current process."""
        worker_info = torch.utils.data.get_worker_info()
        seed = self.initial_random_state
        
        if worker_info is not None:
            if seed is not None:
                seed += worker_info.id
            else:
                seed = int(time.time() * 1000) % (2**32 -1) + worker_info.id
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _split_files(self, test_ratio: float = 0.2, val_ratio: float = 0.2):
        """Split files into train/val/test sets"""
        if self.rng is None:
            self._initialize_rng()
        
        # Get valid files with enough cells
        valid_files = []
        for info in self.metadata_cache.file_info:
            if info['n_cells'] >= self.sample_size:
                valid_files.append(info['path'])
        
        log.info(f"Found {len(valid_files)} files with >= {self.sample_size} cells")
        
        if len(valid_files) == 0:
            raise ValueError(f"No files found with >= {self.sample_size} cells")
        
        # Shuffle files
        shuffled_files = self.rng.permutation(valid_files)
        
        # Split files
        n_test = max(1, int(len(shuffled_files) * test_ratio)) if len(shuffled_files) * test_ratio > 1 else 0
        n_val = max(1, int(len(shuffled_files) * val_ratio)) if len(shuffled_files) * val_ratio > 1 else 0
        
        if len(shuffled_files) > n_test + n_val:
            n_train = len(shuffled_files) - n_test - n_val
        else:
            n_train = 0

        self.test_files = shuffled_files[:n_test].tolist()
        self.val_files = shuffled_files[n_test:n_test+n_val].tolist()
        self.train_files = shuffled_files[n_test+n_val:].tolist()
        
        log.info(f"Split files: {len(self.train_files)} train, {len(self.val_files)} val, {len(self.test_files)} test")
    
    def _generate_locality_samples(self):
        """Generate locality-aware samples from active files."""
        if self.rng is None:
            self._initialize_rng()
            
        log.info(f"[{self.mode}] Generating locality-aware samples...")
        start_time = time.time()
        
        self.samples = []
        active_file_set = set(self.active_files)
        
        if not active_file_set:
            log.warning(f"No active files for mode {self.mode}. Dataset will be empty.")
            return

        total_samples_created = 0
        
        for file_idx, file_info in enumerate(self.metadata_cache.file_info):
            if file_info['path'] not in active_file_set:
                continue
                
            n_cells = file_info['n_cells']
            if n_cells < self.sample_size:
                continue
            
            # Create contiguous samples from this file
            n_samples = n_cells // self.sample_size
            for i in range(n_samples):
                start_idx = i * self.sample_size
                end_idx = start_idx + self.sample_size
                local_indices = np.arange(start_idx, end_idx)
                
                # Create sample tuple: (file_idx, local_indices)
                self.samples.append((file_idx, local_indices))
                total_samples_created += 1
        
        # Shuffle all samples to ensure randomness across files
        self.rng.shuffle(self.samples)
        
        generation_time = time.time() - start_time
        log.info(f"Generated {len(self.samples)} locality-aware samples in {generation_time:.2f}s for {self.mode} mode")
        log.info(f"From {len(active_file_set)} active files")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a sample by index."""
        if self.rng is None:
            self._initialize_rng()
        
        # Get sample information
        file_idx, local_indices = self.samples[idx]
        file_info = self.metadata_cache.file_info[file_idx]
        
        # Load expression data
        expression_data = self.metadata_cache.load_expression_data_from_file(file_idx, local_indices)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(expression_data).float()
        
        # Build metadata
        metadata = {
            'file_path': file_info['path'],
            'file_idx': file_idx,
            'config_idx': file_info['config_idx'],
            'sample_size': len(local_indices),
            'n_genes_found': file_info['found_genes'],
            'dataset_path': self.dataset_configs[file_info['config_idx']].path
        }
        
        return features_tensor, metadata
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            'total_memory_gb': self.metadata_cache._get_memory_usage(),
            'n_files': len(self.metadata_cache.file_info),
            'n_genes': self.n_genes,
            'n_datasets': len(self.dataset_configs),
            'active_files': len(self.active_files),
            'total_samples': len(self.samples)
        }

class TestSamplerDataset(Dataset):
    """
    Simple dataset class for testing/evaluation on a single H5AD file or AnnData object.
    Samples cells from the file/object and returns gene expression data.
    """
    
    def __init__(
        self,
        adata_or_path,  # Can be either str (file path) or AnnData object
        genelist_path: str,
        sample_size: int = 128,
        mode: str = 'eval',
        max_samples: Optional[int] = None,
        gene_name_col: Optional[str] = None,
        filter_organism: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the test dataset.
        
        Args:
            adata_or_path: Either path to H5AD file (str) or AnnData object
            genelist_path: Path to pickle file containing gene list
            sample_size: Number of cells per sample
            mode: Dataset mode (usually 'eval' for testing)
            max_samples: Maximum number of samples to generate (None = all possible)
            gene_name_col: Column name for gene names in var
            filter_organism: Whether to filter for Homo sapiens
            random_state: Random seed for reproducibility
        """
        self.adata_or_path = adata_or_path
        self.genelist_path = genelist_path
        self.sample_size = sample_size
        self.mode = mode
        self.max_samples = max_samples
        self.gene_name_col = gene_name_col
        self.filter_organism = filter_organism
        self.random_state = random_state
        
        # Type checking
        if isinstance(adata_or_path, str):
            self.is_adata_object = False
            self.adata_path = adata_or_path
            self.adata_object = None
        else:
            # Assume it's an AnnData object
            self.is_adata_object = True
            self.adata_path = None
            self.adata_object = adata_or_path
        
        # Initialize RNG
        self.rng = np.random.RandomState(random_state)
        
        # Load target genes from pickle file
        self.target_genes = load_gene_list(genelist_path)
        self.n_genes = len(self.target_genes)
        
        # Load file metadata and prepare samples
        if self.is_adata_object:
            self._load_adata_metadata()
        else:
            self._load_file_metadata()
        
        self._generate_samples()
        
        source_info = f"AnnData object" if self.is_adata_object else self.adata_path
        log.info(f"TestSamplerDataset initialized with {len(self.samples)} samples from {source_info}")
    
    def _load_adata_metadata(self):
        """Load metadata from AnnData object"""
        log.info("Loading metadata from AnnData object")
        
        adata = self.adata_object
        
        # Check organism filtering
        if self.filter_organism:
            if "organism" not in adata.obs.columns:
                log.warning("'organism' column not found in obs, assuming all cells are valid")
                self.human_mask = np.ones(adata.n_obs, dtype=bool)
            else:
                organism_values = adata.obs["organism"].values
                self.human_mask = organism_values == 'Homo sapiens'
                if not self.human_mask.any():
                    raise ValueError("No Homo sapiens cells found in the AnnData object")
        else:
            self.human_mask = np.ones(adata.n_obs, dtype=bool)
        
        self.n_human_cells = self.human_mask.sum()
        
        # Get gene information
        # Try to use raw data first, then fall back to main data
        if adata.raw is not None:
            gene_names = adata.raw.var.index.values
            use_raw = True
            log.info("Using raw data for gene names")
        else:
            gene_names = adata.var.index.values
            use_raw = False
            log.info("Using main data for gene names")
        
        # If gene_name_col is specified, try to use that column
        if self.gene_name_col is not None:
            var_data = adata.raw.var if use_raw else adata.var
            if self.gene_name_col in var_data.columns:
                gene_names = var_data[self.gene_name_col].values
                log.info(f"Using '{self.gene_name_col}' column for gene names")
            else:
                log.warning(f"'{self.gene_name_col}' not found in var, using index")
        
        # Convert to uppercase and handle string types
        if gene_names.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            gene_names = np.char.upper(gene_names.astype(str))
        else:
            gene_names = np.array([str(g).upper() for g in gene_names])
        
        # Create gene mapping
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        self.gene_mapping = {}
        found_genes = 0
        for target_idx, gene in enumerate(self.target_genes):
            if gene in gene_to_idx:
                self.gene_mapping[target_idx] = gene_to_idx[gene]
                found_genes += 1
        
        if found_genes == 0:
            raise ValueError("No target genes found in the AnnData object")
        
        # Store file info
        self.file_info = {
            'path': 'AnnData object',
            'n_cells': self.n_human_cells,
            'n_genes': len(gene_names),
            'use_raw': use_raw,
            'gene_mapping': self.gene_mapping,
            'organism_mask': self.human_mask,
            'is_sparse': True,  # Assume sparse for AnnData objects
            'found_genes': found_genes
        }
        
        log.info(f"Loaded AnnData with {self.n_human_cells} human cells, {found_genes}/{len(self.target_genes)} target genes")
    
    def _load_file_metadata(self):
        """Load metadata from the H5AD file"""
        log.info(f"Loading metadata from {self.adata_path}")
        
        with h5py.File(self.adata_path, "r") as f:
            # Check organism filtering
            if self.filter_organism:
                obs = f["obs"]
                if "organism" not in obs:
                    log.warning("'organism' column not found, assuming all cells are valid")
                    n_cells = f["X"].shape[0] if hasattr(f["X"], 'shape') else f["X"].attrs['shape'][0]
                    self.human_mask = np.ones(n_cells, dtype=bool)
                else:
                    organism_ds = obs["organism"]
                    if "categories" in organism_ds:
                        organism_categories = safe_decode_array(organism_ds["categories"][:])
                        organism_codes = organism_ds["codes"][:]
                        organism_values = organism_categories[organism_codes]
                    else:
                        organism_values = safe_decode_array(organism_ds[:])
                    
                    self.human_mask = organism_values == 'Homo sapiens'
                    if not self.human_mask.any():
                        raise ValueError("No Homo sapiens cells found in the file")
            else:
                n_cells = f["X"].shape[0] if hasattr(f["X"], 'shape') else f["X"].attrs['shape'][0]
                self.human_mask = np.ones(n_cells, dtype=bool)
            
            self.n_human_cells = self.human_mask.sum()
            
            # Get gene information
            use_raw = "raw" in f and "var" in f["raw"]
            gene_names = get_gene_names_from_h5(f, self.gene_name_col, use_raw=use_raw)
            
            if gene_names is None:
                raise ValueError("Could not find gene names in the file")
            
            # Create gene mapping
            gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
            self.gene_mapping = {}
            found_genes = 0
            for target_idx, gene in enumerate(self.target_genes):
                if gene in gene_to_idx:
                    self.gene_mapping[target_idx] = gene_to_idx[gene]
                    found_genes += 1
            
            if found_genes == 0:
                raise ValueError("No target genes found in the file")
            
            # Store file info
            if use_raw:
                X_group = f["raw"]["X"]
            else:
                X_group = f["X"]
            
            self.file_info = {
                'path': self.adata_path,
                'n_cells': self.n_human_cells,
                'n_genes': len(gene_names),
                'use_raw': use_raw,
                'gene_mapping': self.gene_mapping,
                'organism_mask': self.human_mask,
                'is_sparse': isinstance(X_group, h5py.Group) and 'data' in X_group and 'indices' in X_group,
                'found_genes': found_genes
            }
            
            log.info(f"Loaded file with {self.n_human_cells} human cells, {found_genes}/{len(self.target_genes)} target genes")
    
    def _generate_samples(self):
        """
        Generate samples from the file/object.
        For the last incomplete part, it stores instructions for upsampling.
        """
        log.info("Generating samples...")

        if self.n_human_cells < 1:
            log.warning("No valid human cells found. Dataset will be empty.")
            self.samples = []
            return
            
        # Get all valid human cell indices (these are the 'local' indices)
        human_cell_indices_local = np.arange(self.n_human_cells)

        # Limit samples if max_samples is specified
        if self.max_samples is not None:
            max_cells = self.max_samples * self.sample_size
            if len(human_cell_indices_local) > max_cells:
                human_cell_indices_local = human_cell_indices_local[:max_cells]

        self.samples = []

        # Generate full samples
        n_full_samples = len(human_cell_indices_local) // self.sample_size
        for i in range(n_full_samples):
            start_idx = i * self.sample_size
            end_idx = start_idx + self.sample_size
            # For full batches, the indices to load are the same as the reconstruction map
            local_indices_to_load = human_cell_indices_local[start_idx:end_idx]
            reindexing_map = np.arange(self.sample_size)
            self.samples.append((local_indices_to_load, reindexing_map))

        # Handle remaining cells with upsampling
        remaining_cells_count = len(human_cell_indices_local) - n_full_samples * self.sample_size
        if remaining_cells_count > 0:
            # These are the actual local indices of the remaining cells
            remaining_local_indices = human_cell_indices_local[n_full_samples * self.sample_size:]
            
            # These are the indices we need to upsample FROM, relative to the `remaining_local_indices` array
            indices_to_upsample_from = np.arange(remaining_cells_count)
            
            need_to_fill = self.sample_size - remaining_cells_count
            
            # Perform upsampling
            upsampled_relative_indices = self.rng.choice(
                indices_to_upsample_from,
                size=need_to_fill,
                replace=True
            )
            
            # The reindexing map tells us how to construct the final batch from the loaded remaining cells
            reindexing_map = np.concatenate([indices_to_upsample_from, upsampled_relative_indices])
            
            # The indices to load from the file are just the unique remaining ones
            local_indices_to_load = remaining_local_indices
            
            self.samples.append((local_indices_to_load, reindexing_map))

        log.info(f"Generated {len(self.samples)} samples for {self.mode} mode (eval with upsampling)")
        if remaining_cells_count > 0 and remaining_cells_count < self.sample_size:
            log.info(f"Last sample will be constructed from {remaining_cells_count} unique cells.")
    
    def load_expression_data(self, local_indices: np.ndarray) -> np.ndarray:
        """
        Load expression data for specified cell indices.
        Routes to appropriate loading method based on data source.
        """
        if self.is_adata_object:
            return self.load_expression_data_from_adata(local_indices)
        else:
            return self.load_expression_data_from_file(local_indices)
    
    def load_expression_data_from_adata(self, local_indices: np.ndarray) -> np.ndarray:
        """
        Load expression data from AnnData object for specified cell indices.
        """
        # Sort indices for consistency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]
        
        try:
            adata = self.adata_object
            
            # Get absolute row indices in the original AnnData object
            human_indices = np.where(self.file_info['organism_mask'])[0]
            absolute_indices_to_load = human_indices[sorted_local_indices]
            
            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes), dtype=np.float32)
            gene_mapping = self.file_info['gene_mapping']
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))
            
            # Get expression data
            if self.file_info['use_raw'] and adata.raw is not None:
                # Use raw data
                expr_data = adata.raw.X[absolute_indices_to_load, :]
            else:
                # Use main data
                expr_data = adata.X[absolute_indices_to_load, :]
            
            # Convert sparse to dense if needed
            if hasattr(expr_data, 'toarray'):
                expr_data = expr_data.toarray()
            
            # Select target gene columns
            expr_subset = expr_data[:, source_indices_in_file]
            mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)
            
        except Exception as e:
            log.exception(f"Error loading expression data from AnnData object: {e}")
            raise
        
        # Restore original request order
        reverse_sort_order = np.empty_like(sort_order)
        reverse_sort_order[sort_order] = np.arange(len(sort_order))
        
        return mapped_matrix[reverse_sort_order]

    def load_expression_data_from_file(self, local_indices: np.ndarray) -> np.ndarray:
        """
        Load expression data for specified cell indices from H5AD file.
        Reuses the optimized loading logic from SimplifiedDatasetCache.
        """
        # Sort indices for better h5py reading efficiency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]
        
        try:
            f = get_h5_handle(self.file_info['path'])
            X_group = f["raw"]["X"] if self.file_info['use_raw'] else f["X"]
            
            # Get absolute row indices in the original HDF5 file
            human_indices = np.where(self.file_info['organism_mask'])[0]
            absolute_indices_to_load = human_indices[sorted_local_indices]
            
            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes), dtype=np.float32)
            gene_mapping = self.file_info['gene_mapping']
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))
            
            if self.file_info['is_sparse']:
                # Handle sparse matrix (CSR format)
                attrs = dict(X_group.attrs)
                encoding_type = attrs.get("encoding-type", "csr_matrix")
                
                if encoding_type == "csr_matrix":
                    # CSR format: efficient row slicing (using optimized logic from SimplifiedDatasetCache)
                    log.debug(f"Using optimized CSR row reading for {len(absolute_indices_to_load)} rows")
                    
                    indptr_h5 = X_group["indptr"]
                    data_h5 = X_group["data"]
                    indices_h5 = X_group["indices"]

                    # 1) Get all required indptr start and end points
                    starts = indptr_h5[absolute_indices_to_load]
                    ends = indptr_h5[absolute_indices_to_load + 1]
                    
                    # 2) Detect physical data block boundaries
                    block_breaks = np.where(ends[:-1] != starts[1:])[0] + 1
                    row_blocks = np.split(np.arange(len(absolute_indices_to_load)), block_breaks)

                    # 3) Read data block by block to minimize I/O operations
                    data_slices = []
                    indices_slices = []
                    
                    row_order = np.concatenate(row_blocks) if row_blocks else np.array([], dtype=int)
                    
                    for block in row_blocks:
                        if len(block) == 0: 
                            continue
                        
                        first_row_in_block_idx = block[0]
                        last_row_in_block_idx = block[-1]
                        
                        data_start = starts[first_row_in_block_idx]
                        data_end = ends[last_row_in_block_idx]
                        
                        # Single I/O operation to read entire contiguous block
                        data_block = data_h5[data_start:data_end]
                        indices_block = indices_h5[data_start:data_end]
                        
                        # Split large block back into individual rows
                        row_lengths = ends[block] - starts[block]
                        cuts = np.cumsum(row_lengths)
                        
                        if len(cuts) > 0:
                            data_slices.extend(np.split(data_block, cuts[:-1]))
                            indices_slices.extend(np.split(indices_block, cuts[:-1]))

                    # 4) Construct sparse matrix from collected slices
                    if not data_slices:
                        sparse_subset = sp.csr_matrix((len(local_indices), self.file_info['n_genes']), dtype=np.float32)
                    else:
                        all_data = np.concatenate(data_slices)
                        all_indices = np.concatenate(indices_slices)

                        new_indptr = np.zeros(len(local_indices) + 1, dtype=indptr_h5.dtype)
                        row_lengths_in_order = ends[row_order] - starts[row_order]
                        new_indptr[1:] = np.cumsum(row_lengths_in_order)
                        
                        sparse_subset = sp.csr_matrix(
                            (all_data, all_indices, new_indptr),
                            shape=(len(local_indices), self.file_info['n_genes']),
                            dtype=np.float32
                        )
                        
                    # 5) Select target gene columns and convert to dense
                    expr_subset = sparse_subset[:, source_indices_in_file].toarray()
                    mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)
                
                elif encoding_type == "csc_matrix":
                    # CSC format: column-oriented reading strategy
                    log.debug(f"Using CSC column-oriented reading for {len(absolute_indices_to_load)} rows")
                    
                    data_h5 = X_group["data"]
                    indices_h5 = X_group["indices"] 
                    indptr_h5 = X_group["indptr"]
                    
                    # Get actual matrix shape
                    # For CSC, indptr length is n_cols + 1, i.e., number of genes + 1
                    n_genes_in_file = len(indptr_h5) - 1
                    n_cells_in_file = self.file_info['n_cells']
                    
                    # Adaptive strategy selection
                    load_threshold = min(1000, n_cells_in_file * 0.1)  # 10% of cells or 1000, whichever is smaller
                    
                    if len(absolute_indices_to_load) <= load_threshold:
                        # Strategy A: Row-by-row reading (suitable for few rows)
                        log.debug(f"Using row-by-row CSC reading for {len(absolute_indices_to_load)} rows")
                        
                        result_matrix = np.zeros((len(absolute_indices_to_load), n_genes_in_file), dtype=np.float32)
                        
                        # Create mapping from absolute_index to result_matrix row number
                        abs_to_res_map = {abs_idx: i for i, abs_idx in enumerate(absolute_indices_to_load)}
                        
                        # Process each gene (column)
                        for gene_idx in range(n_genes_in_file):
                            col_start = indptr_h5[gene_idx]
                            col_end = indptr_h5[gene_idx + 1]
                            
                            if col_end > col_start:
                                # Read all non-zero cell indices and values for this gene (column)
                                col_cell_indices = indices_h5[col_start:col_end]
                                col_values = data_h5[col_start:col_end]
                                
                                # Use np.isin to find which cells we need are present in this column
                                isin_mask = np.isin(col_cell_indices, absolute_indices_to_load, assume_unique=True)
                                
                                if np.any(isin_mask):
                                    # Get absolute indices of cells that have values
                                    matched_abs_indices = col_cell_indices[isin_mask]
                                    # Get corresponding values
                                    matched_values = col_values[isin_mask]
                                    
                                    # Fill result matrix using the mapping we created earlier
                                    for abs_idx, value in zip(matched_abs_indices, matched_values):
                                        result_matrix_row = abs_to_res_map[abs_idx]
                                        result_matrix[result_matrix_row, gene_idx] = value
                    
                    else:
                        # Strategy B: Load full matrix then slice (suitable for many rows)
                        log.debug(f"Using full matrix load for {len(absolute_indices_to_load)} rows (threshold: {load_threshold})")
                        
                        # Load entire CSC matrix
                        data = data_h5[:]
                        indices = indices_h5[:]
                        indptr = indptr_h5[:]
                        
                        # Construct scipy CSC matrix (Note: this is transposed, genes x cells)
                        csc_matrix_transposed = sp.csc_matrix(
                            (data, indices, indptr), 
                            shape=(n_genes_in_file, n_cells_in_file)
                        )
                        
                        # Transpose to get normal cells x genes CSR matrix
                        csr_matrix = csc_matrix_transposed.T
                        
                        # Now we can efficiently perform row slicing
                        subset_sparse = csr_matrix[absolute_indices_to_load, :]
                        result_matrix = subset_sparse.toarray()
                    
                    # Map to target genes
                    mapped_matrix[:, target_indices_in_result] = result_matrix[:, source_indices_in_file].astype(np.float32)
                
                else:
                    # Unknown sparse format - try to infer from indptr length
                    log.warning(f"Unknown sparse encoding type '{encoding_type}' in {self.file_info['path']}")
                    log.warning("Attempting to infer format from indptr length")
                    
                    indptr_len = len(X_group['indptr'])
                    
                    if indptr_len == self.file_info['n_cells'] + 1:
                        # Likely CSR format
                        log.debug("Guessing CSR format based on indptr length")
                        # Fall back to basic CSR logic
                        raise NotImplementedError("CSR format inference - please add explicit encoding-type to file")
                        
                    elif indptr_len == self.file_info['n_genes'] + 1:
                        # Likely CSC format  
                        log.debug("Guessing CSC format based on indptr length")
                        # Fall back to basic CSC logic
                        raise NotImplementedError("CSC format inference - please add explicit encoding-type to file")
                    else:
                        raise ValueError(f"Cannot determine sparse matrix format for {self.file_info['path']}")
            
            else:
                # Dense matrix: direct slicing
                expr_subset = X_group[sorted(absolute_indices_to_load)][:, source_indices_in_file]
                mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)
            
        except Exception as e:
            log.exception(f"Error loading expression data: {e}")
            raise
        
        # Restore original request order
        reverse_sort_order = np.empty_like(sort_order)
        reverse_sort_order[sort_order] = np.arange(len(sort_order))
        
        return mapped_matrix[reverse_sort_order]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a sample by index using the two-step loading strategy."""
        
        # 1. Get the instructions for this sample
        local_indices_to_load, reindexing_map = self.samples[idx]

        # 2. Load data for ONLY the unique indices from the file.
        #    `local_indices_to_load` is now guaranteed to be unique and sorted.
        loaded_data = self.load_expression_data(local_indices_to_load)

        # 3. In memory, construct the final batch with duplicates using the reindexing map.
        expression_data = loaded_data[reindexing_map]
        
        # Convert to tensor
        features_tensor = torch.from_numpy(expression_data).float()
        
        # Build metadata
        metadata = {
            'file_path': self.file_info['path'],
            'sample_idx': idx,
            'sample_size': len(expression_data), # The final size of the batch
            'n_genes_found': self.file_info['found_genes'],
            'n_unique_cells_loaded': len(local_indices_to_load) # Info for debugging
        }
        
        return features_tensor, metadata
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        source_info = 'AnnData object' if self.is_adata_object else self.adata_path
        return {
            'source': source_info,
            'n_human_cells': self.n_human_cells,
            'n_genes': self.n_genes,
            'n_samples': len(self.samples),
            'sample_size': self.sample_size,
            'genes_found': self.file_info['found_genes'],
            'genes_total': len(self.target_genes)
        }

def create_train_val_test_datasets(
    dataset_configs: List[DatasetConfig],
    target_genes: Optional[List[str]] = None,
    genelist_path: Optional[str] = None,
    sample_size: int = 128,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_state: int = 42,
    cache_file: Optional[str] = None
) -> Tuple[SimplifiedMultiDataset, SimplifiedMultiDataset, SimplifiedMultiDataset]:
    """Create train, validation, and test datasets with file-level splits."""
    
    # Create a temporary base dataset instance just to perform the file splitting
    log.info("Creating base dataset for file splitting...")
    base_dataset = SimplifiedMultiDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        mode='train',
        random_state=random_state,
        cache_file=cache_file
    )
    
    # Perform file splitting using the base instance
    base_dataset._split_files(test_ratio, val_ratio)
    train_files = base_dataset.train_files
    val_files = base_dataset.val_files
    test_files = base_dataset.test_files

    # Now create the actual datasets using the pre-computed splits
    log.info("Creating training dataset...")
    train_dataset = SimplifiedMultiDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        mode='train',
        random_state=random_state,
        cache_file=cache_file,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files
    )
    
    log.info("Creating validation dataset...")
    val_dataset = SimplifiedMultiDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        mode='val',
        random_state=random_state,
        cache_file=cache_file,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files
    )
    
    log.info("Creating test dataset...")
    test_dataset = SimplifiedMultiDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        mode='test',
        random_state=random_state,
        cache_file=cache_file,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files
    )
    
    return train_dataset, val_dataset, test_dataset


# Example usage functions
def create_example_configs():
    """Example of how to create dataset configurations"""
    config1 = DatasetConfig(
        path="/path/to/dataset1",
        filter_organism=True
    )
    
    config2 = DatasetConfig(
        path="/path/to/dataset2",
        filter_organism=False
    )
    
    return [config1, config2]


def compute_and_save_hvg_union(
    dataset_configs: List[DatasetConfig],
    output_path: str,
    n_top_genes: int = 1000,
    theta: float = 100.0,
    max_cells_per_file: int = 10000,
    random_state: int = 42
) -> List[str]:
    """Compute HVG union and save to file."""
    log.info("Starting HVG computation process...")
    
    hvg_genes = compute_hvg_union(
        dataset_configs=dataset_configs,
        n_top_genes=n_top_genes,
        theta=theta,
        max_cells_per_file=max_cells_per_file,
        output_path=output_path,
        random_state=random_state
    )
    
    log.info(f"HVG computation completed. Saved {len(hvg_genes)} genes to {output_path}")
    return hvg_genes


def create_datasets_from_gene_list(
    dataset_configs: List[DatasetConfig],
    genelist_path: str,
    sample_size: int = 128,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_state: int = 42,
    cache_file: Optional[str] = None
) -> Tuple[SimplifiedMultiDataset, SimplifiedMultiDataset, SimplifiedMultiDataset]:
    """Create datasets from a pre-computed gene list file."""
    log.info(f"Creating datasets from gene list: {genelist_path}")
    
    return create_train_val_test_datasets(
        dataset_configs=dataset_configs,
        target_genes=None,  # Load from file
        genelist_path=genelist_path,
        sample_size=sample_size,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
        cache_file=cache_file
    )

