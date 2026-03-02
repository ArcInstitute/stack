import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List, Union
import h5py
import scipy.sparse as sp
import logging
import threading
import time
from collections import OrderedDict
import gc
import psutil
import os
from pathlib import Path
import pickle
import traceback
from dataclasses import dataclass
import random
import torch.distributed as dist
import anndata
import math

from ..gene_processing import filter_gene_names, get_gene_names_from_h5, safe_decode_array
from ..h5_manager import get_h5_handle, reset_h5_handle_pool, worker_init_fn
from ..hvg import compute_analytic_pearson_residuals, compute_hvg_union

log = logging.getLogger(__name__)

_get_h5_handle = get_h5_handle
_reset_h5_handle_pool = reset_h5_handle_pool
_worker_init_fn = worker_init_fn

@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    path: str
    type: str  # 'human' or 'drug'
    filter_organism: bool = True
    gene_name_col: Optional[str] = None
    
    # For human datasets
    donor_col: Optional[str] = None
    cell_type_col: Optional[str] = None
    
    # For drug datasets  
    condition_col: Optional[str] = None
    cell_line_col: Optional[str] = None
    control_condition: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if self.type == 'human':
            if not self.donor_col or not self.cell_type_col:
                raise ValueError("Human datasets require donor_col and cell_type_col")
        elif self.type == 'drug':
            if not self.condition_col or not self.cell_line_col or not self.control_condition:
                raise ValueError("Drug datasets require condition_col, cell_line_col, and control_condition")
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")
    
    @property
    def group_col(self) -> str:
        """Column used for grouping (donor_id for human, condition for drug)"""
        return self.donor_col if self.type == 'human' else self.condition_col
    
    @property
    def identity_col(self) -> str:
        """Column used for cell identity (cell_type for human, cell_line for drug)"""
        return self.cell_type_col if self.type == 'human' else self.cell_line_col


def load_gene_list(genelist_path: str) -> List[str]:
    """Load previously computed gene list from pickle file."""
    with open(genelist_path, 'rb') as f:
        return pickle.load(f)


class MultiDatasetMetadataCache:
    """
    Enhanced metadata cache implementing the two-phase strategy with comprehensive data loading.
    
    Core features:
    - Efficient metadata storage and retrieval
    - Pre-computed acceleration dictionaries for fast replacement candidate lookup
    - Precise IO that loads only requested cells
    - Complete metadata tracking including file indices and config indices
    """
    _instances: dict[str, "MultiDatasetMetadataCache"] = {}

    @classmethod
    def get_singleton(cls, cache_key: str, **kwargs) -> "MultiDatasetMetadataCache":
        """Return one shared instance per cache key."""
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(**kwargs)
        return cls._instances[cache_key]
    
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        target_genes: List[str],
        cache_file: Optional[str] = None,
        max_memory_gb: Optional[float] = None,
        cache_ratio: float = 0.3,
        block_size: int = 1000
    ):
        """
        Initialize the metadata cache with all necessary tracking information.
        
        Args:
            dataset_configs: List of dataset configurations
            target_genes: List of target gene names to extract
            cache_file: Optional path to cache metadata
            max_memory_gb: Maximum memory usage in GB (for compatibility)
            cache_ratio: Cache ratio (for compatibility)
            block_size: Block size (for compatibility)
        """
        self.dataset_configs = dataset_configs
        self.target_genes = target_genes
        self.n_genes = len(target_genes)
        
        # Setup memory management (kept for compatibility)
        self._setup_memory_limits(max_memory_gb, cache_ratio)
        
        log.info("Initializing Enhanced MultiDatasetMetadataCache...")
        start_time = time.time()
        
        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            try:
                self._load_from_cache(cache_file)
                log.info(f"Loaded metadata from cache: {cache_file}")
            except Exception as e:
                log.warning(f"Failed to load cache, rebuilding: {e}")
                self._rebuild_and_save_cache(cache_file)
        else:
            # Build and save new cache
            self._rebuild_and_save_cache(cache_file)
        
        # PERFORMANCE FIX: Precompute cell indices for vectorized operations
        self.cell_config_indices = np.array([
            self.group_mapping[gid]['config_idx'] 
            for gid in self.global_group_ids
        ])
        
        self.cell_file_indices = np.array([
            self.cell_to_file_mapping[i][0] 
            for i in range(self.n_cells)
        ])
        
        load_time = time.time() - start_time
        log.info(f"Initialization completed in {load_time:.2f}s. Total cells: {self.n_cells}")
        log.info(f"Current memory usage: ~{self._get_memory_usage():.2f}GB")
    
    def _rebuild_and_save_cache(self, cache_file: Optional[str]):
        """Helper function that encapsulates the complete build, precompute, and save workflow"""
        self._build_all_metadata()
        # CORE FIX: Precompute acceleration pools BEFORE saving to cache
        self._precompute_acceleration_pools()
        if cache_file:
            self._save_to_cache(cache_file)
        
    def _setup_memory_limits(self, max_memory_gb, cache_ratio):
        """Setup memory management parameters (kept for compatibility)"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if max_memory_gb is None:
            self.max_memory_gb = available_memory_gb * 0.8
        else:
            self.max_memory_gb = min(max_memory_gb, available_memory_gb * 0.9)
        
        self.cache_memory_limit = self.max_memory_gb * cache_ratio
        
    def _get_memory_usage(self):
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    def _save_to_cache(self, cache_file: str):
        """Save metadata to cache file including acceleration pools"""
        try:
            cache_data = {
                'file_info': self.file_info,
                'group_mapping': self.group_mapping,
                'cell_to_file_mapping': self.cell_to_file_mapping,
                'global_group_ids': self.global_group_ids,
                'cell_identities': self.cell_identities,
                'conditions': self.conditions,
                'dataset_types': self.dataset_types,
                'n_cells': self.n_cells,
                'target_genes': self.target_genes,
                'dataset_configs': self.dataset_configs,
                # CORE FIX: Save acceleration pools to cache
                'group_identity_pool': self.group_identity_pool,
                'identity_to_groups_map': self.identity_to_groups_map,
                'file_identity_pool': self.file_identity_pool,
                'config_identity_pool': self.config_identity_pool,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            log.info(f"Saved metadata cache to {cache_file}")
        except Exception as e:
            log.exception(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_file: str):
        """Load metadata from cache file including acceleration pools"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.file_info = cache_data['file_info']
        self.group_mapping = cache_data['group_mapping']
        self.cell_to_file_mapping = cache_data['cell_to_file_mapping']
        self.global_group_ids = cache_data['global_group_ids']
        self.cell_identities = cache_data['cell_identities']
        self.conditions = cache_data['conditions']
        self.dataset_types = cache_data['dataset_types']
        self.n_cells = cache_data['n_cells']
        
        # Verify target genes match
        if cache_data['target_genes'] != self.target_genes:
            raise ValueError("Target genes in cache don't match current target genes")
        
        # Verify dataset configs match (simplified check)
        if len(cache_data['dataset_configs']) != len(self.dataset_configs):
            raise ValueError("Dataset configurations don't match cache")
        
        # CORE FIX: Load acceleration pools from cache
        self.group_identity_pool = cache_data['group_identity_pool']
        self.identity_to_groups_map = cache_data['identity_to_groups_map']
        self.file_identity_pool = cache_data['file_identity_pool']
        self.config_identity_pool = cache_data['config_identity_pool']
        
        log.info(f"Loaded acceleration pools from cache:")
        log.info(f"  Group-identity combinations: {len(self.group_identity_pool)}")
        log.info(f"  File-identity combinations: {len(self.file_identity_pool)}")
        log.info(f"  Config-identity combinations: {len(self.config_identity_pool)}")
        log.info(f"  Unique identities: {len(self.identity_to_groups_map)}")
    
    def _build_all_metadata(self):
        """Load metadata from all H5AD files using h5py"""
        self.file_info = []
        self.group_mapping = {}  # Global group ID to original group ID mapping
        self.cell_to_file_mapping = []  # Maps global cell index to (file_idx, local_cell_idx)
        self.global_group_ids = []  # Global group IDs for all cells
        self.cell_identities = []  # Cell identity (type/line) for each cell
        self.conditions = []  # Condition for each cell (human: "human", drug: actual condition)
        self.dataset_types = []  # Dataset type for each cell ("human" or "drug")
        
        global_cell_idx = 0
        global_group_counter = 0
        group_to_global = {}  # (config_idx, original_group) -> global_group_id
        
        # Create gene set for efficient lookup
        target_gene_set = set(self.target_genes)
        
        for config_idx, config in enumerate(self.dataset_configs):
            log.info(f"Loading metadata from dataset {config_idx + 1}/{len(self.dataset_configs)}: {config.path} ({config.type})")
            
            h5ad_files = list(Path(config.path).glob("*.h5ad")) + list(Path(config.path).glob("*.h5"))
            h5ad_files = sorted(list(set(h5ad_files)))
            
            for file_idx, h5ad_file in enumerate(h5ad_files):
                try:
                    log.info(f"  Processing {h5ad_file.name}...")
                    
                    with h5py.File(h5ad_file, "r") as f:
                        obs = f["obs"]
                        
                        # Get organism filter mask
                        if config.filter_organism:
                            if "organism" not in obs:
                                log.warning(f"    'organism' column not found, skipping")
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
                                log.warning(f"    No Homo sapiens cells found, skipping")
                                continue
                        else:
                            # Get total number of cells
                            if hasattr(f["X"], 'shape'):
                                n_cells = f["X"].shape[0]
                            else:
                                n_cells = f["X"].attrs['shape'][0]
                            human_mask = np.ones(n_cells, dtype=bool)
                        
                        # Check required columns
                        if config.group_col not in obs:
                            log.warning(f"    '{config.group_col}' column not found, skipping")
                            continue
                        
                        if config.identity_col not in obs:
                            log.warning(f"    '{config.identity_col}' column not found, skipping")
                            continue
                        
                        # Get group information (donor_id for human, condition for drug)
                        group_ds = obs[config.group_col]
                        if "categories" in group_ds:
                            group_categories = safe_decode_array(group_ds["categories"][:])
                            group_codes = group_ds["codes"][:]
                            all_group_values = group_categories[group_codes]
                        else:
                            all_group_values = safe_decode_array(group_ds[:])
                        
                        # Get cell identity information (cell_type for human, cell_line for drug)
                        identity_ds = obs[config.identity_col]
                        if "categories" in identity_ds:
                            identity_categories = safe_decode_array(identity_ds["categories"][:])
                            identity_codes = identity_ds["codes"][:]
                            all_identity_values = identity_categories[identity_codes]
                        else:
                            all_identity_values = safe_decode_array(identity_ds[:])
                        
                        # Filter all values by organism mask
                        group_ids = all_group_values[human_mask]
                        identities = all_identity_values[human_mask]
                        
                        # For drug datasets, also get the actual condition column for replacement logic
                        if config.type == 'drug':
                            # group_ids already contains the condition values
                            current_conditions = group_ids.copy()
                        else:
                            # For human datasets, condition is just "human"
                            current_conditions = np.array(["human"] * len(group_ids))
                        
                        # Map to global group IDs
                        file_global_groups = []
                        for group_id in group_ids:
                            group_key = (config_idx, str(group_id))
                            if group_key not in group_to_global:
                                group_to_global[group_key] = global_group_counter
                                self.group_mapping[global_group_counter] = {
                                    'config_idx': config_idx,
                                    'original_id': str(group_id),
                                    'dataset_type': config.type,
                                    'path': config.path,
                                    'group_col': config.group_col
                                }
                                global_group_counter += 1
                            file_global_groups.append(group_to_global[group_key])
                        
                        # Get gene information
                        gene_names = None
                        use_raw = False
                        
                        use_raw = "raw" in f and "var" in f["raw"]
                        gene_names = get_gene_names_from_h5(f, config.gene_name_col, use_raw=use_raw)
                        
                        if gene_names is None:
                            log.warning(f"    Could not find gene names, skipping")
                            continue
                        
                        # Create gene index mapping - only for genes that exist in target_genes
                        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
                        gene_mapping = {}
                        found_genes = 0
                        for target_idx, gene in enumerate(self.target_genes):
                            if gene in gene_to_idx:
                                gene_mapping[target_idx] = gene_to_idx[gene]
                                found_genes += 1
                        
                        if found_genes == 0:
                            log.warning(f"    No target genes found in {h5ad_file.name}, skipping")
                            continue
                        
                        log.info(f"    Found {found_genes}/{len(self.target_genes)} target genes")
                        
                        # Get matrix info
                        if use_raw:
                            X_group = f["raw"]["X"]
                        else:
                            X_group = f["X"]
                        
                        # Check if sparse
                        is_sparse = isinstance(X_group, h5py.Group) and 'data' in X_group and 'indices' in X_group
                        
                        # Store file information
                        file_info = {
                            'path': str(h5ad_file),
                            'config_idx': config_idx,
                            'n_cells': len(file_global_groups),  # Number of filtered cells
                            'n_genes': len(gene_names),
                            'use_raw': use_raw,
                            'gene_mapping': gene_mapping,
                            'start_cell_idx': global_cell_idx,
                            'end_cell_idx': global_cell_idx + len(file_global_groups),
                            'organism_mask': human_mask,  # Store mask for later use
                            'is_sparse': is_sparse,
                            'found_genes': found_genes
                        }
                        self.file_info.append(file_info)
                        
                        # Update cell mapping
                        for local_idx in range(len(file_global_groups)):
                            self.cell_to_file_mapping.append((len(self.file_info) - 1, local_idx))
                            self.global_group_ids.append(file_global_groups[local_idx])
                            self.cell_identities.append(identities[local_idx])
                            self.conditions.append(current_conditions[local_idx])
                            self.dataset_types.append(config.type)
                        
                        global_cell_idx += len(file_global_groups)
                        log.info(f"    Added {len(file_global_groups)} cells, total: {global_cell_idx}")
                    
                except Exception as e:
                    log.exception(f"    Error processing {h5ad_file.name}: {e}")
                    continue
        
        self.n_cells = global_cell_idx
        self.global_group_ids = np.array(self.global_group_ids)
        self.cell_identities = np.array(self.cell_identities)
        self.conditions = np.array(self.conditions)
        self.dataset_types = np.array(self.dataset_types)
        
        log.info(f"Loaded metadata for {self.n_cells} cells from {len(self.file_info)} files")
        log.info(f"Total unique groups: {len(self.group_mapping)}")
        
        # Log dataset statistics
        for config_idx, config in enumerate(self.dataset_configs):
            mask = np.array([self.group_mapping[gid]['config_idx'] == config_idx for gid in self.global_group_ids])
            n_cells_in_dataset = mask.sum()
            log.info(f"Dataset {config_idx} ({config.type}): {n_cells_in_dataset} cells")
    
    def _precompute_acceleration_pools(self):
        """
        CORE ADDITION: Pre-compute acceleration dictionaries for fast replacement lookup.
        
        These dictionaries enable lightning-fast lookups during sample generation:
        - group_identity_pool: (identity, group_id) -> [cell_indices]
        - identity_to_groups_map: identity -> [group_ids] 
        - file_identity_pool: (identity, file_idx) -> [cell_indices] (for intra-file replacement)
        - config_identity_pool: (identity, config_idx) -> [cell_indices] (for intra-dataset replacement)
        """
        log.info("Pre-computing acceleration pools...")
        start_time = time.time()

        # Dictionary A: (identity, group_id) -> [cell indices list]
        self.group_identity_pool = {}
        # Dictionary B: identity -> [group IDs that have this identity]
        self.identity_to_groups_map = {}
        # Dictionary C: (identity, file_idx) -> [cell indices list] (for intra-file replacement)
        self.file_identity_pool = {}
        # Dictionary D: (identity, config_idx) -> [cell indices list] (for intra-dataset replacement)
        self.config_identity_pool = {}

        # Build all dictionaries in a single pass
        for cell_idx in range(self.n_cells):
            identity = self.cell_identities[cell_idx]
            group_id = self.global_group_ids[cell_idx]
            file_idx, _ = self.cell_to_file_mapping[cell_idx]
            config_idx = self.group_mapping[group_id]['config_idx']
            
            # Fill Dictionary A: group-identity pool
            key_a = (identity, group_id)
            if key_a not in self.group_identity_pool:
                self.group_identity_pool[key_a] = []
            self.group_identity_pool[key_a].append(cell_idx)
            
            # Fill Dictionary B: identity to groups mapping
            if identity not in self.identity_to_groups_map:
                self.identity_to_groups_map[identity] = set()
            self.identity_to_groups_map[identity].add(group_id)
            
            # Fill Dictionary C: file-identity pool
            key_c = (identity, file_idx)
            if key_c not in self.file_identity_pool:
                self.file_identity_pool[key_c] = []
            self.file_identity_pool[key_c].append(cell_idx)
            
            # Fill Dictionary D: config-identity pool
            key_d = (identity, config_idx)
            if key_d not in self.config_identity_pool:
                self.config_identity_pool[key_d] = []
            self.config_identity_pool[key_d].append(cell_idx)
        
        # Convert sets to lists for random selection and pool to numpy arrays
        for identity in self.identity_to_groups_map:
            self.identity_to_groups_map[identity] = list(self.identity_to_groups_map[identity])

        # <<< OPTIMIZATION >>> Convert lists in pools to numpy arrays for faster indexing
        for pool in [self.group_identity_pool, self.file_identity_pool, self.config_identity_pool]:
            for key in pool:
                pool[key] = np.array(pool[key])

        log.info(f"Acceleration pools pre-computed in {time.time() - start_time:.2f}s")
        log.info(f"  Group-identity combinations: {len(self.group_identity_pool)}")
        log.info(f"  File-identity combinations: {len(self.file_identity_pool)}")
        log.info(f"  Config-identity combinations: {len(self.config_identity_pool)}")
        log.info(f"  Unique identities: {len(self.identity_to_groups_map)}")

    # <<< OPTIMIZATION 1: High-performance sparse HDF5 reading with block merging >>>
    def _load_specific_rows_from_file(self, file_idx: int, local_indices: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED IO FUNCTION: Efficiently load specified rows from a single file using format-aware reading strategy.
        This version supports both CSR and CSC sparse formats with automatic detection.
        """
        file_info = self.file_info[file_idx]
        
        # Sort indices for better h5py reading efficiency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]

        try:
            f = _get_h5_handle(file_info['path'])
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

    def load_expression_data(self, cell_indices: np.ndarray) -> np.ndarray:
        """
        Enhanced data loading with precise IO strategy.
        
        Groups requested cells by file and loads only the exact requested cells.
        """
        if len(cell_indices) == 0:
            return np.empty((0, self.n_genes), dtype=np.float32)

        result = np.empty((len(cell_indices), self.n_genes), dtype=np.float32)

        # Group cells to load by file
        file_groups = {}
        for i, global_cell_idx in enumerate(cell_indices):
            file_idx, local_row_idx = self.cell_to_file_mapping[global_cell_idx]
            if file_idx not in file_groups:
                file_groups[file_idx] = []
            file_groups[file_idx].append((i, local_row_idx))

        # Load data from each file
        for file_idx, indices_list in file_groups.items():
            result_positions = np.array([item[0] for item in indices_list])
            local_indices_in_file = np.array([item[1] for item in indices_list])
            
            # Call the core IO function
            file_data = self._load_specific_rows_from_file(file_idx, local_indices_in_file)
            
            # Place results in correct positions
            result[result_positions] = file_data
            
        return result
    
    # <<< OPTIMIZATION 2: Vectorized cell replacement logic using NumPy >>>
    def find_replacement_cells(
        self, 
        cell_indices: np.ndarray, 
        rng: np.random.RandomState,
        replacement_ratio: float = 0.25,
        use_intra_file_first: bool = True,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        OPTIMIZED: Find replacement cells with comprehensive strategy support using NumPy.
        This version avoids creating pandas DataFrames inside the performance-critical path.
        """
        n_cells = len(cell_indices)
        n_replace_target = int(n_cells * replacement_ratio)
        
        if n_replace_target == 0:
            return cell_indices, {}
        
        # Step 1: Select identities for replacement using efficient NumPy grouping
        cell_identities = self.get_cell_identities(cell_indices)
        
        # Get unique identities, their counts, and inverse mapping
        unique_identities, identity_inverse_map, identity_counts = np.unique(
            cell_identities, return_inverse=True, return_counts=True
        )
        
        # Shuffle identity groups to randomize selection
        shuffled_identity_indices = rng.permutation(len(unique_identities))
        
        # Select identity groups until we reach target replacement count
        cells_to_be_replaced = []
        total_selected = 0
        
        for identity_idx in shuffled_identity_indices:
            count = identity_counts[identity_idx]
            
            # Get all cells belonging to this identity group
            group_mask = identity_inverse_map == identity_idx
            cells_in_group = cell_indices[group_mask]

            if total_selected + count <= n_replace_target:
                cells_to_be_replaced.extend(cells_in_group)
                total_selected += count
            else:
                remaining_quota = n_replace_target - total_selected
                if remaining_quota > 0:
                    selected_cells = rng.choice(
                        cells_in_group, 
                        size=remaining_quota, 
                        replace=False
                    )
                    cells_to_be_replaced.extend(selected_cells)
                break
        
        indices_to_replace = np.array(cells_to_be_replaced)
        if len(indices_to_replace) == 0:
             return cell_indices, {}
        indices_to_keep = np.setdiff1d(cell_indices, indices_to_replace, assume_unique=True)
        
        # Step 2: Group cells to be replaced by identity for strategic replacement
        replace_identities = self.get_cell_identities(indices_to_replace)
        #unique_replace_identities = rng.permutation(np.unique(replace_identities))
        unique_replace_identities = np.unique(replace_identities)
        
        # Step 3: Execute strategic replacement for each selected identity
        final_replacement_map = {}
        unreplaced_cells = []

        for identity in unique_replace_identities:
            # Get all original cells for this identity that need replacement
            original_cells_mask = replace_identities == identity
            original_cells = indices_to_replace[original_cells_mask]
            
            # Get sample cell to determine context
            sample_cell = original_cells[0]
            sample_dataset_type = self.dataset_types[sample_cell]
            sample_group = self.global_group_ids[sample_cell]
            sample_config_idx = self.cell_config_indices[sample_cell]
            sample_file_idx = self.cell_file_indices[sample_cell]
            
            # Determine replacement rule based on dataset type
            if sample_dataset_type == 'human':
                replacement_rule = ('DIFFERENT_GROUP',)
            elif sample_dataset_type == 'drug':
                control_condition = self.dataset_configs[sample_config_idx].control_condition
                replacement_rule = ('CONTROL_GROUP', control_condition)
            else:
                log.warning(f"Unknown dataset type '{sample_dataset_type}' for identity '{identity}'. Skipping.")
                unreplaced_cells.extend(original_cells)
                continue
            
            # Determine scope priority
            primary_scope = 'file' if use_intra_file_first else 'dataset'
            fallback_scope = 'dataset' if use_intra_file_first else 'file'
            
            # Try to find replacement cells
            replacement_cells = self._find_and_sample_replacements(
                original_cells, identity, replacement_rule, primary_scope, fallback_scope,
                sample_group, sample_config_idx, sample_file_idx, rng
            )
            
            if replacement_cells is not None:
                # Create mapping between original and replacement cells
                for original_cell, replacement_cell in zip(original_cells, replacement_cells):
                    final_replacement_map[original_cell] = replacement_cell
            else:
                # No replacement found, add to unreplaced list
                unreplaced_cells.extend(original_cells)

        # Step 4: Merge unreplaced cells back to keep list
        if unreplaced_cells:
            indices_to_keep = np.concatenate([indices_to_keep, np.array(unreplaced_cells)])

        return indices_to_keep, final_replacement_map
    
    def _find_and_sample_replacements(
        self,
        original_cells: List[int],
        identity: str,
        replacement_rule: Tuple,
        primary_scope: str,
        fallback_scope: str,
        sample_group: int,
        sample_config_idx: int,
        sample_file_idx: int,
        rng: np.random.RandomState
    ) -> Optional[np.ndarray]:
        """
        Core replacement finding logic with comprehensive strategy support.
        
        Args:
            original_cells: List of cells to find replacements for
            identity: Cell identity to match
            replacement_rule: Tuple describing replacement rule
            primary_scope: 'file' or 'dataset' - primary strategy
            fallback_scope: 'file' or 'dataset' - fallback strategy
            sample_group: Sample group ID for context
            sample_config_idx: Sample config index for context
            sample_file_idx: Sample file index for context
            rng: Random number generator
            
        Returns:
            Array of replacement cell indices or None if no candidates found
        """
        n_needed = len(original_cells)
        
        # Try primary scope first
        candidates = self._get_candidate_pool(
            identity, replacement_rule, primary_scope,
            sample_group, sample_config_idx, sample_file_idx
        )
        
        used_strategy = primary_scope
        
        # Sample replacement cells if candidates found
        if len(candidates) > 0:
            if len(candidates) >= n_needed:
                #selected_replacements = rng.choice(candidates, size=n_needed, replace=False)
                max_start_index = len(candidates) - n_needed
                start_index = rng.randint(0, max_start_index + 1)
                selected_replacements = candidates[start_index : start_index + n_needed]
            else:
                # Use replacement with repetition if not enough candidates
                indices = np.arange(n_needed) % len(candidates)
                selected_replacements = candidates[indices]
            return selected_replacements
        else:
            log.info(f"No replacement candidates found for identity '{identity}' with any strategy.")
            return None
    
    # <<< OPTIMIZATION 3: Vectorized candidate pool filtering >>>
    def _get_candidate_pool(
        self,
        identity: str,
        replacement_rule: Tuple,
        scope: str,
        sample_group: int,
        sample_config_idx: int,
        sample_file_idx: int
    ) -> np.ndarray:
        """
        OPTIMIZED: Get candidate cell pool based on identity, rule, and scope using NumPy.
        """
        rule_type = replacement_rule[0]
        
        # Get base candidates from pre-computed pools
        if scope == 'file':
            base_candidates = self.file_identity_pool.get((identity, sample_file_idx), np.array([], dtype=np.int64))
        elif scope == 'dataset':
            base_candidates = self.config_identity_pool.get((identity, sample_config_idx), np.array([], dtype=np.int64))
        else:
            return np.array([], dtype=np.int64)
        
        # <<< BUG FIX IS HERE >>>
        # If the base pool is empty, no candidates can be found. Return early.
        if len(base_candidates) == 0:
            return np.array([], dtype=np.int64)
        
        # Apply replacement rule filtering using vectorized operations
        if rule_type == 'DIFFERENT_GROUP':
            # For human datasets: exclude same group (donor)
            candidate_groups = self.global_group_ids[base_candidates]
            mask = candidate_groups != sample_group
            return base_candidates[mask]
                    
        elif rule_type == 'CONTROL_GROUP':
            # For drug datasets: only include control condition
            control_condition = replacement_rule[1]
            candidate_conditions = self.conditions[base_candidates]
            # Case-insensitive comparison
            mask = np.core.defchararray.lower(candidate_conditions) == control_condition.lower()
            return base_candidates[mask]
        
        # Fallback if rule_type is unknown, return empty
        return np.array([], dtype=np.int64)
        
    def get_group_names(self, indices: np.ndarray) -> np.ndarray:
        """Return group IDs for the provided cell indices."""
        return self.global_group_ids[indices]
    
    def get_cell_identities(self, indices: np.ndarray) -> np.ndarray:
        """Return cell identities for the provided cell indices."""
        return self.cell_identities[indices]
    
    def get_conditions(self, indices: np.ndarray) -> np.ndarray:
        """Return conditions for the provided cell indices."""
        return self.conditions[indices]
    
    def get_dataset_types(self, indices: np.ndarray) -> np.ndarray:
        """Return dataset types for the provided cell indices."""
        return self.dataset_types[indices]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            'total_memory_gb': self._get_memory_usage(),
            'n_files': len(self.file_info),
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'n_datasets': len(self.dataset_configs),
            'dataset_types': [config.type for config in self.dataset_configs],
            'acceleration_pools': {
                'group_identity_combinations': len(self.group_identity_pool),
                'file_identity_combinations': len(self.file_identity_pool),
                'config_identity_combinations': len(self.config_identity_pool),
                'unique_identities': len(self.identity_to_groups_map)
            }
        }


class MultiDatasetSplittableDataset(Dataset):
    """
    Enhanced dataset class with locality-aware sampling and comprehensive replacement strategy support.
    
    Key features:
    - Physical locality-aware sampling for optimal I/O patterns
    - Supports both human and drug dataset replacement strategies
    - Configurable intra-file vs intra-dataset replacement priority
    - Proper random state management for multi-process environments
    - All original parameters restored and enhanced
    """
    
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        target_genes: Optional[List[str]] = None,
        genelist_path: Optional[str] = None,
        sample_size: int = 128,
        replacement_ratio: float = 0.25,
        intra_file_replacement_prob: float = 1.0,  # RESTORED: Probability of using intra-file first
        min_cells_per_group: int = 128,
        train_groups: Optional[List[str]] = None,
        val_groups: Optional[List[str]] = None,
        test_groups: Optional[List[str]] = None,
        mode: str = 'train',
        random_state: Optional[int] = 42,
        resample: bool = False,
        cache_file: Optional[str] = None,
        max_memory_gb: Optional[float] = None,  # RESTORED: For compatibility
        cache_ratio: float = 0.3,  # RESTORED: For compatibility
        block_size: int = 1000  # RESTORED: For compatibility
    ):
        """
        Initialize the enhanced dataset with full parameter support.
        
        Args:
            dataset_configs: List of dataset configurations
            target_genes: List of target gene names, if None will load from genelist_path
            genelist_path: Path to saved gene list file
            sample_size: Number of cells to sample from each group
            replacement_ratio: Fraction of cells to replace (0.0-1.0)
            intra_file_replacement_prob: Probability of prioritizing file-level replacement (0.0-1.0)
                                       1.0 = always try file-level first, then fallback to dataset-level
                                       0.0 = always try dataset-level first, then fallback to file-level
                                       0.5 = 50% chance of either strategy as primary
            min_cells_per_group: Min cells required per group
            train_groups: List of group IDs for training
            val_groups: List of group IDs for validation
            test_groups: List of group IDs for testing
            mode: 'train', 'val', or 'test'
            random_state: Random seed
            resample: Whether to resample (for training)
            cache_file: Optional path to cache metadata
            max_memory_gb: Maximum memory usage in GB (for compatibility)
            cache_ratio: Fraction of memory to use for caching (for compatibility)
            block_size: Number of rows per cache block (for compatibility)
        """
        self.dataset_configs = dataset_configs
        self.sample_size = sample_size
        self.replacement_ratio = replacement_ratio
        self.intra_file_replacement_prob = intra_file_replacement_prob  # RESTORED
        self.min_cells_per_group = min_cells_per_group
        self.mode = mode
        self.resample = resample
        
        # Store initial random state for proper multi-process handling
        self.initial_random_state = random_state
        self.rng = None  # Lazy initialization to fix multi-process issues
        
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
        
        # Load metadata with enhanced cache
        self.metadata_cache = MultiDatasetMetadataCache.get_singleton(
            cache_key,
            dataset_configs=dataset_configs,
            target_genes=self.target_genes,
            cache_file=cache_file,
            max_memory_gb=max_memory_gb,
            cache_ratio=cache_ratio,
            block_size=block_size
        )

        all_unique_identities = np.unique(self.metadata_cache.cell_identities)
        # Create the mapping and store it in the dataset instance itself.
        self.identity_to_id_map = {
            identity: i for i, identity in enumerate(all_unique_identities)
        }
        log.info(f"[{self.mode}] Created mapping for {len(self.identity_to_id_map)} unique cell identities.")
        
        # Use provided group splits or create new ones
        if train_groups is not None and val_groups is not None and test_groups is not None:
            self.train_groups = train_groups
            self.val_groups = val_groups
            self.test_groups = test_groups
        else:
            self._split_groups()
        
        # Set active groups based on mode
        if mode == 'train':
            self.active_groups = self.train_groups
        elif mode == 'val':
            self.active_groups = self.val_groups
        elif mode == 'test':
            self.active_groups = self.test_groups
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Generate samples using locality-aware strategy
        self._generate_samples_by_locality()
    
    def _initialize_rng(self):
        """Initialize RNG for current process, fixing multi-process randomness issues."""
        # Use PyTorch's recommended approach for worker processes
        worker_info = torch.utils.data.get_worker_info()
        seed = self.initial_random_state
        
        if worker_info is not None:
            # We are in a worker process
            # Add worker_id to the seed to ensure different workers have different states
            if seed is not None:
                seed += worker_info.id
            else:
                seed = int(time.time() * 1000) % (2**32 -1) + worker_info.id
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _split_groups(self, test_ratio: float = 0.2, val_ratio: float = 0.2):
        """Split groups into train/val/test sets"""
        if self.rng is None:
            self._initialize_rng()
            
        all_indices = np.arange(self.metadata_cache.n_cells)
        group_ids = self.metadata_cache.get_group_names(all_indices)
        
        # Get unique groups and their cell counts
        unique_groups, group_counts = np.unique(group_ids, return_counts=True)
        
        # Filter groups with sufficient cells
        valid_groups = unique_groups[group_counts >= self.min_cells_per_group]
        
        log.info(f"Found {len(valid_groups)} groups with >= {self.min_cells_per_group} cells")
        
        # Split by dataset type to ensure balanced representation
        human_groups = []
        drug_groups = []
        
        for group_id in valid_groups:
            group_info = self.metadata_cache.group_mapping[group_id]
            config = self.dataset_configs[group_info['config_idx']]
            if config.type == 'human':
                human_groups.append(group_id)
            else:
                drug_groups.append(group_id)
        
        log.info(f"Groups by type: {len(human_groups)} human, {len(drug_groups)} drug")
        
        # Shuffle and split each type separately
        def split_group_list(groups, test_r, val_r):
            if len(groups) == 0:
                return [], [], []
            shuffled = self.rng.permutation(groups)
            n_test = max(1, int(len(shuffled) * test_r)) if len(shuffled) * test_r > 1 else 0
            n_val = max(1, int(len(shuffled) * val_r)) if len(shuffled) * val_r > 1 else 0
            
            # Ensure at least one training sample if possible
            if len(shuffled) > n_test + n_val:
                n_train = len(shuffled) - n_test - n_val
            else:
                n_train = 0

            test_groups = shuffled[:n_test].tolist()
            val_groups = shuffled[n_test:n_test+n_val].tolist()
            train_groups = shuffled[n_test+n_val:].tolist()
            return train_groups, val_groups, test_groups
        
        # Split human groups
        human_train, human_val, human_test = split_group_list(human_groups, test_ratio, val_ratio)
        
        # Split drug groups
        drug_train, drug_val, drug_test = split_group_list(drug_groups, test_ratio, val_ratio)
        
        # Combine splits
        self.train_groups = human_train + drug_train
        self.val_groups = human_val + drug_val
        self.test_groups = human_test + drug_test
        
        log.info(f"Split groups: {len(self.train_groups)} train, {len(self.val_groups)} val, {len(self.test_groups)} test")
        log.info(f"  Human - train: {len(human_train)}, val: {len(human_val)}, test: {len(human_test)}")
        log.info(f"  Drug - train: {len(drug_train)}, val: {len(drug_val)}, test: {len(drug_test)}")
    
    # <<< CORE OPTIMIZATION: Locality-aware sampling strategy >>>
    def _generate_samples_by_locality(self):
        """
        OPTIMIZED SAMPLING: Generate samples based on physical locality while preserving group semantics.
        
        This approach transforms random access patterns into sequential patterns:
        1. Groups cells by (file_idx, group_id) to maintain group semantic consistency
        2. Sorts cells within each file-group by local row index for physical locality
        3. Creates contiguous blocks as samples within each group
        4. Achieves significant I/O performance improvement while preserving replacement logic
        """
        if self.rng is None:
            self._initialize_rng()
            
        log.info(f"[{self.mode}] Generating locality-aware samples with group preservation...")
        start_time = time.time()
        
        self.samples = []
        active_groups_set = set(self.active_groups)
        
        if not active_groups_set:
            log.warning(f"No active groups for mode {self.mode}. Dataset will be empty.")
            return

        # FIXED: Group cells by (file_idx, group_id) to preserve group semantics
        cells_by_file_group = {}
        for i in range(self.metadata_cache.n_cells):
            group_id = self.metadata_cache.global_group_ids[i]
            if group_id in active_groups_set:
                file_idx, local_row_idx = self.metadata_cache.cell_to_file_mapping[i]
                file_group_key = (file_idx, group_id)
                if file_group_key not in cells_by_file_group:
                    cells_by_file_group[file_group_key] = []
                # Store (global_index, local_row_index) tuples
                cells_by_file_group[file_group_key].append((i, local_row_idx))

        # Process each (file, group) combination to create locality-aware samples
        total_samples_created = 0
        skipped_file_groups = 0
        
        for (file_idx, group_id), cells in cells_by_file_group.items():
            if len(cells) < self.sample_size:
                skipped_file_groups += 1
                continue
            
            # Sort by local row index to create physical locality within the group
            cells.sort(key=lambda x: x[1])
            
            # Extract sorted global indices
            sorted_global_indices = np.array([cell[0] for cell in cells])
            
            # Create contiguous samples (blocks) from sorted indices within this group
            n_samples = len(sorted_global_indices) // self.sample_size
            for i in range(n_samples):
                start = i * self.sample_size
                end = start + self.sample_size
                sample_indices = sorted_global_indices[start:end]
                
                # All cells in this sample belong to the same group by construction
                group_info = self.metadata_cache.group_mapping[group_id]
                
                # Create sample tuple
                self.samples.append((
                    group_id, 
                    group_info['original_id'], 
                    group_info['dataset_type'], 
                    sample_indices
                ))
                total_samples_created += 1
                
        # Shuffle all samples to ensure randomness across file-groups
        self.rng.shuffle(self.samples)
        
        generation_time = time.time() - start_time
        log.info(f"Generated {len(self.samples)} locality-aware samples in {generation_time:.2f}s for {self.mode} mode")
        log.info(f"  Processed {len(cells_by_file_group)} (file, group) combinations")
        log.info(f"  Skipped {skipped_file_groups} combinations with < {self.sample_size} cells")
        
        if self.samples:
            human_samples = sum(1 for _, _, dtype, _ in self.samples if dtype == 'human')
            drug_samples = sum(1 for _, _, dtype, _ in self.samples if dtype == 'drug')
            log.info(f"  Samples by type: {human_samples} human, {drug_samples} drug")
            log.info(f"  Average I/O locality: {total_samples_created} contiguous blocks preserving group semantics")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Enhanced __getitem__ with comprehensive replacement strategy support.
        
        This method now:
        1. Decides replacement strategy based on intra_file_replacement_prob
        2. Uses dataset-type-aware replacement rules (human vs drug)
        3. Performs efficient IO with precise loading
        4. Assembles final tensor with proper alignment
        """
        if self.rng is None:
            self._initialize_rng()
        
        # Get basic sample information
        group_id, original_group_id, dataset_type, cell_indices = self.samples[idx]

        # RESTORED: Decide replacement strategy based on probability
        use_intra_file_first = self.rng.random() < self.intra_file_replacement_prob

        # <<< USE OPTIMIZED VERSION >>>
        # Get replacement information using enhanced logic
        indices_to_keep, replacement_map = self.metadata_cache.find_replacement_cells(
            cell_indices, 
            self.rng,
            self.replacement_ratio,
            use_intra_file_first=use_intra_file_first,
        )

        # Assemble loading lists with proper alignment
        originals_to_be_replaced = np.array(list(replacement_map.keys()), dtype=np.int64)
        replacements = np.array(list(replacement_map.values()), dtype=np.int64)

        # Build input matrices: kept cells + original/replacement cells
        left_side_indices = np.concatenate([indices_to_keep, originals_to_be_replaced])
        right_side_indices = np.concatenate([indices_to_keep, replacements])

        cell_identities_str = self.metadata_cache.get_cell_identities(left_side_indices)

        ## v8 edit
        unique_types = self.rng.permutation(np.unique(cell_identities_str))        

        target_per_type = math.ceil(left_side_indices.shape[0] / len(unique_types))

        balanced_positions = []
        
        for t in unique_types:
            (positions_of_type,) = np.where(cell_identities_str == t)
            if len(positions_of_type) > target_per_type:
                subsampled_positions = self.rng.choice(positions_of_type, size=int(target_per_type), replace=False)
                balanced_positions.extend(subsampled_positions)
            else:
                balanced_positions.extend(positions_of_type)

        balanced_positions = np.array(balanced_positions, dtype=np.int64)

        n_needed = left_side_indices.shape[0] - len(balanced_positions)
        if n_needed > 0:
            reps = np.ones(len(balanced_positions), dtype=np.int64)
            extra_idx = self.rng.choice(len(balanced_positions), size=n_needed, replace=True)
            np.add.at(reps, extra_idx, 1)
            final_positions = np.repeat(balanced_positions, reps)
        else:
            final_positions = balanced_positions

        left_side_indices = left_side_indices[final_positions]
        right_side_indices = right_side_indices[final_positions]
        cell_identities_str = cell_identities_str[final_positions]

        _, first_occurrence_indices = np.unique(left_side_indices, return_index=True)
        
        temp_mask = np.zeros(left_side_indices.shape[0], dtype=bool)
        temp_mask[first_occurrence_indices] = True
        
        position_mask = torch.from_numpy(temp_mask)

        ## v8 edit ends
        
        # Convert strings to integer IDs using the map we created in __init__
        cell_type_ids = [self.identity_to_id_map[s] for s in cell_identities_str]
        cell_type_ids_tensor = torch.LongTensor(cell_type_ids)

        # Efficient IO: Load unique cells only once
        unique_indices_to_load, inverse_map = np.unique(
            np.concatenate([left_side_indices, right_side_indices]),
            return_inverse=True
        )
        unique_data = self.metadata_cache.load_expression_data(unique_indices_to_load)
        
        # Reconstruct full data matrix from unique loaded data
        full_data = unique_data[inverse_map]
        
        # Split into left and right side data
        n_left = len(left_side_indices)
        left_matrix = full_data[:n_left]
        right_matrix = full_data[n_left:]
            
        # Concatenate to create final features tensor
        ground_truth_tensor = torch.from_numpy(left_matrix).float()
        observed_tensor = torch.from_numpy(right_matrix).float()
        
        # ENHANCED: Build comprehensive metadata
        metadata = {
            'group_id': str(group_id),
            'original_group_id': str(original_group_id),
            'dataset_type': dataset_type,
            'sampled_cell_count': len(cell_indices),
            'n_kept': len(indices_to_keep),
            'n_replaced': len(replacement_map),
        }
        
        return ground_truth_tensor, observed_tensor, cell_type_ids_tensor, position_mask, metadata

    def resample_training_data(self):
        """Resample training data - call this for each epoch"""
        # NOTE: Locality-aware sampling makes resample less meaningful
        # since samples are now determined by physical file structure
        if self.mode == 'train' and self.resample:
            log.info(f"Resampling training data for worker...")
            self._generate_samples_by_locality()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return self.metadata_cache.get_cache_stats()
    
    def __len__(self) -> int:
        return len(self.samples)


class TestSamplerDataset(Dataset):
    """
    A simplified dataset for model inference (get_prediction, get_latent_representation).
    Uses the optimized loading methods from SimplifiedDatasetCache for H5AD files.

    This class takes a base dataset and a test dataset. It processes the test
    dataset in chunks of `n_kept_cells` and, for each chunk, samples a
    corresponding set of cells from the base dataset. It handles upsampling if
    either dataset has insufficient cells and ensures all gene expression
    matrices are aligned to a common `target_genes` list.
    """

    def __init__(
        self,
        base_adata_or_path: Union[str, anndata.AnnData],
        test_adata_or_path: Union[str, anndata.AnnData],
        genelist_path: str,
        n_cells: int,
        n_kept_cells: int,
        eval_col: str = 'disease',
        control_cond: str = 'control',
        split_col: str = 'donor_id',
        cell_type_col: str = 'cell_type',
        mode: str = 'in-context',
        setting: str = 'representation',
    ):
        """
        Initializes the InferenceDataset.

        Args:
            base_adata_or_path: The base AnnData object or path to .h5ad file,
                                used to provide context cells.
            test_adata_or_path: The test AnnData object or path to .h5ad file,
                                containing cells for prediction.
            genelist_path: Path to pickle file containing gene list.
            n_cells: The number of base cells per sample. This
                                     should match the model's `n_cells` parameter.
            n_kept_cells: The number of test cells per sample. This should match
                          the model's `n_kept_cell` parameter.
            cell_type_col: The column name in `base_adata.obs` that
                                contains cell type annotations.
        """
        log.info("Initializing InferenceDataset...")
        self.target_genes = load_gene_list(genelist_path)
        self.n_genes = len(self.target_genes)
        self.n_cells = n_cells
        self.n_kept_cells = n_kept_cells
        self.eval_col = eval_col
        self.control_cond = control_cond
        self.split_col = split_col
        self.cell_type_col = cell_type_col
        self.mode = mode

        # Process base data
        log.info("Processing 'base' data...")
        if isinstance(base_adata_or_path, str):
            self.base_is_h5ad = True
            self.base_path = base_adata_or_path
            self.base_file_info = self._setup_h5ad_file_info(base_adata_or_path)
            # Pre-load cell types once to avoid repeated I/O
            self.base_eval_vals = self._load_meta_col_from_h5ad(base_adata_or_path, eval_col)
            self.base_split_vals = self._load_meta_col_from_h5ad(base_adata_or_path, split_col)
            self.base_cell_types = self._load_meta_col_from_h5ad(base_adata_or_path, cell_type_col)
            self.base_expr = None  # Will be loaded on demand
        else:
            self.base_is_h5ad = False
            self.base_adata = base_adata_or_path
            self.base_expr, self.base_eval_vals, self.base_split_vals, self.base_cell_types, self.base_gene_map = self._load_and_process_adata(
                base_adata_or_path, self.target_genes, eval_col, split_col, cell_type_col
            )
        
        self.n_base_cells = self.base_file_info['n_cells'] if self.base_is_h5ad else self.base_expr.shape[0]
        log.info(f"Loaded {self.n_base_cells} base cells.")

        # Process test data
        log.info("Processing 'test' data...")
        if isinstance(test_adata_or_path, str):
            self.test_is_h5ad = True
            self.test_path = test_adata_or_path
            self.test_file_info = self._setup_h5ad_file_info(test_adata_or_path)
            self.test_eval_vals = self._load_meta_col_from_h5ad(test_adata_or_path, eval_col)
            self.test_split_vals = self._load_meta_col_from_h5ad(test_adata_or_path, split_col)
            self.test_cell_types = self._load_meta_col_from_h5ad(test_adata_or_path, cell_type_col)
            self.test_expr = None  # Will be loaded on demand
        else:
            self.test_is_h5ad = False
            self.test_adata = test_adata_or_path
            self.test_expr, self.test_eval_vals, self.test_split_vals, self.test_cell_types, self.test_gene_map = self._load_and_process_adata(
                test_adata_or_path, self.target_genes, eval_col, split_col, cell_type_col
            )
        test_control_mask = self.test_eval_vals == self.control_cond
        self.test_control_indices = np.where(test_control_mask)[0]
        self.test_treat_indices = np.where(~test_control_mask)[0]
        self.n_test_control_cells = len(self.test_control_indices)
        self.n_test_treat_cells = len(self.test_treat_indices)
        
        self.n_test_cells = self.test_file_info['n_cells'] if self.test_is_h5ad else self.test_expr.shape[0]
        self.n_total_test_cells = self.n_test_cells  # Store original number for output shaping
        log.info(f"Loaded {self.n_test_cells} test cells.")

        control_mask = self.base_eval_vals == self.control_cond
        self.base_control_indices = np.where(control_mask)[0]
        self.base_treat_indices = np.where(~control_mask)[0]
        self.n_control_cells = len(self.base_control_indices)
        self.n_treat_cells = len(self.base_treat_indices)

        # Calculate the total number of samples. In the in-context mode, test data are (n_cells - n_kept_cells) in replacement data. 
        # In the personalized mode, test data are n_kept_cells in orig data.
        if setting == 'representation':
            if mode == 'in-context':
                self.n_samples = max(math.ceil(self.n_control_cells / self.n_kept_cells),math.ceil(self.n_treat_cells / self.n_kept_cells),math.ceil(self.n_test_cells / (self.n_cells - self.n_kept_cells))) 
            else:
                self.n_samples = max(math.ceil(self.n_control_cells / self.n_kept_cells),math.ceil(self.n_treat_cells / (self.n_cells - self.n_kept_cells)),math.ceil(self.n_test_cells / self.n_kept_cells))
        else:
            if mode == 'in-context':
                self.n_samples = math.ceil(self.n_test_cells / (self.n_cells - self.n_kept_cells))
            else:
                self.n_samples = math.ceil(self.n_test_cells / self.n_kept_cells)
        log.info(f"Will generate {self.n_samples} inference samples.")
    
    def _load_meta_col_from_h5ad(self, h5ad_path: str, meta_col: str) -> np.ndarray:
        """Pre-load all cell types from H5AD file to avoid repeated I/O"""
        log.info(f"Pre-loading {meta_col} from {h5ad_path}")
        
        f = _get_h5_handle(h5ad_path)
        obs = f["obs"]
        n_cells = f["X"].shape[0] if hasattr(f["X"], 'shape') else f["X"].attrs['shape'][0]
        
        if meta_col is None:
            return np.array(["unknown"] * n_cells)
        
        if meta_col in obs:
            meta_ds = obs[meta_col]
            if "categories" in meta_ds:
                categories = safe_decode_array(meta_ds["categories"][:])
                codes = meta_ds["codes"][:]
                all_meta = categories[codes]
            else:
                all_meta = safe_decode_array(meta_ds[:])
        else:
            log.warning(f"Metadata column '{meta_col}' not found")
            all_meta = np.array(["unknown"] * n_cells)
        
        return all_meta

    def _setup_h5ad_file_info(self, h5ad_path: str) -> Dict[str, Any]:
        """Setup file info for H5AD file using the same logic as SimplifiedDatasetCache"""
        log.info(f"Setting up H5AD file info for {h5ad_path}")
        
        f = _get_h5_handle(h5ad_path)
        
        # For inference, we typically don't filter by organism
        # Assume all cells are valid
        n_cells = f["X"].shape[0] if hasattr(f["X"], 'shape') else f["X"].attrs['shape'][0]
        human_mask = np.ones(n_cells, dtype=bool)  # Keep all cells for inference
        n_human_cells = n_cells
        
        # Get gene information using the existing helper function
        use_raw = "raw" in f and "var" in f["raw"]
        gene_names = get_gene_names_from_h5(f, None, use_raw=use_raw)  # Use default gene names
        
        if gene_names is None:
            raise ValueError("Could not find gene names in the file")
        
        # Create gene mapping
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        gene_mapping = {}
        found_genes = 0
        for target_idx, gene in enumerate(self.target_genes):
            if gene in gene_to_idx:
                gene_mapping[target_idx] = gene_to_idx[gene]
                found_genes += 1
        
        if found_genes == 0:
            raise ValueError("No target genes found in the file")
        
        # Store file info in the same format as SimplifiedDatasetCache
        if use_raw:
            X_group = f["raw"]["X"]
        else:
            X_group = f["X"]
        
        file_info = {
            'path': h5ad_path,
            'n_cells': n_human_cells,
            'n_genes': len(gene_names),
            'use_raw': use_raw,
            'gene_mapping': gene_mapping,
            'organism_mask': human_mask,
            'is_sparse': isinstance(X_group, h5py.Group) and 'data' in X_group and 'indices' in X_group,
            'found_genes': found_genes
        }
        
        log.info(f"H5AD file setup: {n_human_cells} cells, {found_genes}/{len(self.target_genes)} target genes")
        return file_info

    def _load_and_process_adata(
        self,
        adata: anndata.AnnData,
        target_genes: List[str],
        eval_col: Optional[str], 
        split_col: Optional[str],
        cell_type_col: Optional[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[int, int]]:
        """
        A private helper to load an AnnData object and prepare its contents.

        Args:
            adata: AnnData object.
            target_genes: The list of genes to align with.
            cell_type_col: The column in .obs for cell type annotations.

        Returns:
            A tuple containing:
            - expression_matrix (np.ndarray): The cells x genes expression matrix.
            - cell_types (np.ndarray or None): Array of cell type strings.
            - gene_map (Dict[int, int]): A map from target gene index to source gene index.
        """
        # Ensure expression matrix is a NumPy array
        if hasattr(adata.X, 'toarray'):
            expression_matrix = adata.X.toarray()
        else:
            expression_matrix = np.asarray(adata.X)

        # Extract cell types if a column is provided
        eval_vals = None
        if eval_col:
            if eval_col in adata.obs.columns:
                eval_vals = adata.obs[eval_col].values.astype(str)
            else:
                log.warning(f"Eval column '{eval_col}' not found in .obs. "
                            f"Behavior may not be expected.")
                eval_vals = np.array(["unknown"] * adata.n_obs)
        else:
            eval_vals = np.array(["unknown"] * adata.n_obs)

        split_vals = None
        if split_col:
            if split_col in adata.obs.columns:
                split_vals = adata.obs[split_col].values.astype(str)
            else:
                log.warning(f"Split column '{split_col}' not found in .obs. "
                            f"Fallback to loading in order.")
                split_vals = np.array(["unknown"] * adata.n_obs)
        else:
            split_vals = np.array(["unknown"] * adata.n_obs)
        
        cell_types = None
        if cell_type_col:
            if cell_type_col in adata.obs.columns:
                cell_types = adata.obs[cell_type_col].values.astype(str)
            else:
                log.warning(f"Cell type column '{cell_type_col}' not found in .obs. "
                            f"Cell types will not be available.")
                cell_types = np.array(["unknown"] * adata.n_obs)
        else:
            cell_types = np.array(["unknown"] * adata.n_obs)

        # Create a map to align genes with the target list
        source_gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}
        gene_map = {}
        genes_found = 0
        for i, gene in enumerate(target_genes):
            if gene in source_gene_to_idx:
                gene_map[i] = source_gene_to_idx[gene]
                genes_found += 1

        log.info(f"Found {genes_found}/{len(target_genes)} target genes in the provided AnnData.")
        if genes_found == 0:
            raise ValueError("No target genes were found in the AnnData object. "
                             "Please check if gene names (e.g., HUGO symbols) match.")

        return expression_matrix, eval_vals, split_vals, cell_types, gene_map
    
    def _load_expression_data_from_h5ad(self, file_info: Dict[str, Any], local_indices: np.ndarray) -> np.ndarray:
        """
        Load expression data from H5AD file using the optimized method from SimplifiedDatasetCache.
        This is essentially a copy of SimplifiedDatasetCache.load_expression_data_from_file
        but adapted for single file usage.
        """
        # Sort indices for better h5py reading efficiency
        sort_order = np.argsort(local_indices)
        sorted_local_indices = local_indices[sort_order]

        try:
            f = _get_h5_handle(file_info['path'])
            X_group = f["raw"]["X"] if file_info['use_raw'] else f["X"]
                
            # Get absolute row indices (in this case, same as local since we don't filter organisms)
            absolute_indices_to_load = sorted_local_indices

            # Create result matrix mapped to target genes
            mapped_matrix = np.zeros((len(local_indices), self.n_genes), dtype=np.float32)
            gene_mapping = file_info['gene_mapping']
            target_indices_in_result = np.array(list(gene_mapping.keys()))
            source_indices_in_file = np.array(list(gene_mapping.values()))

            if file_info['is_sparse']:
                # Handle sparse matrix (use the optimized logic from SimplifiedDatasetCache)
                attrs = dict(X_group.attrs)
                encoding_type = attrs.get("encoding-type", "csr_matrix")
                log.debug(f"Processing sparse matrix with encoding: {encoding_type}")
                
                if encoding_type == "csr_matrix":
                    # CSR format: efficient row slicing
                    log.debug(f"Using optimized CSR row reading for {len(absolute_indices_to_load)} rows")
                    
                    indptr_h5 = X_group["indptr"]
                    data_h5 = X_group["data"]
                    indices_h5 = X_group["indices"]

                    # Get all required indptr start and end points
                    starts = indptr_h5[absolute_indices_to_load]
                    ends = indptr_h5[absolute_indices_to_load + 1]
                    
                    # Detect physical data block boundaries
                    block_breaks = np.where(ends[:-1] != starts[1:])[0] + 1
                    row_blocks = np.split(np.arange(len(absolute_indices_to_load)), block_breaks)

                    # Read data block by block to minimize I/O operations
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

                    # Construct sparse matrix from collected slices
                    if not data_slices:
                        sparse_subset = sp.csr_matrix((len(local_indices), file_info['n_genes']), dtype=np.float32)
                    else:
                        all_data = np.concatenate(data_slices)
                        all_indices = np.concatenate(indices_slices)

                        new_indptr = np.zeros(len(local_indices) + 1, dtype=indptr_h5.dtype)
                        if row_order.size:  # Guard against empty row_order
                            row_lengths_in_order = ends[row_order] - starts[row_order]
                            new_indptr[1:] = np.cumsum(row_lengths_in_order)
                        # If row_order is empty, new_indptr stays all zeros (valid for empty CSR)
                        
                        sparse_subset = sp.csr_matrix(
                            (all_data, all_indices, new_indptr),
                            shape=(len(local_indices), file_info['n_genes']),
                            dtype=np.float32
                        )
                        
                    # Select target gene columns and convert to dense
                    expr_subset = sparse_subset[:, source_indices_in_file].toarray()
                    mapped_matrix[:, target_indices_in_result] = expr_subset.astype(np.float32)

                elif encoding_type == "csc_matrix":
                    # CSC format: column-oriented reading strategy
                    log.debug(f"Using CSC column-oriented reading for {len(absolute_indices_to_load)} rows")
                    
                    data_h5 = X_group["data"]
                    indices_h5 = X_group["indices"] 
                    indptr_h5 = X_group["indptr"]
                    
                    # Get actual matrix shape
                    n_genes_in_file = len(indptr_h5) - 1
                    n_cells_in_file = file_info['n_cells']
                    
                    # Adaptive strategy selection
                    load_threshold = min(1000, n_cells_in_file * 0.1)
                    
                    if len(absolute_indices_to_load) <= load_threshold:
                        # Strategy A: Row-by-row reading
                        log.debug(f"Using row-by-row CSC reading for {len(absolute_indices_to_load)} rows")
                        
                        result_matrix = np.zeros((len(absolute_indices_to_load), n_genes_in_file), dtype=np.float32)
                        
                        # Create mapping from absolute_index to result_matrix row number
                        abs_to_res_map = {abs_idx: i for i, abs_idx in enumerate(absolute_indices_to_load)}
                        
                        # Process each target gene (column)
                        for gene_idx in range(n_genes_in_file):
                            col_start = indptr_h5[gene_idx]
                            col_end = indptr_h5[gene_idx + 1]
                            
                            if col_end > col_start:
                                col_cell_indices = indices_h5[col_start:col_end]
                                col_values = data_h5[col_start:col_end]
                                
                                isin_mask = np.isin(col_cell_indices, absolute_indices_to_load, assume_unique=True)
                                
                                if np.any(isin_mask):
                                    matched_abs_indices = col_cell_indices[isin_mask]
                                    matched_values = col_values[isin_mask]
                                    
                                    for abs_idx, value in zip(matched_abs_indices, matched_values):
                                        result_matrix_row = abs_to_res_map[abs_idx]
                                        result_matrix[result_matrix_row, gene_idx] = value
                    
                    else:
                        # Strategy B: Load full matrix then slice
                        log.debug(f"Using full matrix load for {len(absolute_indices_to_load)} rows")
                        
                        data = data_h5[:]
                        indices = indices_h5[:]
                        indptr = indptr_h5[:]
                        
                        csc_matrix_transposed = sp.csc_matrix(
                            (data, indices, indptr), 
                            shape=(n_genes_in_file, n_cells_in_file)
                        )
                        
                        csr_matrix = csc_matrix_transposed.T
                        subset_sparse = csr_matrix[absolute_indices_to_load, :]
                        result_matrix = subset_sparse.toarray()
                    
                    # Map to target genes
                    mapped_matrix[:, target_indices_in_result] = result_matrix[:, source_indices_in_file].astype(np.float32)
                    
                else:
                    raise ValueError(f"Unknown sparse encoding type '{encoding_type}' in {file_info['path']}")
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

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
        """
        Retrieves one sample for inference. Different behavior depending on the mode.

        Args:
            idx: The index of the sample.

        Returns:
            A tuple (replacement_features, orig_features, metadata):
            - replacement_features (torch.Tensor): Shape (n_cells, n_genes).
            - orig_features (torch.Tensor): Shape (n_kept_cells, n_genes).
            - metadata (Dict): A dictionary containing metadata for cells.
        """
        if self.mode == 'in-context':
            n_test_cells_per_sample = self.n_cells - self.n_kept_cells
            n_base_controls_per_sample = self.n_kept_cells
            n_base_treats_per_sample = self.n_kept_cells
        else:
            n_test_cells_per_sample = self.n_kept_cells
            n_base_controls_per_sample = self.n_kept_cells
            n_base_treats_per_sample = self.n_cells - self.n_kept_cells

        # --- 1. Prepare test data (the cells to be predicted) ---
        test_start_idx = idx * n_test_cells_per_sample
        test_end_idx = min(test_start_idx + n_test_cells_per_sample, self.n_test_cells)
        test_indices = np.arange(test_start_idx, test_end_idx)

        # Upsample if the last batch is smaller than required
        if len(test_indices) < n_test_cells_per_sample:
            if len(test_indices) == 0:
                raise IndexError("Index out of bounds with no cells to sample from.")
            
            candidates = test_indices
            n_needed = n_test_cells_per_sample
            
            if len(candidates) < n_needed:
                # Use replacement with repetition if not enough candidates
                indices = np.arange(n_needed) % len(candidates)
                test_indices = candidates[indices]

        if self.test_is_h5ad:
            test_expr_slice = self._load_expression_data_from_h5ad(self.test_file_info, test_indices)
            test_eval_vals_slice = self.test_eval_vals[test_indices]
            test_split_vals_slice = self.test_split_vals[test_indices]
            test_cell_types_slice = self.test_cell_types[test_indices]
        else:
            test_expr_slice = self.test_expr[test_indices, :]
            test_eval_vals_slice = self.test_eval_vals[test_indices]
            test_split_vals_slice = self.test_split_vals[test_indices]
            test_cell_types_slice = self.test_cell_types[test_indices]

        # --- 2. Prepare base data (separate controls and treatments) ---
        # Find control and non-control indices in base data
        control_indices = self.base_control_indices
        treat_indices = self.base_treat_indices
        
        if len(control_indices) > 0:
            control_start_idx = (idx * n_base_controls_per_sample) % len(control_indices)
            control_end_idx = min(control_start_idx + n_base_controls_per_sample, len(control_indices))
            selected_control_indices = control_indices[control_start_idx:control_end_idx]
            
            if len(selected_control_indices) < n_base_controls_per_sample:
                if len(selected_control_indices) == 0:
                    raise IndexError("Index out of bounds with no cells to sample from.")
                indices = np.arange(n_base_controls_per_sample) % len(control_indices)
                selected_control_indices = control_indices[indices]
        else:
            selected_control_indices = np.array([], dtype=int)
        
        # Sample treatment cells
        if len(treat_indices) > 0:
            treat_start_idx = (idx * n_base_treats_per_sample) % len(treat_indices)
            treat_end_idx = min(treat_start_idx + n_base_treats_per_sample, len(treat_indices))
            selected_treat_indices = treat_indices[treat_start_idx:treat_end_idx]
            
            if len(selected_treat_indices) < n_base_treats_per_sample:
                if len(selected_control_indices) == 0:
                    raise IndexError("Index out of bounds with no cells to sample from.")
                indices = np.arange(n_base_treats_per_sample) % len(treat_indices)
                selected_treat_indices = treat_indices[indices]
        else:
            selected_treat_indices = np.array([], dtype=int)

        # Combine control and treatment indices
        base_indices = np.concatenate([selected_control_indices, selected_treat_indices])

        if self.base_is_h5ad:
            base_expr_slice = self._load_expression_data_from_h5ad(self.base_file_info, base_indices)
            base_eval_vals_slice = self.base_eval_vals[base_indices]
            base_split_vals_slice = self.base_split_vals[base_indices]
            base_cell_types_slice = self.base_cell_types[base_indices]
        else:
            base_expr_slice = self.base_expr[base_indices, :]
            base_eval_vals_slice = self.base_eval_vals[base_indices]
            base_split_vals_slice = self.base_split_vals[base_indices]
            base_cell_types_slice = self.base_cell_types[base_indices]

        # --- 3. Align genes for both matrices ---
        # For H5AD files, genes are already aligned in the loading functions
        if self.test_is_h5ad:
            final_test_expr = test_expr_slice
        else:
            # Align test genes
            final_test_expr = np.zeros((n_test_cells_per_sample, self.n_genes), dtype=np.float32)
            test_target_indices = np.array(list(self.test_gene_map.keys()))
            test_source_indices = np.array(list(self.test_gene_map.values()))
            final_test_expr[:, test_target_indices] = test_expr_slice[:, test_source_indices]
            
        final_test_expr = torch.from_numpy(final_test_expr).float()

        if self.base_is_h5ad:
            final_base_expr = base_expr_slice
        else:
            # Align base genes
            final_base_expr = np.zeros((len(base_indices), self.n_genes), dtype=np.float32)
            base_target_indices = np.array(list(self.base_gene_map.keys()))
            base_source_indices = np.array(list(self.base_gene_map.values()))
            final_base_expr[:, base_target_indices] = base_expr_slice[:, base_source_indices]
            
        final_control_expr = torch.from_numpy(final_base_expr[:len(selected_control_indices)]).float()
        final_treat_expr = torch.from_numpy(final_base_expr[len(selected_control_indices):]).float()

        # --- 4. Construct final tensors based on mode ---
        if self.mode == 'in-context':
            replacement_features = torch.cat([final_control_expr, final_test_expr], dim=0)
            orig_features = final_treat_expr
            
            # Combine metadata for replacement_features
            replacement_eval_vals = np.concatenate([base_eval_vals_slice[:self.n_kept_cells], test_eval_vals_slice])
            replacement_split_vals = np.concatenate([base_split_vals_slice[:self.n_kept_cells], test_split_vals_slice])
            replacement_cell_types = np.concatenate([base_cell_types_slice[:self.n_kept_cells], test_cell_types_slice])
            
            metadata = {
                #'eval_vals': replacement_eval_vals.tolist(),
                #'split_vals': replacement_split_vals.tolist(),
                'cell_type': replacement_cell_types.tolist(),
                #'orig_eval_vals': base_eval_vals_slice[self.n_kept_cells:].tolist(),
                #'orig_split_vals': base_split_vals_slice[self.n_kept_cells:].tolist(),
                #'orig_cell_type': base_cell_types_slice[self.n_kept_cells:].tolist()
            }
        else:
            replacement_features = torch.from_numpy(final_base_expr).float()
            orig_features = final_test_expr
            
            metadata = {
                #'eval_vals': base_eval_vals_slice.tolist(),
                #'split_vals': base_split_vals_slice.tolist(),
                'cell_type': base_cell_types_slice.tolist(),
                #'orig_eval_vals': test_eval_vals_slice.tolist(),
                #'orig_split_vals': test_split_vals_slice.tolist(),
                #'orig_cell_type': test_cell_types_slice.tolist()
            }

        return (
            replacement_features,
            orig_features,
            metadata
        )


def create_train_val_test_datasets(
    dataset_configs: List[DatasetConfig],
    target_genes: Optional[List[str]] = None,
    genelist_path: Optional[str] = None,
    sample_size: int = 128,
    replacement_ratio: float = 0.25,
    intra_file_replacement_prob: float = 1.0,  # RESTORED
    min_cells_per_group: int = 128,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_state: int = 42,
    cache_file: Optional[str] = None,
    max_memory_gb: Optional[float] = None,  # RESTORED
    cache_ratio: float = 0.3,  # RESTORED
    block_size: int = 1000  # RESTORED
) -> Tuple[MultiDatasetSplittableDataset, MultiDatasetSplittableDataset, MultiDatasetSplittableDataset]:
    """
    Create train, validation, and test datasets with comprehensive parameter support.
    
    Args:
        dataset_configs: List of dataset configurations
        target_genes: List of target gene names, if None will load from genelist_path
        genelist_path: Path to saved gene list file
        sample_size: Number of cells to sample from each group
        replacement_ratio: Fraction of cells to replace (0.0-1.0)
        intra_file_replacement_prob: Probability of using intra-file replacement first (0.0-1.0)
        min_cells_per_group: Min cells required per group
        test_ratio: Fraction of groups for testing
        val_ratio: Fraction of groups for validation
        random_state: Random seed
        cache_file: Optional path to cache metadata
        max_memory_gb: Maximum memory usage in GB (for compatibility)
        cache_ratio: Fraction of memory to use for caching (for compatibility)
        block_size: Number of rows per cache block (for compatibility)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Create a temporary base dataset instance just to perform the group splitting logic
    # This avoids instantiating the full dataset three times
    log.info("Creating base dataset for group splitting...")
    base_dataset = MultiDatasetSplittableDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=min_cells_per_group, # Use min_cells_per_group for splitting logic
        mode='train', # a placeholder mode
        random_state=random_state,
        resample=False,
        cache_file=cache_file,
        max_memory_gb=max_memory_gb,
        cache_ratio=cache_ratio,
        block_size=block_size
    )
    
    # Perform group splitting using the base instance
    base_dataset._split_groups(test_ratio, val_ratio)
    train_groups = base_dataset.train_groups
    val_groups = base_dataset.val_groups
    test_groups = base_dataset.test_groups

    # Now create the actual datasets using the pre-computed splits
    # This ensures they share the same metadata cache instance via the singleton
    log.info("Creating training dataset...")
    train_dataset = MultiDatasetSplittableDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        replacement_ratio=replacement_ratio,
        intra_file_replacement_prob=intra_file_replacement_prob,
        min_cells_per_group=min_cells_per_group,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        mode='train',
        random_state=random_state,
        resample=True,  # Enable resampling for training data
        cache_file=cache_file,
        max_memory_gb=max_memory_gb,
        cache_ratio=cache_ratio,
        block_size=block_size
    )
    
    log.info("Creating validation dataset...")
    val_dataset = MultiDatasetSplittableDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        replacement_ratio=replacement_ratio,
        intra_file_replacement_prob=intra_file_replacement_prob,
        min_cells_per_group=min_cells_per_group,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        mode='val',
        random_state=random_state,
        resample=False,
        cache_file=cache_file,
        max_memory_gb=max_memory_gb,
        cache_ratio=cache_ratio,
        block_size=block_size
    )
    
    log.info("Creating test dataset...")
    test_dataset = MultiDatasetSplittableDataset(
        dataset_configs=dataset_configs,
        target_genes=target_genes,
        genelist_path=genelist_path,
        sample_size=sample_size,
        replacement_ratio=replacement_ratio,
        intra_file_replacement_prob=intra_file_replacement_prob,
        min_cells_per_group=min_cells_per_group,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        mode='test',
        random_state=random_state,
        resample=False,
        cache_file=cache_file,
        max_memory_gb=max_memory_gb,
        cache_ratio=cache_ratio,
        block_size=block_size
    )
    
    return train_dataset, val_dataset, test_dataset


# Example usage functions
def create_example_configs():
    """Example of how to create dataset configurations"""
    
    # Human dataset configuration
    human_config = DatasetConfig(
        path="/path/to/human_pbmc_data",
        type="human",
        donor_col="donor_id",
        cell_type_col="cell_type",
        filter_organism=True
    )
    
    # Drug screening dataset configuration
    drug_config = DatasetConfig(
        path="/path/to/drug_screening_data",
        type="drug", 
        condition_col="drug_condition",
        cell_line_col="cell_line",
        control_condition="dmso",
        filter_organism=False
    )
    
    return [human_config, drug_config]


def compute_and_save_hvg_union(
    dataset_configs: List[DatasetConfig],
    output_path: str,
    n_top_genes: int = 1000,
    theta: float = 100.0,
    max_cells_per_file: int = 10000,
    random_state: int = 42
) -> List[str]:
    """
    Compute HVG union and save to file. This function decouples HVG computation from dataset creation.
    
    Args:
        dataset_configs: List of dataset configurations
        output_path: Path to save the gene list
        n_top_genes: Number of top HVGs to select per file
        theta: Overdispersion parameter for negative binomial
        max_cells_per_file: Maximum number of cells to use per file
        random_state: Random seed for reproducible subsampling
        
    Returns:
        List of gene names (union of all HVGs)
    """
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
    replacement_ratio: float = 0.25,
    intra_file_replacement_prob: float = 1.0,  # RESTORED
    min_cells_per_group: int = 128,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_state: int = 42,
    cache_file: Optional[str] = None,
    max_memory_gb: Optional[float] = None,  # RESTORED
    cache_ratio: float = 0.3,  # RESTORED
    block_size: int = 1000  # RESTORED
) -> Tuple[MultiDatasetSplittableDataset, MultiDatasetSplittableDataset, MultiDatasetSplittableDataset]:
    """
    Create datasets from a pre-computed gene list file with full parameter support.
    
    Args:
        dataset_configs: List of dataset configurations
        genelist_path: Path to saved gene list file
        sample_size: Number of cells to sample from each group
        replacement_ratio: Fraction of cells to replace (0.0-1.0)
        intra_file_replacement_prob: Probability of using intra-file replacement first (0.0-1.0)
        min_cells_per_group: Min cells required per group
        test_ratio: Fraction of groups for testing
        val_ratio: Fraction of groups for validation
        random_state: Random seed
        cache_file: Optional path to cache metadata
        max_memory_gb: Maximum memory usage in GB (for compatibility)
        cache_ratio: Fraction of memory to use for caching (for compatibility)
        block_size: Number of rows per cache block (for compatibility)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    log.info(f"Creating datasets from gene list: {genelist_path}")
    
    return create_train_val_test_datasets(
        dataset_configs=dataset_configs,
        target_genes=None,  # Load from file
        genelist_path=genelist_path,
        sample_size=sample_size,
        replacement_ratio=replacement_ratio,
        intra_file_replacement_prob=intra_file_replacement_prob,
        min_cells_per_group=min_cells_per_group,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
        cache_file=cache_file,
        max_memory_gb=max_memory_gb,
        cache_ratio=cache_ratio,
        block_size=block_size
    )