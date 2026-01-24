"""Dataset classes for perturbation prediction experiments."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tomllib
except ImportError:
    import tomli as tomllib

log = logging.getLogger(__name__)

__all__ = [
    "PerturbationDataset",
    "PerturbationDatasetConfig",
    "create_perturbation_datasets",
    "create_fold_datasets",
    "get_fold_info",
]


@dataclass
class PerturbationDatasetConfig:
    """Configuration for perturbation dataset.

    For 4-fold cross-validation using TOML files:
    - Set split_toml_dir to the directory containing {cell_line}.toml files
    - Set fold to 0-3 to specify which cell line is held out for testing
    - Fold mapping: 0=K562, 1=HepG2, 2=Jurkat, 3=RPE1
    """

    data_path: str
    perturbation_col: str = "gene"
    control_label: str = "non-targeting"
    cell_line_col: str = "cell_line"
    n_cells: int = 128
    genelist_path: Optional[str] = None
    split_toml_dir: Optional[str] = None
    cell_lines: List[str] = field(default_factory=lambda: ["K562", "HepG2", "Jurkat", "RPE1"])
    fold: Optional[int] = None  # 0-3 for 4-fold CV, None to combine all


def parse_split_toml(toml_path: str) -> Set[str]:
    """Parse a TOML file to extract held-out test perturbations.

    Expected format:
        [fewshot."replogle.<cell_line>"]
        test = ["gene1", "gene2", ...]

    Args:
        toml_path: Path to the TOML file.

    Returns:
        Set of gene names that should be held out for testing.
    """
    test_genes = set()

    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)

        # Look for fewshot section
        if "fewshot" in config:
            for key, value in config["fewshot"].items():
                if isinstance(value, dict) and "test" in value:
                    test_list = value["test"]
                    if isinstance(test_list, list):
                        test_genes.update(test_list)
                        log.info(f"Found {len(test_list)} test genes in {key}")

    except Exception as e:
        log.warning(f"Failed to parse TOML file {toml_path}: {e}")

    return test_genes


def load_split_configs(split_toml_dir: str, cell_lines: List[str]) -> Dict[str, Set[str]]:
    """Load split configurations from TOML files.

    Args:
        split_toml_dir: Directory containing TOML files.
        cell_lines: List of cell lines to load configs for.

    Returns:
        Dictionary mapping cell line names to sets of test gene names.
    """
    split_dir = Path(split_toml_dir)
    splits = {}

    for cell_line in cell_lines:
        toml_name = f"{cell_line.lower()}.toml"
        toml_path = split_dir / toml_name

        if toml_path.exists():
            test_genes = parse_split_toml(str(toml_path))
            splits[cell_line] = test_genes
            log.info(f"Loaded {len(test_genes)} test genes for {cell_line}")
        else:
            log.warning(f"TOML file not found: {toml_path}")
            splits[cell_line] = set()

    return splits


class PerturbationDataset(Dataset):
    """Dataset for perturbation prediction experiments.

    Loads data from an h5ad file containing both control and perturbed cells.
    Samples pairs of control cells and perturbed cells for flow matching training.

    Args:
        config: Dataset configuration.
        mode: One of 'train', 'val', or 'test'.
        random_state: Random seed for reproducibility.
        test_split_ratio: Ratio of test perturbations if no TOML provided.
        val_split_ratio: Ratio of validation perturbations.
    """

    def __init__(
        self,
        config: PerturbationDatasetConfig,
        mode: str = "train",
        random_state: int = 42,
        test_split_ratio: float = 0.2,
        val_split_ratio: float = 0.1,
    ) -> None:
        self.config = config
        self.mode = mode
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Load the h5ad file
        self._load_data()

        # Build perturbation vocabulary
        self._build_perturbation_vocab()

        # Determine train/val/test splits
        self._setup_splits(test_split_ratio, val_split_ratio)

        # Generate sample indices
        self._generate_samples()

        log.info(
            f"PerturbationDataset initialized: mode={mode}, "
            f"{len(self.samples)} samples, {len(self.perturbation_to_id)} perturbations"
        )

    def _load_data(self) -> None:
        """Load h5ad data and extract relevant information."""
        import anndata as ad

        log.info(f"Loading data from {self.config.data_path}")
        self.adata = ad.read_h5ad(self.config.data_path, backed="r")

        # Get perturbation and cell line info
        self.perturbations = self.adata.obs[self.config.perturbation_col].values
        self.n_cells_total = len(self.perturbations)

        if self.config.cell_line_col in self.adata.obs.columns:
            # Normalize cell line names to uppercase for consistent matching
            raw_cell_lines = self.adata.obs[self.config.cell_line_col].values
            self.cell_lines = np.array([str(cl).upper() for cl in raw_cell_lines])
        else:
            self.cell_lines = np.array(["UNKNOWN"] * self.n_cells_total)

        # Get gene names
        self.gene_names = self.adata.var_names.values
        self.n_genes = len(self.gene_names)

        # Identify control cells
        self.control_mask = self.perturbations == self.config.control_label
        self.control_indices = np.where(self.control_mask)[0]

        log.info(
            f"Loaded {self.n_cells_total} cells, {self.n_genes} genes, "
            f"{len(self.control_indices)} control cells"
        )

        # If we have a genelist, build the mapping
        self.gene_mapping = None
        self.target_genes = None
        if self.config.genelist_path:
            self._setup_gene_mapping()

    def _setup_gene_mapping(self) -> None:
        """Setup mapping from target genes to dataset genes."""
        with open(self.config.genelist_path, "rb") as f:
            self.target_genes = pickle.load(f)

        gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(self.gene_names)}

        self.gene_mapping = {}
        found = 0
        for target_idx, gene in enumerate(self.target_genes):
            gene_upper = gene.upper()
            if gene_upper in gene_to_idx:
                self.gene_mapping[target_idx] = gene_to_idx[gene_upper]
                found += 1

        log.info(f"Gene mapping: {found}/{len(self.target_genes)} target genes found")

    def _build_perturbation_vocab(self) -> None:
        """Build vocabulary mapping perturbation names to IDs."""
        # Get unique perturbations (excluding control)
        unique_perts = np.unique(self.perturbations)
        unique_perts = [p for p in unique_perts if p != self.config.control_label]

        # Create mapping (reserve 0 for padding/unknown)
        self.perturbation_to_id = {pert: idx + 1 for idx, pert in enumerate(unique_perts)}
        self.id_to_perturbation = {idx: pert for pert, idx in self.perturbation_to_id.items()}

        # Build index of cells per perturbation
        self.pert_to_indices = {}
        for pert in unique_perts:
            self.pert_to_indices[pert] = np.where(self.perturbations == pert)[0]

        log.info(f"Found {len(self.perturbation_to_id)} unique perturbations")

    def _setup_splits(
        self, test_split_ratio: float, val_split_ratio: float
    ) -> None:
        """Setup train/val/test splits based on perturbations AND cell lines.

        For 4-fold cross-validation (e.g., fold 0 = K562 as test cell line):
        - Train: ALL perts from other cell lines (HepG2, Jurkat, RPE1)
                 + TRAIN perts from fold cell line (K562)
        - Test: TEST perts from fold cell line (K562) only

        This way test perturbations ARE seen during training (in other cell lines),
        so the learned perturbation embeddings are meaningful.
        """
        all_perts = list(self.perturbation_to_id.keys())

        # Normalize config cell lines to uppercase for consistent matching
        normalized_cell_lines = [cl.upper() for cl in self.config.cell_lines]

        # Try to load TOML-based splits
        self.test_perts = set()
        self.fold_cell_line = None

        if self.config.split_toml_dir:
            split_configs = load_split_configs(
                self.config.split_toml_dir, self.config.cell_lines
            )
            # Normalize split_configs keys to uppercase
            split_configs = {k.upper(): v for k, v in split_configs.items()}

            if self.config.fold is not None:
                # 4-fold CV: use only the specified fold's cell line for test
                if self.config.fold < 0 or self.config.fold >= len(normalized_cell_lines):
                    raise ValueError(
                        f"Fold {self.config.fold} out of range. "
                        f"Must be 0-{len(normalized_cell_lines)-1} for cell lines: "
                        f"{normalized_cell_lines}"
                    )

                self.fold_cell_line = normalized_cell_lines[self.config.fold]
                self.test_perts = split_configs.get(self.fold_cell_line, set())

                log.info(
                    f"4-fold CV: Fold {self.config.fold} - "
                    f"Test cell line: {self.fold_cell_line}, Test perts: {len(self.test_perts)}"
                )
                log.info(
                    f"Train = ALL perts in other cell lines + non-test perts in {self.fold_cell_line}"
                )
            else:
                # No fold specified: combine all test genes (original behavior)
                for cell_line, genes in split_configs.items():
                    self.test_perts.update(genes)
                log.info(
                    f"Combined mode: Using all TOML test genes ({len(self.test_perts)} total)"
                )

        # Build cell-line index first to know which perts exist in which cell lines
        self._build_cell_line_index()

        # Get perturbations that actually exist in fold_cell_line
        if self.fold_cell_line:
            perts_in_fold = set(
                pert for (pert, cl) in self.pert_cellline_to_indices.keys()
                if cl == self.fold_cell_line
            )
        else:
            perts_in_fold = set(all_perts)

        # Filter TOML test genes to only those that exist in fold cell line
        test_perts_in_fold = self.test_perts & perts_in_fold
        log.info(
            f"TOML test genes: {len(self.test_perts)}, "
            f"exist in {self.fold_cell_line}: {len(test_perts_in_fold)}"
        )

        # Split TOML test genes (that exist in fold) into val and test
        test_perts_list = list(test_perts_in_fold)
        self.rng.shuffle(test_perts_list)
        n_val = max(1, int(len(test_perts_list) * val_split_ratio))
        self.val_perts = set(test_perts_list[:n_val])
        self.test_perts = set(test_perts_list[n_val:])  # Remaining for test

        # Train perts in fold cell line = perts that exist minus (val + test)
        self.train_perts_in_fold = perts_in_fold - self.val_perts - self.test_perts

        log.info(
            f"Final split in {self.fold_cell_line}: "
            f"{len(self.train_perts_in_fold)} train, {len(self.val_perts)} val, {len(self.test_perts)} test"
        )

    def _build_cell_line_index(self) -> None:
        """Build index of cells per perturbation per cell line.

        Optimized to iterate once over cells instead of O(n_perts * n_cell_lines).
        """
        log.info("Building cell-line aware index...")
        self.pert_cellline_to_indices = {}

        # Single pass over all cells to build index
        for idx in range(len(self.perturbations)):
            pert = self.perturbations[idx]
            cell_line = self.cell_lines[idx]

            # Skip controls
            if pert == self.config.control_label:
                continue

            key = (pert, cell_line)
            if key not in self.pert_cellline_to_indices:
                self.pert_cellline_to_indices[key] = []
            self.pert_cellline_to_indices[key].append(idx)

        # Convert lists to numpy arrays
        for key in self.pert_cellline_to_indices:
            self.pert_cellline_to_indices[key] = np.array(
                self.pert_cellline_to_indices[key], dtype=np.int64
            )

        log.info(f"Built index with {len(self.pert_cellline_to_indices)} (pert, cell_line) combinations")

    def _generate_samples(self) -> None:
        """Generate sample indices for the current mode.

        Cell-line aware sampling for 4-fold CV:
        - train: ALL perts from OTHER cell lines + train_perts from fold cell line
        - val: val_perts from fold cell line
        - test: test_perts from fold cell line only
        """
        self.samples = []
        all_perts = set(self.perturbation_to_id.keys())
        # Cell lines are already normalized to uppercase in _load_data
        all_cell_lines = list(np.unique(self.cell_lines))
        # fold_cell_line is also uppercase from _setup_splits
        other_cell_lines = [cl for cl in all_cell_lines if cl != self.fold_cell_line]

        if self.mode == "test":
            # Test: test_perts in fold_cell_line only
            if self.fold_cell_line is None:
                # Combined mode: use all test perts from all cell lines
                for pert in self.test_perts:
                    self._add_samples_for_pert(pert, cell_lines=all_cell_lines)
            else:
                for pert in self.test_perts:
                    self._add_samples_for_pert(pert, cell_lines=[self.fold_cell_line])

        elif self.mode == "val":
            # Val: val_perts from fold cell line (same cell type as test, different perts)
            if self.fold_cell_line:
                for pert in self.val_perts:
                    self._add_samples_for_pert(pert, cell_lines=[self.fold_cell_line])
            else:
                for pert in self.val_perts:
                    self._add_samples_for_pert(pert, cell_lines=all_cell_lines)

        elif self.mode == "train":
            # Train: ALL perts from OTHER cell lines + train_perts from fold cell line
            if self.fold_cell_line:
                # All perts from other cell lines
                for pert in all_perts:
                    self._add_samples_for_pert(pert, cell_lines=other_cell_lines)
                # Train perts (non-test, non-val) from fold cell line
                for pert in self.train_perts_in_fold:
                    self._add_samples_for_pert(pert, cell_lines=[self.fold_cell_line])
            else:
                # No fold specified: use all perts from all cell lines
                for pert in all_perts:
                    self._add_samples_for_pert(pert, cell_lines=all_cell_lines)

        # Shuffle samples
        self.rng.shuffle(self.samples)

        log.info(f"Generated {len(self.samples)} samples for {self.mode} mode")

    def _add_samples_for_pert(
        self,
        pert: str,
        cell_lines: List[str],
    ) -> None:
        """Add samples for a perturbation from specified cell lines.

        If a perturbation doesn't have enough cells, resample with replacement
        to ensure every perturbation is represented.
        """
        for cell_line in cell_lines:
            key = (pert, cell_line)
            if key not in self.pert_cellline_to_indices:
                continue

            indices = self.pert_cellline_to_indices[key]
            n_available = len(indices)

            if n_available == 0:
                continue

            # Always create at least 1 sample per (pert, cell_line)
            # If not enough cells, resample with replacement
            if n_available >= self.config.n_cells:
                # Enough cells: create multiple non-overlapping samples
                n_samples = n_available // self.config.n_cells
                for i in range(n_samples):
                    start_idx = i * self.config.n_cells
                    end_idx = start_idx + self.config.n_cells
                    sample_indices = indices[start_idx:end_idx]
                    self.samples.append((pert, sample_indices))
            else:
                # Not enough cells: resample with replacement to fill n_cells
                sample_indices = self.rng.choice(
                    indices, size=self.config.n_cells, replace=True
                )
                self.samples.append((pert, sample_indices))

    def _load_expression(self, indices: np.ndarray) -> np.ndarray:
        """Load expression data for given cell indices.

        Handles duplicate indices (from resampling with replacement) by loading
        unique indices first, then expanding back.
        """
        # Get unique indices and mapping back to original positions
        unique_indices, inverse_mapping = np.unique(indices, return_inverse=True)

        # h5py requires indices in increasing order (unique already sorted)
        expr_data = self.adata.X[unique_indices, :]

        # Convert sparse to dense if needed
        if hasattr(expr_data, "toarray"):
            expr_data = expr_data.toarray()

        # Expand back to original size (handles duplicates)
        expr_data = expr_data[inverse_mapping, :]

        # Apply gene mapping if available
        if self.gene_mapping is not None and self.target_genes is not None:
            n_target_genes = len(self.target_genes)
            mapped_data = np.zeros((len(indices), n_target_genes), dtype=np.float32)

            target_indices = np.array(list(self.gene_mapping.keys()))
            source_indices = np.array(list(self.gene_mapping.values()))
            mapped_data[:, target_indices] = expr_data[:, source_indices]

            return mapped_data

        return expr_data.astype(np.float32)

    def _sample_control_cells(self, n_cells: int) -> np.ndarray:
        """Sample control cells for context."""
        if len(self.control_indices) < n_cells:
            # Upsample if not enough control cells
            indices = self.rng.choice(
                self.control_indices, size=n_cells, replace=True
            )
        else:
            indices = self.rng.choice(
                self.control_indices, size=n_cells, replace=False
            )
        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dictionary containing:
                - control_features: Control cell expression (n_cells, n_genes)
                - perturbed_features: Perturbed cell expression (n_cells, n_genes)
                - perturbation_id: Perturbation ID (scalar)
                - perturbation_name: Perturbation name (string, for debugging)
        """
        pert, pert_indices = self.samples[idx]

        # Load perturbed cell expression
        perturbed_expr = self._load_expression(pert_indices)

        # Sample and load control cells
        control_indices = self._sample_control_cells(self.config.n_cells)
        control_expr = self._load_expression(control_indices)

        # Get perturbation ID
        pert_id = self.perturbation_to_id[pert]

        return {
            "control_features": torch.from_numpy(control_expr).float(),
            "perturbed_features": torch.from_numpy(perturbed_expr).float(),
            "perturbation_id": torch.tensor(pert_id, dtype=torch.long),
            "perturbation_name": pert,
        }

    def get_num_perturbations(self) -> int:
        """Return total number of unique perturbations (for embedding table)."""
        return len(self.perturbation_to_id) + 1  # +1 for padding/unknown

    def get_perturbation_id(self, name: str) -> int:
        """Get perturbation ID from name."""
        return self.perturbation_to_id.get(name, 0)

    def get_perturbation_name(self, id: int) -> str:
        """Get perturbation name from ID."""
        return self.id_to_perturbation.get(id, "unknown")

    def get_n_genes(self) -> int:
        """Return number of genes."""
        if self.target_genes is not None:
            return len(self.target_genes)
        return self.n_genes


def create_perturbation_datasets(
    config: PerturbationDatasetConfig,
    random_state: int = 42,
    test_split_ratio: float = 0.2,
    val_split_ratio: float = 0.1,
) -> Tuple[PerturbationDataset, PerturbationDataset, PerturbationDataset]:
    """Create train, validation, and test perturbation datasets.

    Args:
        config: Dataset configuration.
        random_state: Random seed.
        test_split_ratio: Ratio for test split.
        val_split_ratio: Ratio for validation split.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = PerturbationDataset(
        config=config,
        mode="train",
        random_state=random_state,
        test_split_ratio=test_split_ratio,
        val_split_ratio=val_split_ratio,
    )

    # Share the splits with val and test datasets
    val_dataset = PerturbationDataset(
        config=config,
        mode="val",
        random_state=random_state,
        test_split_ratio=test_split_ratio,
        val_split_ratio=val_split_ratio,
    )

    test_dataset = PerturbationDataset(
        config=config,
        mode="test",
        random_state=random_state,
        test_split_ratio=test_split_ratio,
        val_split_ratio=val_split_ratio,
    )

    return train_dataset, val_dataset, test_dataset


def create_fold_datasets(
    config: PerturbationDatasetConfig,
    fold: int,
    random_state: int = 42,
    val_split_ratio: float = 0.1,
) -> Tuple[PerturbationDataset, PerturbationDataset, PerturbationDataset]:
    """Create train, validation, and test datasets for a specific CV fold.

    This is the recommended way to create datasets for 4-fold cross-validation.

    Args:
        config: Dataset configuration (fold will be overridden).
        fold: Fold index (0-3 for K562, HepG2, Jurkat, RPE1).
        random_state: Random seed.
        val_split_ratio: Ratio for validation split from training data.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Example:
        >>> config = PerturbationDatasetConfig(
        ...     data_path="/data/replogle_concat.h5ad",
        ...     split_toml_dir="/data/replogle_nogwps_v2/",
        ... )
        >>> # Run fold 0 (K562 as test)
        >>> train, val, test = create_fold_datasets(config, fold=0)
    """
    # Create a copy of config with the specified fold
    fold_config = PerturbationDatasetConfig(
        data_path=config.data_path,
        perturbation_col=config.perturbation_col,
        control_label=config.control_label,
        cell_line_col=config.cell_line_col,
        n_cells=config.n_cells,
        genelist_path=config.genelist_path,
        split_toml_dir=config.split_toml_dir,
        cell_lines=config.cell_lines,
        fold=fold,
    )

    train_dataset = PerturbationDataset(
        config=fold_config,
        mode="train",
        random_state=random_state,
        val_split_ratio=val_split_ratio,
    )

    val_dataset = PerturbationDataset(
        config=fold_config,
        mode="val",
        random_state=random_state,
        val_split_ratio=val_split_ratio,
    )

    test_dataset = PerturbationDataset(
        config=fold_config,
        mode="test",
        random_state=random_state,
        val_split_ratio=val_split_ratio,
    )

    return train_dataset, val_dataset, test_dataset


def get_fold_info(cell_lines: Optional[List[str]] = None) -> Dict[int, str]:
    """Get mapping of fold indices to cell line names.

    Args:
        cell_lines: List of cell lines. Defaults to ["K562", "HepG2", "Jurkat", "RPE1"].

    Returns:
        Dictionary mapping fold index to cell line name.

    Example:
        >>> get_fold_info()
        {0: 'K562', 1: 'HepG2', 2: 'Jurkat', 3: 'RPE1'}
    """
    if cell_lines is None:
        cell_lines = ["K562", "HepG2", "Jurkat", "RPE1"]
    return {i: cell_line for i, cell_line in enumerate(cell_lines)}
