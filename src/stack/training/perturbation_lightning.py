"""Lightning module for training the PerturbationExpert model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..models.perturbation.expert import PerturbationExpert
from .utils import configure_scheduler


class LightningPerturbationExpert(pl.LightningModule):
    """Lightning wrapper for PerturbationExpert with flow matching training.

    This module handles:
    - Flow matching training with MSE loss on predicted velocities
    - Validation with sampling metrics (correlation, MSE)
    - Proper handling of frozen Stack encoder vs. trainable parameters

    Args:
        stack_checkpoint: Path to frozen Stack model checkpoint.
        num_perturbations: Number of unique perturbations.
        hidden_dim: Hidden dimension of velocity network.
        cond_dim: Dimension of conditioning vectors.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        warmup_epochs: Number of warmup epochs.
        num_sampling_steps: Number of Euler steps for sampling.
        scheduler_config: Optional scheduler configuration.
    """

    def __init__(
        self,
        stack_checkpoint: str,
        num_perturbations: int,
        hidden_dim: int = 1024,
        cond_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        num_sampling_steps: int = 100,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Create the model
        self.model = PerturbationExpert(
            stack_checkpoint=stack_checkpoint,
            num_perturbations=num_perturbations,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.num_sampling_steps = num_sampling_steps
        self.scheduler_config = scheduler_config or {}

    def forward(
        self,
        control_features: torch.Tensor,
        perturbed_features: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        return self.model(
            control_features=control_features,
            perturbed_features=perturbed_features,
            perturbation_ids=perturbation_ids,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with flow matching loss."""
        control = batch["control_features"]
        perturbed = batch["perturbed_features"]
        pert_ids = batch["perturbation_id"]

        result = self.model(
            control_features=control,
            perturbed_features=perturbed,
            perturbation_ids=pert_ids,
        )

        loss = result["loss"]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/t_mean", result["t"].mean(), on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with flow matching loss and sampling metrics."""
        control = batch["control_features"]
        perturbed = batch["perturbed_features"]
        pert_ids = batch["perturbation_id"]

        # Compute flow matching loss
        result = self.model(
            control_features=control,
            perturbed_features=perturbed,
            perturbation_ids=pert_ids,
            return_velocity=True,
        )

        loss = result["loss"]
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Sample and compute metrics (only on first batch to save time)
        if batch_idx == 0:
            sampled = self.model.sample(
                control_features=control,
                perturbation_ids=pert_ids,
                num_steps=self.num_sampling_steps,
            )

            # Compute correlation between sampled and actual perturbed
            # Note: perturbed is ALREADY log1p normalized from dataset
            pred_log1p = sampled["perturbed_log1p"]
            target_log1p = perturbed  # Already log1p normalized

            # Per-gene correlation (averaged across batch and cells)
            gene_corr = self._compute_gene_correlation(pred_log1p, target_log1p)
            self.log("val/gene_corr", gene_corr, on_epoch=True, prog_bar=True, sync_dist=True)

            # MSE in log1p space
            mse = F.mse_loss(pred_log1p, target_log1p)
            self.log("val/mse_log1p", mse, on_epoch=True, sync_dist=True)

        return {"val_loss": loss}

    def _compute_gene_correlation(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute average Pearson correlation across genes.

        Args:
            pred: Predictions of shape (B, n_cells, n_genes).
            target: Targets of shape (B, n_cells, n_genes).

        Returns:
            Average correlation across all genes.
        """
        # Flatten batch and cells: (B * n_cells, n_genes)
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

        # Compute per-gene correlation
        pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=0, keepdim=True)

        pred_std = pred_centered.std(dim=0) + 1e-8
        target_std = target_centered.std(dim=0) + 1e-8

        corr = (pred_centered * target_centered).mean(dim=0) / (pred_std * target_std)

        # Return mean correlation (ignoring NaNs)
        valid_corr = corr[~torch.isnan(corr)]
        if len(valid_corr) > 0:
            return valid_corr.mean()
        return torch.tensor(0.0, device=pred.device)

    def configure_optimizers(self):
        """Configure optimizer with only trainable parameters."""
        # Only optimize trainable parameters (velocity network + conditioning)
        trainable_params = self.model.get_trainable_parameters()

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler_dict = configure_scheduler(optimizer, self.scheduler_config)
        if scheduler_dict:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return {"optimizer": optimizer}

    @torch.no_grad()
    def predict_perturbation(
        self,
        control_features: torch.Tensor,
        perturbation_ids: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate perturbed expression predictions.

        Args:
            control_features: Control cell expression (B, n_cells, n_genes).
            perturbation_ids: Perturbation IDs (B,).
            num_steps: Number of sampling steps (defaults to self.num_sampling_steps).

        Returns:
            Dictionary with predicted perturbed expression.
        """
        self.model.eval()
        num_steps = num_steps or self.num_sampling_steps
        return self.model.sample(
            control_features=control_features,
            perturbation_ids=perturbation_ids,
            num_steps=num_steps,
        )


class PerturbationDataModule(pl.LightningDataModule):
    """Lightning DataModule for perturbation datasets.

    Args:
        data_config: Configuration for the perturbation dataset.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        random_state: Random seed.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 4,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        from ..data.perturbation import PerturbationDataset, PerturbationDatasetConfig

        config = PerturbationDatasetConfig(**self.data_config)

        if stage == "fit" or stage is None:
            self.train_dataset = PerturbationDataset(
                config=config,
                mode="train",
                random_state=self.random_state,
            )
            self.val_dataset = PerturbationDataset(
                config=config,
                mode="val",
                random_state=self.random_state,
            )

        if stage == "test" or stage is None:
            self.test_dataset = PerturbationDataset(
                config=config,
                mode="test",
                random_state=self.random_state,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_num_perturbations(self) -> int:
        """Get total number of perturbations."""
        if self.train_dataset is not None:
            return self.train_dataset.get_num_perturbations()
        # Create temporary dataset to get count
        from ..data.perturbation import PerturbationDataset, PerturbationDatasetConfig

        config = PerturbationDatasetConfig(**self.data_config)
        temp_dataset = PerturbationDataset(
            config=config,
            mode="train",
            random_state=self.random_state,
        )
        return temp_dataset.get_num_perturbations()


__all__ = [
    "LightningPerturbationExpert",
    "PerturbationDataModule",
]
