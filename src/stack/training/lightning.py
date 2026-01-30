"""Lightning module that wraps the :class:`StateICLModel`."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

import pytorch_lightning as pl
import torch

from ..model import StateICLModel, scShiftAttentionModel
from .utils import configure_scheduler


class LightningGeneModel(pl.LightningModule):
    """Thin Lightning wrapper that delegates heavy lifting to :class:`StateICLModel`."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None,
        model_class: Type[StateICLModel] = StateICLModel,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_class"])
        self.model = model_class(**model_config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}

    def forward(self, features: torch.Tensor, return_loss: bool = True):  # type: ignore[override]
        return self.model(features, return_loss=return_loss)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        features, _ = batch
        result = self.model(features, return_loss=True)
        total_loss = result["loss"]
        self.log("tloss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("recon", result["recon_loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("sw", result["sw_loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/masked_mae", result["masked_mae"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/masked_corr", result["masked_corr"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/mask_rate", result["mask_rate"], on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        features, _ = batch
        result = self.model(features, return_loss=True)
        total_loss = result["loss"]
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", result["recon_loss"], on_epoch=True, sync_dist=True)
        self.log("val/sw_loss", result["sw_loss"], on_epoch=True, sync_dist=True)
        self.log("val/masked_mae", result["masked_mae"], on_epoch=True, sync_dist=True)
        self.log("val/masked_corr", result["masked_corr"], on_epoch=True, sync_dist=True)
        self.log("val/mask_rate", result["mask_rate"], on_epoch=True, sync_dist=True)
        return {"val_loss": total_loss}

    def test_step(self, batch, batch_idx):  # type: ignore[override]
        features, _ = batch
        result = self.model(features, return_loss=True)
        total_loss = result["loss"]
        self.log("test_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/recon_loss", result["recon_loss"], on_epoch=True, sync_dist=True)
        self.log("test/sw_loss", result["sw_loss"], on_epoch=True, sync_dist=True)
        self.log("test/masked_mae", result["masked_mae"], on_epoch=True, sync_dist=True)
        self.log("test/masked_corr", result["masked_corr"], on_epoch=True, sync_dist=True)
        self.log("test/mask_rate", result["mask_rate"], on_epoch=True, sync_dist=True)
        return {"test_loss": total_loss}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler_dict = configure_scheduler(optimizer, self.scheduler_config)
        if scheduler_dict:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return {"optimizer": optimizer}


# Backwards compatibility -------------------------------------------------------------------------
class LegacyLightningGeneModel(LightningGeneModel):
    """Variant that instantiates :class:`scShiftAttentionModel` for backwards compatibility."""

    def __init__(self, model_config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(model_config=model_config, model_class=scShiftAttentionModel, **kwargs)
