"""Lightning module definitions for the fine-tuning pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch

from ..model_finetune import ICL_FinetunedModel

log = logging.getLogger(__name__)


class LightningFinetunedModel(pl.LightningModule):
    """PyTorch Lightning module that trains a student model against a frozen teacher."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None,
        n_kept_cell: int = 96,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ICL_FinetunedModel(**model_config)
        self.teacher_model = ICL_FinetunedModel(**model_config)

        log.info("Freezing teacher model parameters for distillation")
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.n_kept_cell = n_kept_cell
        self.teacher_ema_decay = 0.95
        self.ema_every_n_steps = 500

        self.train_metrics = []
        self.val_metrics = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:  # type: ignore[override]
        log.info("`on_fit_start` hook called. Syncing teacher model weights from student...")
        self.teacher_model.load_state_dict(self.model.state_dict())

        log.info("VERIFYING teacher weights against student weights...")
        student_param = self.model.layers[0].cell_attn.qkv.weight
        teacher_param = self.teacher_model.layers[0].cell_attn.qkv.weight

        if torch.equal(student_param, teacher_param):
            log.info("SUCCESS: Teacher weights match student weights.")
            log.info("Sample tensor sum: %s", student_param.sum().item())
        else:
            diff = torch.sum(torch.abs(student_param - teacher_param))
            log.error("FAILURE: Teacher and student weights diverged; diff=%s", diff.item())
            raise RuntimeError("Teacher model failed to sync with student model.")

        log.info("Teacher model weights synced successfully.")
        try:
            model_config = self.hparams.model_config
            n_kept_cell = self.hparams.n_kept_cell

            dummy_obs = torch.rand(1, model_config['n_cells'], model_config['n_genes'], device=self.device)
            dummy_gt = torch.rand(1, model_config['n_cells'], model_config['n_genes'], device=self.device)
            dummy_ct = torch.zeros(1, model_config['n_cells'], dtype=torch.long, device=self.device)
            dummy_mask = torch.ones(1, model_config['n_cells'], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                _ = self.model(
                    observed_features=dummy_obs,
                    ground_truth_features=dummy_gt,
                    cell_type_ids=dummy_ct,
                    position_mask=dummy_mask,
                    n_kept_cell=n_kept_cell,
                    return_loss=False,
                )
                _ = self.teacher_model(
                    observed_features=dummy_gt,
                    ground_truth_features=dummy_gt,
                    cell_type_ids=dummy_ct,
                    position_mask=dummy_mask,
                    mask_genes=False,
                    n_kept_cell=model_config['n_cells'],
                    return_loss=False,
                )
            log.info("Model architecture verification passed for both student and teacher.")
        except Exception as exc:  # pragma: no cover - diagnostic logging path
            log.error("Model architecture mismatch detected during verification: %s", exc)
            raise

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {k: v for k, v in state_dict.items() if k.startswith("model.")}
        log.info("Checkpoint pruned to student parameters only (kept 'model.' prefix).")

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------
    def _forward_pass_with_teacher(self, batch: Any) -> Dict[str, Any]:
        ground_truth_features, observed_features, cell_type_ids, position_mask, _ = batch

        with torch.no_grad():
            teacher_output = self.teacher_model(
                observed_features=ground_truth_features,
                ground_truth_features=ground_truth_features,
                cell_type_ids=cell_type_ids,
                position_mask=position_mask,
                mask_genes=False,
                n_kept_cell=self.hparams.model_config['n_cells'],
                return_loss=False,
            )
            target_embeddings = teacher_output['final_cell_embeddings'].detach()

        result = self.model(
            observed_features=observed_features,
            ground_truth_features=ground_truth_features,
            cell_type_ids=cell_type_ids,
            position_mask=position_mask,
            t_cell_embeddings=target_embeddings,
            n_kept_cell=self.n_kept_cell,
            return_loss=True,
        )
        return result

    def forward(self, *inputs: Any, **kwargs: Any):  # type: ignore[override]
        return self.model(*inputs, **kwargs)

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        result = self._forward_pass_with_teacher(batch)

        total_loss = result['loss']
        recon_loss = result['recon_loss']
        mmd_loss = result['mmd_loss']
        sw_loss = result['sw_loss']
        cls_loss = result['cls_loss']
        cls_acc = result['cls_acc']

        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/mmd_loss', mmd_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/sw_loss', sw_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/cls_loss', cls_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/cls_acc', cls_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        result = self._forward_pass_with_teacher(batch)

        total_loss = result['loss']
        recon_loss = result['recon_loss']
        mmd_loss = result['mmd_loss']
        sw_loss = result['sw_loss']
        cls_loss = result['cls_loss']
        cls_acc = result['cls_acc']

        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', recon_loss, on_epoch=True, sync_dist=True)
        self.log('val/mmd_loss', mmd_loss, on_epoch=True, sync_dist=True)
        self.log('val/sw_loss', sw_loss, on_epoch=True, sync_dist=True)
        self.log('val/cls_loss', cls_loss, on_epoch=True, sync_dist=True)
        self.log('val/cls_acc', cls_acc, on_epoch=True, sync_dist=True)

        if 'masked_mae' in result:
            self.log('val/masked_mae', result['masked_mae'], on_epoch=True, sync_dist=True)
        if 'masked_corr' in result:
            self.log('val/masked_corr', result['masked_corr'], on_epoch=True, sync_dist=True)
        if 'sw_predict' in result:
            self.log('val/sw_predict', result['sw_predict'], on_epoch=True, sync_dist=True)
        if 'mask_rate' in result:
            self.log('val/mask_rate', result['mask_rate'], on_epoch=True, sync_dist=True)

        return {'val_loss': total_loss}

    def test_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        result = self._forward_pass_with_teacher(batch)

        total_loss = result['loss']
        recon_loss = result['recon_loss']
        mmd_loss = result['mmd_loss']
        sw_loss = result['sw_loss']
        cls_loss = result['cls_loss']
        cls_acc = result['cls_acc']

        self.log('test_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test/recon_loss', recon_loss, on_epoch=True, sync_dist=True)
        self.log('test/mmd_loss', mmd_loss, on_epoch=True, sync_dist=True)
        self.log('test/sw_loss', sw_loss, on_epoch=True, sync_dist=True)
        self.log('test/cls_loss', cls_loss, on_epoch=True, sync_dist=True)
        self.log('test/cls_acc', cls_acc, on_epoch=True, sync_dist=True)

        if 'masked_mae' in result:
            self.log('test/masked_mae', result['masked_mae'], on_epoch=True, sync_dist=True)
        if 'masked_corr' in result:
            self.log('test/masked_corr', result['masked_corr'], on_epoch=True, sync_dist=True)
        if 'sw_predict' in result:
            self.log('test/sw_predict', result['sw_predict'], on_epoch=True, sync_dist=True)
        if 'mask_rate' in result:
            self.log('test/mask_rate', result['mask_rate'], on_epoch=True, sync_dist=True)

        return {'test_loss': total_loss}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        config: Dict[str, Any] = {'optimizer': optimizer}

        if not self.scheduler_config:
            return config

        scheduler_type = self.scheduler_config.get('type', 'cosine')
        if scheduler_type == 'cosine':
            warmup_epochs = self.scheduler_config.get('warmup_epochs', 0)
            T_max = self.scheduler_config.get('T_max', 100)
            eta_min = self.scheduler_config.get('eta_min', 1e-6)

            if warmup_epochs > 0:
                from torch.optim.lr_scheduler import LinearLR, SequentialLR

                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, T_max - warmup_epochs),
                    eta_min=eta_min,
                )
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            config['lr_scheduler'] = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
            return config

        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 10),
                verbose=True,
            )
            config['lr_scheduler'] = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
            return config

        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def on_train_epoch_start(self) -> None:  # type: ignore[override]
        datamodule = self.trainer.datamodule
        if getattr(datamodule, "resample_each_epoch", False):
            datamodule.train_dataset.resample_training_data()
            if self.global_rank == 0:
                log.info(
                    "[Epoch %s] Resampled training data -> %s samples",
                    self.current_epoch,
                    len(datamodule.train_dataset),
                )
                
    @torch.no_grad()
    def _ema_update_teacher(self):
        ema = float(self.teacher_ema_decay)

        for t, s in zip(self.teacher_model.parameters(), self.model.parameters()):
            t.data.mul_(ema).add_(s.data, alpha=1.0 - ema)

        for t_buf, s_buf in zip(self.teacher_model.buffers(), self.model.buffers()):
            t_buf.copy_(s_buf)

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer):
        self.teacher_model.eval()
        step = int(self.global_step)
        if step > 0 and (step % int(self.ema_every_n_steps) == 0):
            self._ema_update_teacher()
            if getattr(self, "global_rank", 0) == 0:
                logging.info(
                    f"EMA teacher updated @ global_step={step}, ema={self.teacher_ema_decay}"
                )


# Backwards compatibility -------------------------------------------------------------------------
class LegacyLightningFinetunedModel(LightningFinetunedModel):
    """Alias retained for legacy import paths."""

    pass
