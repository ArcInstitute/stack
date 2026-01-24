#!/usr/bin/env python3
"""Training script for the PerturbationExpert model with flow matching."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

import yaml


def _import_training_modules():
    """Lazy import heavy dependencies so --help stays fast."""
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy

    # Import directly from specific modules to avoid triggering scvi
    from stack.training.perturbation_lightning import (
        LightningPerturbationExpert,
        PerturbationDataModule,
    )
    from stack.training.utils import build_scheduler_config

    torch.set_float32_matmul_precision("high")

    return {
        "torch": torch,
        "pl": pl,
        "EarlyStopping": EarlyStopping,
        "LearningRateMonitor": LearningRateMonitor,
        "ModelCheckpoint": ModelCheckpoint,
        "TensorBoardLogger": TensorBoardLogger,
        "WandbLogger": WandbLogger,
        "DDPStrategy": DDPStrategy,
        "LightningPerturbationExpert": LightningPerturbationExpert,
        "PerturbationDataModule": PerturbationDataModule,
        "build_scheduler_config": build_scheduler_config,
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_args_with_config(
    args: argparse.Namespace, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge command-line arguments with config file, CLI takes precedence."""
    merged = dict(config)

    # Override with CLI args if provided
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value

    return merged


def configure_logger(args: Dict[str, Any], deps: Dict[str, Any]):
    """Configure training logger."""
    WandbLogger = deps["WandbLogger"]
    TensorBoardLogger = deps["TensorBoardLogger"]

    logger_type = args.get("logger", "tensorboard")
    save_dir = args.get("save_dir", "./checkpoints")
    project_name = args.get("project_name", "perturbation_expert")
    run_name = args.get("run_name", None)

    if logger_type == "wandb":
        return WandbLogger(
            project=project_name,
            name=run_name,
            save_dir=save_dir,
        )
    return TensorBoardLogger(
        save_dir=save_dir,
        name=project_name,
        version=run_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the PerturbationExpert model with flow matching"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Required arguments
    parser.add_argument(
        "--stack-checkpoint",
        type=str,
        default=None,
        help="Path to frozen Stack model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to h5ad file with perturbation data",
    )

    # Optional data arguments
    parser.add_argument(
        "--genelist-path",
        type=str,
        default=None,
        help="Path to gene list pickle file",
    )
    parser.add_argument(
        "--split-toml-dir",
        type=str,
        default=None,
        help="Directory containing TOML files for train/test splits",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for 4-fold CV (0=K562, 1=HepG2, 2=Jurkat, 3=RPE1). "
             "If not specified, combines all test genes from all TOMLs.",
    )
    parser.add_argument(
        "--perturbation-col",
        type=str,
        default="gene",
        help="Column name for perturbation labels",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default="non-targeting",
        help="Label for control cells",
    )
    parser.add_argument(
        "--cell-line-col",
        type=str,
        default="cell_line",
        help="Column name for cell line labels",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=128,
        help="Number of cells per sample",
    )

    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)

    # Inference arguments
    parser.add_argument("--num-sampling-steps", type=int, default=100)

    # Scheduler arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "cosine_restarts", "step", "reduce_on_plateau"],
        default="cosine",
    )
    parser.add_argument("--scheduler-T-max", type=int, default=100)
    parser.add_argument("--scheduler-warmup-epochs", type=int, default=5)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)

    # System arguments
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "ddp_find_unused_parameters_true", "auto"],
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)

    # Logging and checkpointing
    parser.add_argument("--project-name", type=str, default="perturbation_expert")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--logger",
        type=str,
        choices=["wandb", "tensorboard"],
        default="tensorboard",
    )
    parser.add_argument("--log-every-n-steps", type=int, default=10)

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    # Random seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Merge args with config
    settings = merge_args_with_config(args, config)

    # Validate required arguments
    if not settings.get("stack_checkpoint") and not settings.get("stack-checkpoint"):
        parser.error("--stack-checkpoint is required")
    if not settings.get("data_path") and not settings.get("data-path"):
        parser.error("--data-path is required")

    # Normalize key names (replace hyphens with underscores)
    settings = {k.replace("-", "_"): v for k, v in settings.items()}

    # Import dependencies
    deps = _import_training_modules()
    torch = deps["torch"]
    pl = deps["pl"]
    EarlyStopping = deps["EarlyStopping"]
    LearningRateMonitor = deps["LearningRateMonitor"]
    ModelCheckpoint = deps["ModelCheckpoint"]
    DDPStrategy = deps["DDPStrategy"]
    LightningPerturbationExpert = deps["LightningPerturbationExpert"]
    PerturbationDataModule = deps["PerturbationDataModule"]
    build_scheduler_config = deps["build_scheduler_config"]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set random seed
    pl.seed_everything(settings.get("seed", 42))

    # Create save directory
    save_dir = settings.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Setup data module
    fold = settings.get("fold")
    data_config = {
        "data_path": settings["data_path"],
        "perturbation_col": settings.get("perturbation_col", "gene"),
        "control_label": settings.get("control_label", "non-targeting"),
        "cell_line_col": settings.get("cell_line_col", "cell_line"),
        "n_cells": settings.get("n_cells", 128),
        "genelist_path": settings.get("genelist_path"),
        "split_toml_dir": settings.get("split_toml_dir"),
        "fold": fold,
    }

    # Log fold information
    cell_lines = ["K562", "HepG2", "Jurkat", "RPE1"]
    if fold is not None:
        logging.info(f"Running 4-fold CV: Fold {fold} ({cell_lines[fold]} as test set)")
    else:
        logging.info("Running with combined test set (all TOML test genes)")

    data_module = PerturbationDataModule(
        data_config=data_config,
        batch_size=settings.get("batch_size", 16),
        num_workers=settings.get("num_workers", 4),
        random_state=settings.get("seed", 42),
    )

    # Setup data to get number of perturbations
    data_module.setup("fit")
    num_perturbations = data_module.get_num_perturbations()

    logging.info(f"Number of perturbations: {num_perturbations}")

    # Build scheduler config
    scheduler_config = build_scheduler_config({
        "scheduler": settings.get("scheduler", "cosine"),
        "scheduler_T_max": settings.get("scheduler_T_max", 100),
        "scheduler_warmup_epochs": settings.get("scheduler_warmup_epochs", 5),
        "scheduler_eta_min": settings.get("scheduler_eta_min", 1e-6),
    })

    # Create model
    model = LightningPerturbationExpert(
        stack_checkpoint=settings["stack_checkpoint"],
        num_perturbations=num_perturbations,
        hidden_dim=settings.get("hidden_dim", 1024),
        cond_dim=settings.get("cond_dim", 256),
        n_layers=settings.get("n_layers", 6),
        n_heads=settings.get("n_heads", 8),
        dropout=settings.get("dropout", 0.1),
        learning_rate=settings.get("learning_rate", 1e-4),
        weight_decay=settings.get("weight_decay", 1e-4),
        warmup_epochs=settings.get("warmup_epochs", 5),
        num_sampling_steps=settings.get("num_sampling_steps", 100),
        scheduler_config=scheduler_config,
    )

    # Setup logger
    logger = configure_logger(settings, deps)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename="{epoch}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=settings.get("early_stopping_patience", 20),
            min_delta=settings.get("early_stopping_min_delta", 1e-4),
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Setup strategy
    strategy = settings.get("strategy", "ddp")
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)  # Need True for frozen encoder

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=settings.get("max_epochs", 100),
        devices=settings.get("gpus", -1),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        precision=settings.get("precision", "bf16-mixed"),
        gradient_clip_val=settings.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=settings.get("accumulate_grad_batches", 1),
        log_every_n_steps=settings.get("log_every_n_steps", 10),
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        benchmark=True,
    )

    # Log configuration
    logging.info("=" * 80)
    logging.info("TRAINING CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Stack checkpoint: {settings['stack_checkpoint']}")
    logging.info(f"Data path: {settings['data_path']}")
    logging.info(f"Number of perturbations: {num_perturbations}")
    logging.info(f"Hidden dim: {settings.get('hidden_dim', 1024)}")
    logging.info(f"Conditioning dim: {settings.get('cond_dim', 256)}")
    logging.info(f"Number of layers: {settings.get('n_layers', 6)}")
    logging.info(f"Number of heads: {settings.get('n_heads', 8)}")
    logging.info(f"Batch size: {settings.get('batch_size', 16)}")
    logging.info(f"Learning rate: {settings.get('learning_rate', 1e-4)}")
    logging.info(f"Max epochs: {settings.get('max_epochs', 100)}")
    logging.info(f"Sampling steps: {settings.get('num_sampling_steps', 100)}")
    if fold is not None:
        logging.info(f"CV Fold: {fold} (test cell line: {cell_lines[fold]})")
    else:
        logging.info("CV Fold: None (combined mode)")
    logging.info(f"Devices: {trainer.num_devices}")
    logging.info("=" * 80)

    # Train
    trainer.fit(model, data_module)

    # Test
    if data_module.test_dataset is not None and len(data_module.test_dataset) > 0:
        logging.info("Running test evaluation...")
        test_results = trainer.test(model, data_module, ckpt_path="best")
        logging.info(f"Test results: {test_results}")

    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
