#!/usr/bin/env python3
"""Entry point script for training the StateICL model."""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict

try:  # pragma: no cover - runtime import resolution
    from stack.cli_utils import apply_config_from_file, filter_unused_arguments
except ImportError:  # pragma: no cover - script execution fallback
    from cli_utils import apply_config_from_file, filter_unused_arguments  # type: ignore


def _import_training_modules():
    """Lazy import heavy dependencies so ``--help`` stays fast."""

    try:
        import torch  # type: ignore
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
        from pytorch_lightning.strategies import DDPStrategy

        from stack.training.datamodule import MultiDatasetDataModule
        from stack.training.lightning import LegacyLightningGeneModel
        from stack.training.utils import build_scheduler_config, localize_datasets, parse_dataset_configs
    except ImportError:  # pragma: no cover - script execution fallback
        import torch  # type: ignore
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # type: ignore
        from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # type: ignore
        from pytorch_lightning.strategies import DDPStrategy  # type: ignore

        from training.datamodule import MultiDatasetDataModule  # type: ignore
        from training.lightning import LegacyLightningGeneModel  # type: ignore
        from training.utils import build_scheduler_config, localize_datasets, parse_dataset_configs  # type: ignore

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
        "MultiDatasetDataModule": MultiDatasetDataModule,
        "LegacyLightningGeneModel": LegacyLightningGeneModel,
        "build_scheduler_config": build_scheduler_config,
        "localize_datasets": localize_datasets,
        "parse_dataset_configs": parse_dataset_configs,
    }


def build_model_config(args: argparse.Namespace, n_genes: int) -> Dict[str, int | float]:
    return {
        "n_genes": n_genes,
        "n_hidden": args.n_hidden,
        "token_dim": args.token_dim,
        "n_cells": args.sample_size,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "mask_rate_min": args.mask_rate_min,
        "mask_rate_max": args.mask_rate_max,
        "sw_weight": args.sw_weight,
        "n_proj": args.n_proj,
    }


def configure_logger(args: argparse.Namespace):
    if args.logger == "wandb":
        return WandbLogger(project=args.project_name, name=args.run_name, save_dir=args.save_dir)
    return TensorBoardLogger(save_dir=args.save_dir, name=args.project_name, version=args.run_name)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to a YAML/JSON config file")

    parser = argparse.ArgumentParser(description="Train the StateICL model", parents=[config_parser])

    # Data arguments
    parser.add_argument("--dataset_configs", type=str, nargs="+", default=None, help="path[:filter[:gene_column]]")
    parser.add_argument("--genelist_path", type=str, default=None, help="Path to saved gene list file (.pkl)")
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cache_file", type=str, default=None)

    # Model arguments
    parser.add_argument("--n_hidden", type=int, default=100)
    parser.add_argument("--token_dim", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_rate_min", type=float, default=0.1)
    parser.add_argument("--mask_rate_max", type=float, default=0.8)
    parser.add_argument("--sw_weight", type=float, default=0.01)
    parser.add_argument("--n_proj", type=int, default=64)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)

    # Scheduler arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "cosine_restarts", "step", "reduce_on_plateau"],
        default="cosine",
    )
    parser.add_argument("--scheduler_T_max", type=int, default=20)
    parser.add_argument("--scheduler_warmup_epochs", type=int, default=1)
    parser.add_argument("--scheduler_T_0", type=int, default=10)
    parser.add_argument("--scheduler_T_mult", type=int, default=2)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)

    # System arguments
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "ddp_find_unused_parameters_false"])
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Logging and checkpointing
    parser.add_argument("--project_name", type=str, default="scShiftAttention")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="../checkpoints")
    parser.add_argument("--logger", type=str, choices=["wandb", "tensorboard"], default="tensorboard")
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)

    # Data localization
    parser.add_argument("--local_temp_dir", type=str, default="/tmp")
    parser.add_argument("--no_localize_data", action="store_true")

    config_args, remaining_argv = config_parser.parse_known_args()
    apply_config_from_file(parser, config_args.config)

    args = parser.parse_args(remaining_argv)
    filter_unused_arguments(args, ("dataset_configs", "genelist_path"), parser)

    deps = _import_training_modules()
    torch = deps["torch"]
    pl = deps["pl"]
    EarlyStopping = deps["EarlyStopping"]
    LearningRateMonitor = deps["LearningRateMonitor"]
    ModelCheckpoint = deps["ModelCheckpoint"]
    TensorBoardLogger = deps["TensorBoardLogger"]
    WandbLogger = deps["WandbLogger"]
    DDPStrategy = deps["DDPStrategy"]
    MultiDatasetDataModule = deps["MultiDatasetDataModule"]
    LegacyLightningGeneModel = deps["LegacyLightningGeneModel"]
    build_scheduler_config = deps["build_scheduler_config"]
    localize_datasets = deps["localize_datasets"]
    parse_dataset_configs = deps["parse_dataset_configs"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if config_args.config:
        logging.info("Loaded training defaults from %s", config_args.config)

    localization = localize_datasets(args.dataset_configs, args.cache_file, args.local_temp_dir, args.no_localize_data)

    try:
        dataset_configs = parse_dataset_configs(localization.dataset_configs)
        scheduler_cfg = build_scheduler_config(vars(args))

        data_module = MultiDatasetDataModule(
            dataset_configs=dataset_configs,
            genelist_path=args.genelist_path,
            sample_size=args.sample_size,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.random_seed,
            cache_file=localization.cache_file,
        )

        # Setup data module to get dataset info
        data_module.setup()

        # Create save directory if it doesn't exist
        os.makedirs(args.save_dir, exist_ok=True)

        # Save dataset split information
        split_info = data_module.get_split_info()
        split_info_path = os.path.join(args.save_dir, "dataset_splits.json")
        try:
            with open(split_info_path, "w") as f:
                json.dump(split_info, f, indent=4)
            logging.info("Saved dataset split information to %s", split_info_path)
        except Exception as exc:
            logging.error("Failed to save dataset split information: %s", exc)

        model_config = build_model_config(args, data_module.n_genes)
        model = LegacyLightningGeneModel(
            model_config=model_config,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_config=scheduler_cfg,
        )

        logger = configure_logger(args)
        callbacks = [
            ModelCheckpoint(
                dirpath=args.save_dir,
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        strategy = DDPStrategy(find_unused_parameters=False) if args.strategy == "ddp" else args.strategy
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            devices=args.gpus,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy=strategy,
            precision=args.precision,
            gradient_clip_val=1.0,
            accumulate_grad_batches=args.accumulate_grad_batches,
            log_every_n_steps=args.log_every_n_steps,
            logger=logger,
            callbacks=callbacks,
            deterministic=False,
            benchmark=True,
        )

        logging.info("=" * 80)
        logging.info("TRAINING CONFIGURATION")
        logging.info("=" * 80)
        for key, value in model_config.items():
            logging.info("  %s: %s", key, value)
        for idx, cfg in enumerate(dataset_configs):
            logging.info("Dataset %s: %s (filter=%s gene_col=%s)", idx + 1, cfg.path, cfg.filter_organism, cfg.gene_name_col)
        logging.info("Gene list: %s (%s genes)", args.genelist_path, data_module.n_genes)
        logging.info("Sample size: %s", args.sample_size)
        logging.info("Batch size: %s", args.batch_size)
        logging.info("Learning rate: %s", args.learning_rate)
        logging.info("Max epochs: %s", args.max_epochs)
        logging.info("Devices: %s", trainer.num_devices)
        logging.info("Strategy: %s", strategy)
        logging.info("Precision: %s", args.precision)
        logging.info("=" * 80)

        trainer.fit(model, data_module)

        if data_module.test_dataset is not None and len(data_module.test_dataset) > 0:
            logging.info("Running test evaluation...")
            test_results = trainer.test(model, data_module, ckpt_path="best")
            logging.info("Test results: %s", test_results)

        logging.info("Training pipeline completed successfully!")

    finally:
        localization.stage_out(args.save_dir, localization.job_id or str(os.getpid()))


if __name__ == "__main__":
    main()
