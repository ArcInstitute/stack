"""Minimal smoke tests for the CLI entry points to ensure they can be invoked."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

import pytest


def _make_dummy_training_deps() -> Dict[str, Any]:
    class DummyTorch:
        @staticmethod
        def set_float32_matmul_precision(_arg: str) -> None:
            return None

        class cuda:
            @staticmethod
            def is_available() -> bool:  # pragma: no cover - trivial stub
                return False

    class DummyTrainer:
        def __init__(self, *_, **__):
            self.num_devices = 0

        def fit(self, *_, **__):
            return None

        def test(self, *_, **__):
            return []

    class DummyCallback:
        def __init__(self, *_, **__):
            return None

    class DummyLogger:
        def __init__(self, *_, **__):
            return None

    class DummyDDPStrategy:
        def __init__(self, *_, **__):
            return None

    def build_scheduler_config(_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def localize_datasets(dataset_configs, *_args, **_kwargs):
        return SimpleNamespace(
            dataset_configs=dataset_configs,
            cache_file=None,
            job_id="test-job",
            stage_out=lambda *_a, **_kw: None,
        )

    def parse_dataset_configs(dataset_configs):
        configs = dataset_configs or []
        if not isinstance(configs, (list, tuple)):
            configs = [configs]
        return [
            SimpleNamespace(
                path=str(cfg),
                filter_organism=None,
                gene_name_col=None,
                type="human",
                donor_col="donor",
                cell_type_col="celltype",
                condition_col="condition",
                cell_line_col="cellline",
                control_condition="control",
            )
            for cfg in configs
        ]

    class DummyDataModule:
        def __init__(self, *_, **__):
            self.test_dataset = []
            self.dataset_configs = parse_dataset_configs(["dummy"])

        def setup(self, *_args, **_kwargs):
            self.n_genes = 4
            return None

        def get_split_info(self):
            return {"train": [], "val": [], "test": []}

    class DummyModel:
        def __init__(self, *_, **__):
            return None

        @classmethod
        def load_from_checkpoint(cls, *_, **__):
            return cls()

    def build_model_config(args, n_genes):
        return {"n_genes": n_genes}

    return {
        "torch": DummyTorch,
        "pl": SimpleNamespace(Trainer=DummyTrainer),
        "EarlyStopping": DummyCallback,
        "LearningRateMonitor": DummyCallback,
        "ModelCheckpoint": DummyCallback,
        "TensorBoardLogger": DummyLogger,
        "WandbLogger": DummyLogger,
        "Logger": DummyLogger,
        "DDPStrategy": DummyDDPStrategy,
        "MultiDatasetDataModule": DummyDataModule,
        "FinetuneDataModule": DummyDataModule,
        "LightningFinetunedModel": DummyModel,
        "LegacyLightningGeneModel": DummyModel,
        "build_scheduler_config": build_scheduler_config,
        "localize_datasets": localize_datasets,
        "parse_dataset_configs": parse_dataset_configs,
        "configure_logger": DummyLogger,
        "build_model_config": build_model_config,
        "override_model_config_n_cells": lambda *_a, **_kw: {},
    }


def test_stack_train_main_runs(monkeypatch, tmp_path):
    from stack.cli import launch_training
    import sys

    dummy_deps = _make_dummy_training_deps()
    monkeypatch.setattr(launch_training, "_import_training_modules", lambda: dummy_deps)
    for name in ("TensorBoardLogger", "WandbLogger", "EarlyStopping", "LearningRateMonitor", "ModelCheckpoint"):
        monkeypatch.setattr(launch_training, name, dummy_deps[name], raising=False)
    monkeypatch.setattr(launch_training, "configure_logger", lambda *_a, **_kw: dummy_deps["TensorBoardLogger"]())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stack-train",
            "--dataset_configs",
            "dummy-dataset",
            "--genelist_path",
            "dummy-genelist.pkl",
            "--save_dir",
            str(tmp_path),
            "--gpus",
            "0",
        ],
    )
    launch_training.main()
    assert (tmp_path / "dataset_splits.json").exists()


def test_stack_finetune_main_runs(monkeypatch, tmp_path):
    from stack.cli import launch_finetuning
    import sys

    dummy_deps = _make_dummy_training_deps()
    monkeypatch.setattr(launch_finetuning, "_import_training_modules", lambda: dummy_deps)
    for name in ("TensorBoardLogger", "WandbLogger", "EarlyStopping", "LearningRateMonitor", "ModelCheckpoint", "Logger"):
        monkeypatch.setattr(launch_finetuning, name, dummy_deps.get(name, dummy_deps["TensorBoardLogger"]), raising=False)
    monkeypatch.setattr(
        launch_finetuning,
        "configure_logger",
        lambda *_a, **_kw: dummy_deps["TensorBoardLogger"](),
        raising=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stack-finetune",
            "--dataset_configs",
            "dummy-dataset",
            "--genelist_path",
            "dummy-genelist.pkl",
            "--save_dir",
            str(tmp_path),
            "--gpus",
            "0",
        ],
    )
    launch_finetuning.main()
    assert (tmp_path / "dataset_splits.json").exists()


def test_stack_embedding_main_runs(monkeypatch, tmp_path):
    from stack.cli import embedding

    monkeypatch.setattr(
        embedding,
        "extract_embeddings",
        lambda **_kw: ([[]], None),
    )
    saved = {}
    monkeypatch.setattr(
        embedding,
        "save_embeddings",
        lambda embeddings, output_path, **_: saved.update({"path": output_path, "embeddings": embeddings}),
    )

    args = [
        "--checkpoint",
        "dummy.ckpt",
        "--adata",
        "dummy.h5ad",
        "--genelist",
        "dummy-genelist.pkl",
        "--output",
        str(tmp_path / "embeddings.npy"),
    ]
    embedding.main(args)
    assert saved["path"].name == "embeddings.npy"


def test_stack_generation_main_runs(monkeypatch, tmp_path):
    from stack.cli import generation

    monkeypatch.setattr(
        generation,
        "generate",
        lambda **_kw: {"split": "dummy"},
    )
    called = {}
    monkeypatch.setattr(
        generation,
        "save_generations",
        lambda generations, output_dir, **_: called.update({"generations": generations, "output_dir": output_dir}),
    )

    args = [
        "--checkpoint",
        "dummy.ckpt",
        "--base-adata",
        "base.h5ad",
        "--test-adata",
        "test.h5ad",
        "--genelist",
        "genelist.pkl",
        "--output-dir",
        str(tmp_path),
        "--split-column",
        "donor",
        "--concatenate",
    ]
    generation.main(args)
    assert called["output_dir"] == tmp_path
    assert called["generations"] == {"split": "dummy"}
