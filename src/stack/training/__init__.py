"""Training utilities for StateICL."""

# Lazy imports to avoid triggering scvi at module load time
# Heavy modules are imported on-demand when accessed

from .utils import (
    LocalizationContext,
    build_scheduler_config,
    configure_scheduler,
    localize_datasets,
    parse_dataset_configs,
)

__all__ = [
    "MultiDatasetDataModule",
    "LightningGeneModel",
    "LegacyLightningGeneModel",
    "LightningPerturbationExpert",
    "PerturbationDataModule",
    "LocalizationContext",
    "build_scheduler_config",
    "configure_scheduler",
    "localize_datasets",
    "parse_dataset_configs",
]


def __getattr__(name):
    """Lazy import heavy modules on first access."""
    if name == "MultiDatasetDataModule":
        from .datamodule import MultiDatasetDataModule
        return MultiDatasetDataModule
    elif name in ("LightningGeneModel", "LegacyLightningGeneModel"):
        from .lightning import LightningGeneModel, LegacyLightningGeneModel
        if name == "LightningGeneModel":
            return LightningGeneModel
        return LegacyLightningGeneModel
    elif name in ("LightningPerturbationExpert", "PerturbationDataModule"):
        from .perturbation_lightning import LightningPerturbationExpert, PerturbationDataModule
        if name == "LightningPerturbationExpert":
            return LightningPerturbationExpert
        return PerturbationDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
