"""Utilities for loading pretrained StateICL checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import torch

# Lazy imports to avoid triggering scvi at module load time
# from .models.core import StateICLModel, scShiftAttentionModel
# from .model_finetune import ICL_FinetunedModel

LOGGER = logging.getLogger("stack.model_loading")


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_class: str = "scShiftAttentionModel",
    device: Optional["torch.device"] = None,
    strict: bool = True,
) -> "torch.nn.Module":
    """Load a trained model from a checkpoint into evaluation mode.

    This helper mirrors the logic used by the standalone CLI scripts so that
    users can easily reuse their checkpoints directly from Python after
    installing the project as a package.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - informative failure
        raise ModuleNotFoundError(
            "The 'torch' package is required to load StateICL checkpoints."
        ) from exc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

    LOGGER.info("Loading model from %s", checkpoint_path_obj)

    if checkpoint_path_obj.suffix == ".ckpt":
        checkpoint = torch.load(checkpoint_path_obj, map_location=device)

        if "hyper_parameters" not in checkpoint:
            raise ValueError("No hyperparameters found in checkpoint")

        model_config = checkpoint["hyper_parameters"].get("model_config", {})

        # Lazy imports to avoid triggering scvi at module load time
        if model_class in {"scShiftAttentionModel", "StateICLModel"}:
            from .models.core import StateICLModel
            model = StateICLModel(**model_config)
        elif model_class in {"ICLFinetunedModel", "ICL_FinetunedModel"}:
            from .model_finetune import ICL_FinetunedModel
            model = ICL_FinetunedModel(**model_config)
        else:
            raise ValueError(f"Unknown model class: {model_class}")

        if "state_dict" not in checkpoint:
            raise ValueError("No state_dict found in checkpoint")

        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=strict)

    elif checkpoint_path_obj.suffix in {".pth", ".pt"}:
        checkpoint = torch.load(checkpoint_path_obj, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        elif hasattr(checkpoint, "eval"):
            model = checkpoint
        else:
            raise ValueError("Cannot load model from checkpoint format")
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path_obj.suffix}")

    model.to(device)
    model.eval()
    LOGGER.info("Model loaded successfully on %s", device)
    return model


__all__ = ["load_model_from_checkpoint"]
