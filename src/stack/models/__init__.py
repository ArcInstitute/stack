"""Modularized model package for StateICL."""

# Lazy imports to avoid triggering scvi at module load time

__all__ = [
    "StateICLModel",
    "scShiftAttentionModel",
    "ICL_FinetunedModel",
    "PerturbationExpert",
    "PerturbationExpertConfig",
]


def __getattr__(name):
    """Lazy import heavy modules on first access."""
    if name in ("StateICLModel", "scShiftAttentionModel"):
        from .core import StateICLModel, scShiftAttentionModel
        if name == "StateICLModel":
            return StateICLModel
        return scShiftAttentionModel
    elif name == "ICL_FinetunedModel":
        from .finetune import ICL_FinetunedModel
        return ICL_FinetunedModel
    elif name in ("PerturbationExpert", "PerturbationExpertConfig"):
        from .perturbation import PerturbationExpert, PerturbationExpertConfig
        if name == "PerturbationExpert":
            return PerturbationExpert
        return PerturbationExpertConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
