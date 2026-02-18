"""
Example-specific loss functions pinned as a stability layer.

These functions are used by the example fitting scripts. They are either
re-exported from the core library or defined here to decouple examples from
future core library changes.

A ``LOSS_REGISTRY`` dict maps string names (used in YAML configs) to the
actual loss callables.
"""

from __future__ import annotations

from jaxent.src.opt.base import JaxEnt_Loss

# Core library losses (re-exported for stability)
from jaxent.src.opt.losses import (
    hdx_uptake_eye_MSE_loss,
    hdx_uptake_MAE_loss_vectorized,
    hdx_uptake_sigma_MSE_loss,
    maxent_convexKL_loss,
    maxent_L2_loss,
    model_params_L2_loss,
)

# Alias for backward compatibility with example scripts
hdx_uptake_mean_centred_MSE_loss = hdx_uptake_eye_MSE_loss
hdx_uptake_MSE_loss = hdx_uptake_sigma_MSE_loss


# ---------------------------------------------------------------------------
# Registry: string name → callable (for YAML config lookup)
# ---------------------------------------------------------------------------

LOSS_REGISTRY: dict[str, JaxEnt_Loss] = {
    # Primary data-fit losses
    "hdx_uptake_mean_centred_MSE_loss": hdx_uptake_mean_centred_MSE_loss,
    "hdx_uptake_eye_MSE_loss": hdx_uptake_eye_MSE_loss,
    "hdx_uptake_MSE_loss": hdx_uptake_MSE_loss,
    "hdx_uptake_sigma_MSE_loss": hdx_uptake_sigma_MSE_loss,
    "hdx_uptake_MAE_loss_vectorized": hdx_uptake_MAE_loss_vectorized,
    # Regularisation losses
    "maxent_convexKL_loss": maxent_convexKL_loss,
    "maxent_L2_loss": maxent_L2_loss,
    "model_params_L2_loss": model_params_L2_loss,
}


def get_loss_function_by_name(name: str) -> JaxEnt_Loss:
    """Look up a loss function by its string name.

    Raises ``KeyError`` with available names if *name* is not found.
    """
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise KeyError(f"Unknown loss function '{name}'. Available: {available}")
    return LOSS_REGISTRY[name]
