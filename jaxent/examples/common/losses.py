"""
Example-specific loss functions pinned as a stability layer.

These functions are used by the example fitting scripts. They are either
re-exported from the core library or defined here to decouple examples from
future core library changes.

A ``LOSS_REGISTRY`` dict maps string names (used in YAML configs) to the
actual loss callables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxent.src.opt.base import JaxEnt_Loss
from .moprp_weighted_loss import (
    hdx_uptake_inv_sd_MSE_loss,
    hdx_uptake_inv_variance_MSE_loss,
    hdx_uptake_moprp_raw_weighted_MSE_loss,
)

# Core library losses (re-exported for stability)
from jaxent.src.opt.losses import (
    hdx_pf_l2_loss,
    hdx_uptake_eye_MSE_loss,
    hdx_uptake_MAE_loss_vectorized,
    hdx_uptake_sigma_MSE_loss,
    maxent_convexKL_loss,
    maxent_L2_loss,
    model_params_L1_loss,
    model_params_L2_loss,
    hdx_uptake_mean_centred_eye_MSE_loss
)

def maxent_reverse_KL_logp0_loss(model, log_p0, prediction_index):
    """KL(w || p0) = sum_i w_i (log w_i - log p0_i) for the frame-weight simplex w.

    The reference enters as ``log_p0`` (shape ``(n_frames,)``, normalised in log
    space) so that frames with p0 = 0 (log p0 very negative but finite) stay
    finite.  ``w`` is taken as ``log_softmax`` of the optimiser's frame logits,
    i.e. exactly the simplex the forward pass uses.  This is the direction used
    by the MoPrP recovery-gulf study (docs/recovery_gulf_20261103/selgci,
    ``q0_common.fit``); ``maxent_convexKL_loss`` is the opposite direction
    KL(p0 || w).  Intended for ``run_optimization(frame_regularizer=(name, this,
    log_p0, 1/S))``.  Returns ``(loss, loss)`` (no train/val split).
    """
    log_w = jax.nn.log_softmax(model.params.frame_weight_logits)
    loss = jnp.sum(jnp.exp(log_w) * (log_w - jnp.asarray(log_p0)))
    return loss, loss


# Alias for backward compatibility with example scripts
hdx_uptake_mean_centred_MSE_loss = hdx_uptake_mean_centred_eye_MSE_loss
hdx_uptake_MSE_loss = hdx_uptake_eye_MSE_loss


# ---------------------------------------------------------------------------
# Registry: string name → callable (for YAML config lookup)
# ---------------------------------------------------------------------------

LOSS_REGISTRY: dict[str, JaxEnt_Loss] = {
    "hdx_uptake_inv_sd_MSE_loss": hdx_uptake_inv_sd_MSE_loss,
    "hdx_uptake_inv_variance_MSE_loss": hdx_uptake_inv_variance_MSE_loss,
    "hdx_uptake_moprp_raw_weighted_MSE_loss": hdx_uptake_moprp_raw_weighted_MSE_loss,
    # Primary data-fit losses
    "hdx_pf_l2_loss": hdx_pf_l2_loss,
    "hdx_uptake_mean_centred_MSE_loss": hdx_uptake_mean_centred_MSE_loss,
    "hdx_uptake_eye_MSE_loss": hdx_uptake_eye_MSE_loss,
    "hdx_uptake_MSE_loss": hdx_uptake_MSE_loss,
    "hdx_uptake_sigma_MSE_loss": hdx_uptake_sigma_MSE_loss,
    "hdx_uptake_MAE_loss_vectorized": hdx_uptake_MAE_loss_vectorized,
    # Regularisation losses
    "maxent_convexKL_loss": maxent_convexKL_loss,
    "maxent_L2_loss": maxent_L2_loss,
    "maxent_reverse_KL_logp0_loss": maxent_reverse_KL_logp0_loss,
    "model_params_L1_loss": model_params_L1_loss,
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
