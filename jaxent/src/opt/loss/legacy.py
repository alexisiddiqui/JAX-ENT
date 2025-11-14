"""
This file contains the legacy loss functions directly registered in jaxext.src.opt.losses using the legacy loss adapter and added to a Legacy Loss registry so that we can use the original names.
"""

from jaxent.src.opt.loss.base import LegacyLossAdapter, LossRegistry
from jaxent.src.opt.losses import (
    HDX_uptake_convex_KL_loss,
    HDX_uptake_KL_loss,
    L1_frame_weight_consistency_loss,
    convex_KL_frame_weight_consistency_loss,
    corr_frame_weight_consistency_loss,
    cosine_frame_weight_consistency_loss,
    exp_frame_weight_consistency_loss,
    # Consistency losses
    frame_weight_consistency_loss,
    # Functional losses (HDX protection factors)
    hdx_pf_l2_loss,
    hdx_pf_mae_loss,
    hdx_uptake_abs_loss,
    # Functional losses (HDX uptake)
    hdx_uptake_l1_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_MAE_loss,
    hdx_uptake_MAE_loss_vectorized,
    hdx_uptake_mean_centred_l1_loss,
    hdx_uptake_mean_centred_l2_loss,
    hdx_uptake_mean_centred_MAE_loss,
    hdx_uptake_mean_centred_MSE_loss,
    hdx_uptake_monotonicity_loss,
    hdx_uptake_MSE_loss,
    hdxer_mcMSE_loss,
    hdxer_MSE_loss,
    mask_L0_loss,
    # Parameter losses (MaxEnt)
    max_entropy_loss,
    maxent_convexKL_loss,
    maxent_ESS_loss,
    maxent_JSD_loss,
    maxent_L1_loss,
    maxent_L2_loss,
    maxent_W1_loss,
    minent_ESS_loss,
    normalised_frame_weight_consistency_loss,
    sparse_max_entropy_loss,
)

# Register functional losses (HDX protection factors)
LossRegistry.register(
    "legacy_hdx_pf_l2_loss",
    LegacyLossAdapter.wrap_existing_function(hdx_pf_l2_loss),
)

LossRegistry.register(
    "legacy_hdx_pf_mae_loss",
    LegacyLossAdapter.wrap_existing_function(hdx_pf_mae_loss),
)

# Register functional losses (HDX uptake)
LossRegistry.register(
    "legacy_hdx_uptake_l1_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_l1_loss, "functional", flatten=False, name="legacy_hdx_uptake_l1_loss"
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_l2_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_l2_loss, "functional", flatten=False, name="legacy_hdx_uptake_l2_loss"
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_abs_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_abs_loss, "functional", flatten=False, name="legacy_hdx_uptake_abs_loss"
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_mean_centred_l1_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_mean_centred_l1_loss,
        "functional",
        flatten=False,
        name="legacy_hdx_uptake_mean_centred_l1_loss",
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_mean_centred_l2_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_mean_centred_l2_loss,
        "functional",
        flatten=False,
        name="legacy_hdx_uptake_mean_centred_l2_loss",
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_monotonicity_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_monotonicity_loss,
        "functional",
        flatten=False,
        name="legacy_hdx_uptake_monotonicity_loss",
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_MAE_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_MAE_loss, "functional", flatten=False, name="legacy_hdx_uptake_MAE_loss"
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_MSE_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_MSE_loss, "functional", flatten=False, name="legacy_hdx_uptake_MSE_loss"
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_mean_centred_MAE_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_mean_centred_MAE_loss,
        "functional",
        flatten=False,
        name="legacy_hdx_uptake_mean_centred_MAE_loss",
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_mean_centred_MSE_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_mean_centred_MSE_loss,
        "functional",
        flatten=False,
        name="legacy_hdx_uptake_mean_centred_MSE_loss",
    ),
)

LossRegistry.register(
    "legacy_hdx_uptake_MAE_loss_vectorized",
    LegacyLossAdapter.wrap_existing_function(
        hdx_uptake_MAE_loss_vectorized,
        "functional",
        flatten=False,
        name="legacy_hdx_uptake_MAE_loss_vectorized",
    ),
)

LossRegistry.register(
    "legacy_HDX_uptake_KL_loss",
    LegacyLossAdapter.wrap_existing_function(
        HDX_uptake_KL_loss, "functional", flatten=False, name="legacy_HDX_uptake_KL_loss"
    ),
)

LossRegistry.register(
    "legacy_HDX_uptake_convex_KL_loss",
    LegacyLossAdapter.wrap_existing_function(
        HDX_uptake_convex_KL_loss,
        "functional",
        flatten=False,
        name="legacy_HDX_uptake_convex_KL_loss",
    ),
)

LossRegistry.register(
    "legacy_hdxer_MSE_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdxer_MSE_loss, "functional", flatten=False, name="legacy_hdxer_MSE_loss"
    ),
)

LossRegistry.register(
    "legacy_hdxer_mcMSE_loss",
    LegacyLossAdapter.wrap_existing_function(
        hdxer_mcMSE_loss, "functional", flatten=False, name="legacy_hdxer_mcMSE_loss"
    ),
)

# Register parameter losses (MaxEnt)
LossRegistry.register(
    "legacy_max_entropy_loss",
    LegacyLossAdapter.wrap_existing_function(
        max_entropy_loss, "parameter", name="legacy_max_entropy_loss"
    ),
)

LossRegistry.register(
    "legacy_maxent_convexKL_loss",
    LegacyLossAdapter.wrap_existing_function(
        maxent_convexKL_loss, "parameter", name="legacy_maxent_convexKL_loss"
    ),
)

LossRegistry.register(
    "legacy_maxent_JSD_loss",
    LegacyLossAdapter.wrap_existing_function(
        maxent_JSD_loss, "parameter", name="legacy_maxent_JSD_loss"
    ),
)

LossRegistry.register(
    "legacy_maxent_W1_loss",
    LegacyLossAdapter.wrap_existing_function(
        maxent_W1_loss, "parameter", name="legacy_maxent_W1_loss"
    ),
)

LossRegistry.register(
    "legacy_maxent_ESS_loss",
    LegacyLossAdapter.wrap_existing_function(
        maxent_ESS_loss, "parameter", name="legacy_maxent_ESS_loss"
    ),
)

LossRegistry.register(
    "legacy_minent_ESS_loss",
    LegacyLossAdapter.wrap_existing_function(
        minent_ESS_loss, "parameter", name="legacy_minent_ESS_loss"
    ),
)

LossRegistry.register(
    "legacy_maxent_L1_loss",
    LegacyLossAdapter.wrap_existing_function(
        maxent_L1_loss, "parameter", name="legacy_maxent_L1_loss"
    ),
)

LossRegistry.register(
    "legacy_maxent_L2_loss",
    LegacyLossAdapter.wrap_existing_function(
        maxent_L2_loss, "parameter", name="legacy_maxent_L2_loss"
    ),
)

LossRegistry.register(
    "legacy_sparse_max_entropy_loss",
    LegacyLossAdapter.wrap_existing_function(
        sparse_max_entropy_loss, "parameter", name="legacy_sparse_max_entropy_loss"
    ),
)

LossRegistry.register(
    "legacy_mask_L0_loss",
    LegacyLossAdapter.wrap_existing_function(mask_L0_loss, "parameter", name="legacy_mask_L0_loss"),
)

# Register consistency losses
LossRegistry.register(
    "legacy_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        frame_weight_consistency_loss, "consistency", name="legacy_frame_weight_consistency_loss"
    ),
)

LossRegistry.register(
    "legacy_exp_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        exp_frame_weight_consistency_loss,
        "consistency",
        name="legacy_exp_frame_weight_consistency_loss",
    ),
)

LossRegistry.register(
    "legacy_L1_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        L1_frame_weight_consistency_loss,
        "consistency",
        name="legacy_L1_frame_weight_consistency_loss",
    ),
)

LossRegistry.register(
    "legacy_normalised_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        normalised_frame_weight_consistency_loss,
        "consistency",
        name="legacy_normalised_frame_weight_consistency_loss",
    ),
)

LossRegistry.register(
    "legacy_convex_KL_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        convex_KL_frame_weight_consistency_loss,
        "consistency",
        name="legacy_convex_KL_frame_weight_consistency_loss",
    ),
)

LossRegistry.register(
    "legacy_cosine_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        cosine_frame_weight_consistency_loss,
        "consistency",
        name="legacy_cosine_frame_weight_consistency_loss",
    ),
)

LossRegistry.register(
    "legacy_corr_frame_weight_consistency_loss",
    LegacyLossAdapter.wrap_existing_function(
        corr_frame_weight_consistency_loss,
        "consistency",
        name="legacy_corr_frame_weight_consistency_loss",
    ),
)

# Define aliases for backward compatibility (optional)
# Maps original names to legacy names without requiring "legacy_" prefix
LEGACY_ALIASES = {
    "hdx_pf_l2_loss": "legacy_hdx_pf_l2_loss",
    "hdx_pf_mae_loss": "legacy_hdx_pf_mae_loss",
    "hdx_uptake_l2_loss": "legacy_hdx_uptake_l2_loss",
    "max_entropy_loss": "legacy_max_entropy_loss",
    # Add more aliases as needed
}

# Register aliases if needed
for original_name, legacy_name in LEGACY_ALIASES.items():
    LossRegistry.register(original_name, LossRegistry.get(legacy_name))
