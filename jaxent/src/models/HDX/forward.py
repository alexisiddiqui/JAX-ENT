import jax
import jax.numpy as jnp
import numpy as np

from jaxent.src.custom_types.base import FrameAveragingMode, ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters


# fix the typing to use jax arrays
class BV_ForwardPass(ForwardPass[BV_input_features, BV_output_features, BV_Model_Parameters]):
    frame_averaging_mode: FrameAveragingMode = "log_pf"

    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> BV_output_features:
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # print("Model parameters bc, bh:", bc, bh)

        # Convert lists to numpy arrays for computation
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        # print("Contact shapes:", heavy_contacts.shape, acceptor_contacts.shape)
        # print("Sample contacts:", heavy_contacts[0, :5], acceptor_contacts[0, :5])

        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)

        # Convert back to list for output
        log_pf_list = log_pf
        # print("Calculated log_pf:", log_pf[:5])

        return BV_output_features(log_Pf=log_pf_list, k_ints=None)


class BV_uptake_ForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, BV_Model_Parameters]
):
    frame_averaging_mode: FrameAveragingMode = "log_pf"

    def __init__(
        self,
        frame_averaging_mode: FrameAveragingMode = "log_pf",
        frame_groups=None,
    ) -> None:
        self.frame_averaging_mode = frame_averaging_mode
        self.frame_group_masks: tuple[np.ndarray, ...] | None = None
        if frame_groups is not None:
            self.set_frame_groups(frame_groups)

    def set_frame_groups(self, assignments) -> None:
        assignments = np.asarray(assignments)
        if assignments.ndim != 1:
            raise ValueError("frame-group assignments must be one-dimensional")
        self.frame_group_masks = tuple(
            assignments == label for label in np.unique(assignments)
        )

    def average_frames(
        self,
        input_features: BV_input_features,
        parameters: BV_Model_Parameters,
        frame_weights,
        implementation: str = "tensordot",
    ) -> uptake_BV_output_features:
        del implementation  # both specialised reductions are explicit dot products
        bc, bh = parameters.bv_bc, parameters.bv_bh
        log_pf = (
            bc * jnp.asarray(input_features.heavy_contacts)
            + bh * jnp.asarray(input_features.acceptor_contacts)
        )
        k_ints = jnp.asarray(input_features.k_ints)
        timepoints = jnp.asarray(parameters.timepoints).reshape(-1)
        rates = k_ints[:, None] * jnp.exp(-log_pf)

        if self.frame_averaging_mode == "rate":
            mean_rate = rates @ frame_weights
            uptake = 1.0 - jnp.exp(-timepoints[:, None] * mean_rate[None, :])
            return uptake_BV_output_features(uptake)
        if self.frame_averaging_mode == "frame_uptake":
            uptake = jax.vmap(
                lambda timepoint: (1.0 - jnp.exp(-timepoint * rates))
                @ frame_weights
            )(timepoints)
            return uptake_BV_output_features(uptake)
        if self.frame_averaging_mode != "uptake":
            raise ValueError(
                f"average_frames cannot handle {self.frame_averaging_mode!r}"
            )
        if self.frame_group_masks is None:
            raise ValueError("uptake mode requires configured frame-group assignments")

        uptake = jnp.zeros((timepoints.shape[0], rates.shape[0]), dtype=rates.dtype)
        for mask_array in self.frame_group_masks:
            mask = jnp.asarray(mask_array)
            group_weights = jnp.where(mask, frame_weights, 0.0)
            mass = jnp.sum(group_weights)
            safe_mass = jnp.where(mass > 0, mass, 1.0)
            group_rate = rates @ (group_weights / safe_mass)
            group_uptake = 1.0 - jnp.exp(
                -timepoints[:, None] * group_rate[None, :]
            )
            uptake = uptake + jnp.where(mass > 0, mass * group_uptake, 0.0)
        return uptake_BV_output_features(uptake)

    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        # Extract model parameters
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # Convert inputs to JAX arrays
        # heavy_contacts and acceptor_contacts are (n_residues, n_frames)
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        kints = jnp.asarray(input_features.k_ints)  # (n_residues,)
        time_points = jnp.asarray(parameters.timepoints).reshape(-1)  # (n_timepoints,)

        # Compute protection factors per frame: (n_residues, n_frames)
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        pf = jnp.exp(log_pf)  # (n_residues, n_frames)

        # Select kints shape based on pf dimensionality:
        #   pf 1-D (n_residues,) → features were pre-averaged in log-PF mode.
        if pf.ndim == 1:
            kints_for_uptake = kints                            # (n_residues,)
        else:
            kints_for_uptake = jnp.expand_dims(kints, axis=-1)  # (n_residues, 1)

        # Reshape time_points to broadcast over residue (and optional frame) dims without vmap.
        # (n_timepoints,) → (n_timepoints, 1) or (n_timepoints, 1, 1)
        time_reshaped = time_points[(slice(None),) + (None,) * pf.ndim]

        # uptake_per_timepoint: (n_timepoints, n_residues) or (n_timepoints, n_residues, n_frames)
        uptake_per_timepoint = 1 - jnp.exp(-kints_for_uptake * time_reshaped / pf)

        return uptake_BV_output_features(uptake_per_timepoint)


class linear_BV_ForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, linear_BV_Model_Parameters]
):
    """
    Calculate uptake using a linear BV model with bc and bh as parameters at each timepoint.
    """
    frame_averaging_mode: FrameAveragingMode = "log_pf"
    key = m_key("HDX_resPF")

    def __call__(
        self, input_features: BV_input_features, parameters: linear_BV_Model_Parameters
    ) -> uptake_BV_output_features:
        bc, bh = parameters.bv_bc, parameters.bv_bh

        # Convert lists to numpy arrays for computation
        heavy_contacts = jnp.array(input_features.heavy_contacts)
        acceptor_contacts = jnp.array(input_features.acceptor_contacts)

        # compute uptake
        uptake = (bc * heavy_contacts) + (bh * acceptor_contacts)
        # print("uptake")
        # print(uptake)
        return uptake_BV_output_features(uptake=uptake)
