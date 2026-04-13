import jax
import jax.numpy as jnp

from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters


# fix the typing to use jax arrays
class BV_ForwardPass(ForwardPass[BV_input_features, BV_output_features, BV_Model_Parameters]):
    average_first: bool = True  # operate per-frame then average outputs

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
    average_first: bool = True  # operate per-frame then average outputs

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
        time_points = parameters.timepoints.reshape(-1)  # (n_timepoints,)

        # Compute protection factors per frame: (n_residues, n_frames)
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        pf = jnp.exp(log_pf)  # (n_residues, n_frames)

        # Select kints shape based on pf dimensionality:
        #   pf 1-D (n_residues,)        → features were pre-averaged (average_first=True)
        #   pf 2-D (n_residues, n_frames) → per-frame path (average_first=False)
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
    average_first: bool = True  # operate per-frame then average outputs
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
