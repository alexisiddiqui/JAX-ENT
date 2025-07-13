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
    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        # Extract model parameters
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # Convert inputs to JAX arrays
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        # print("heavy_contacts", heavy_contacts.shape)
        # print("acceptor_contacts", acceptor_contacts.shape)
        kints = jnp.asarray(input_features.k_ints)
        # print("kints", kints.shape)
        time_points = parameters.timepoints.reshape(-1)
        # print("timepoint shape", time_points.shape)
        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        # print("logpf", log_pf)

        pf = jnp.exp(log_pf).reshape(-1)

        # Vectorized computation of uptake for each timepoint
        def compute_uptake_for_timepoint(timepoint):
            # Calculate protection factor for each residue at this timepoint

            # Calculate uptake for each residue: Df_i = 1 - exp(-kint_i * timepoint/ Pf_i)
            uptake = 1 - jnp.exp(-kints.reshape(-1) * timepoint / pf)
            # print("timepoint", timepoint)
            # print("uptake", uptake.shape)
            return uptake

        # Compute uptake for each timepoint
        uptake_per_timepoint = jax.vmap(compute_uptake_for_timepoint)(time_points)
        # print("uptake_per_timepoint", uptake_per_timepoint.shape)
        # raise NotImplementedError("stop here")
        # Return the list of timepoint-wise residue-wise uptake arrays
        return uptake_BV_output_features(uptake_per_timepoint)


class linear_BV_ForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, linear_BV_Model_Parameters]
):
    """
    Calculate uptake using a linear BV model with bc and bh as parameters at each timepoint.
    """

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
