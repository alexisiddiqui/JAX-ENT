import jax.numpy as jnp
from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.models.SAXS.features import (
    SAXS_curve_input_features, SAXS_basis_input_features, SAXS_output_features,
)
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters, SAXS_Debye_Parameters


class SAXS_direct_ForwardPass(
    ForwardPass[SAXS_curve_input_features, SAXS_output_features, SAXS_direct_Model_Parameters]
):
    """Identity forward pass: returns frame-averaged I(q) directly."""
    def __call__(
        self,
        input_features: SAXS_curve_input_features,
        parameters: SAXS_direct_Model_Parameters,
    ) -> SAXS_output_features:
        return SAXS_output_features(intensity=jnp.asarray(input_features.intensities))


class SAXS_DebyeForwardPass(
    ForwardPass[SAXS_basis_input_features, SAXS_output_features, SAXS_Debye_Parameters]
):
    """Debye 6-term cross-term combination.

    Input basis_profiles shape: (6, n_q) after frame averaging.
    Order: [Ivv, Ive, Ivh, Iee, Ieh, Ihh]
    Formula:
        I_ens = Ivv - 2*c1*Ive + 2*c2*Ivh + c1^2*Iee - 2*c1*c2*Ieh + c2^2*Ihh
        I_calc = c * I_ens + b
    """
    def __call__(
        self,
        input_features: SAXS_basis_input_features,
        parameters: SAXS_Debye_Parameters,
    ) -> SAXS_output_features:
        bp = jnp.asarray(input_features.basis_profiles)  # (6, n_q)
        Ivv, Ive, Ivh, Iee, Ieh, Ihh = bp[0], bp[1], bp[2], bp[3], bp[4], bp[5]
        c1, c2, c, b = parameters.c1, parameters.c2, parameters.c, parameters.b
        I_ens = (Ivv
                 - 2 * c1 * Ive
                 + 2 * c2 * Ivh
                 + c1 ** 2 * Iee
                 - 2 * c1 * c2 * Ieh
                 + c2 ** 2 * Ihh)
        I_calc = c * I_ens + b
        return SAXS_output_features(intensity=I_calc)
