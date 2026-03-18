import jax.numpy as jnp
from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.models.XLMS.features import XLMS_input_features, XLMS_output_features
from jaxent.src.models.XLMS.parameters import XLMS_Model_Parameters


class XLMS_distance_ForwardPass(
    ForwardPass[XLMS_input_features, XLMS_output_features, XLMS_Model_Parameters]
):
    """Identity forward pass: returns frame-averaged pairwise distance matrix.

    The PairIndexMapping applied by create_functional_loss extracts
    the specific cross-linked residue pairs from the full matrix.
    """
    def __call__(
        self,
        input_features: XLMS_input_features,
        parameters: XLMS_Model_Parameters,
    ) -> XLMS_output_features:
        return XLMS_output_features(distances=jnp.asarray(input_features.distances))
