from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.XLMS.config import XLMS_Config
from jaxent.src.models.XLMS.features import XLMS_input_features
from jaxent.src.models.XLMS.forward import XLMS_distance_ForwardPass
from jaxent.src.models.XLMS.parameters import XLMS_Model_Parameters


class XLMS_distance_model(ForwardModel[XLMS_Model_Parameters, XLMS_input_features, XLMS_Config]):
    """XL-MS ensemble distance model.

    Input: pre-computed pairwise CA-CA distances, shape (n_residues, n_residues, n_frames).
    The forward pass returns the frame-averaged distance matrix. PairIndexMapping
    then extracts the specific cross-linked residue pairs for loss computation.

    featurise() raises NotImplementedError — distances must be pre-computed externally
    and passed as XLMS_input_features directly.
    """

    def __init__(self, config: XLMS_Config) -> None:
        super().__init__(config=config)
        self.forward: dict[m_key, ForwardPass] = {
            m_key("XLMS_distance"): XLMS_distance_ForwardPass()
        }

    def initialise(self, ensemble: list) -> bool:
        return True

    def featurise(self, ensemble: list):
        raise NotImplementedError(
            "XLMS distances must be pre-computed externally. "
            "Create XLMS_input_features directly from a (n_residues, n_residues, n_frames) array."
        )
