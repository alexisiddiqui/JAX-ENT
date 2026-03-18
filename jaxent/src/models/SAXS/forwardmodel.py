from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.SAXS.config import SAXS_Config
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.models.SAXS.forward import SAXS_ReweightedForwardPass, SAXS_DebyeForwardPass
from jaxent.src.models.SAXS.parameters import SAXS_Reweighted_Parameters


class SAXS_model(ForwardModel[SAXS_Reweighted_Parameters, SAXS_curve_input_features, SAXS_Config]):
    """SAXS ensemble reweighting model.

    'reweighted' mode: pre-computed I(q) per structure → frame-weighted average.
    'debye_6term' mode: pre-computed 6 basis profiles → Debye cross-term combination.

    featurise() raises NotImplementedError — curves must be pre-computed externally
    (e.g. CRYSOL, FoXS, Pepsi-SAXS) and passed as SAXS_curve_input_features
    or SAXS_basis_input_features directly.
    """

    def __init__(self, config: SAXS_Config) -> None:
        super().__init__(config=config)
        if config.mode == "reweighted":
            self.forward: dict[m_key, ForwardPass] = {
                m_key("SAXS_Iq"): SAXS_ReweightedForwardPass()
            }
        else:
            self.forward = {m_key("SAXS_Iq"): SAXS_DebyeForwardPass()}

    def initialise(self, ensemble: list) -> bool:
        return True

    def featurise(self, ensemble: list):
        raise NotImplementedError(
            "SAXS curves are pre-computed externally. "
            "Create SAXS_curve_input_features or SAXS_basis_input_features directly."
        )
