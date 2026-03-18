from jaxent.src.custom_types.config import BaseConfig
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters, SAXS_Debye_Parameters


class SAXS_Config(BaseConfig):
    mode: str = "reweighted"

    def __init__(self, mode: str = "reweighted") -> None:
        super().__init__()
        if mode not in ("reweighted", "debye_6term"):
            raise ValueError(f"Unknown SAXS mode: {mode!r}. Use 'reweighted' or 'debye_6term'.")
        self.mode = mode
        self.key = m_key("SAXS_Iq")

    @property
    def forward_parameters(self) -> Model_Parameters:
        if self.mode == "reweighted":
            return SAXS_direct_Model_Parameters()
        return SAXS_Debye_Parameters()


class SAXS_direct_Config(SAXS_Config):
    """Direct (reweighted) SAXS config. Accepts q_values for reference."""
    def __init__(self, q_values=None) -> None:
        super().__init__(mode="reweighted")
        self.q_values = q_values
