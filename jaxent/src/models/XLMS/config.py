from jaxent.src.custom_types.config import BaseConfig
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.models.XLMS.parameters import XLMS_Model_Parameters


class XLMS_Config(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.key = m_key("XLMS_distance")

    @property
    def forward_parameters(self) -> Model_Parameters:
        return XLMS_Model_Parameters()
