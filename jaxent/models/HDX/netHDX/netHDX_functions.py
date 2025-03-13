import numpy as np

from jaxent.models.config import NetHDXConfig
from jaxent.models.HDX.netHDX.features import NetHDX_input_features, NetHDX_output_features
from jaxent.types.base import ForwardPass


class NetHDX_ForwardPass(ForwardPass):
    def __call__(
        self, features: NetHDX_input_features, parameters: NetHDXConfig
    ) -> NetHDX_output_features:
        # Process contact matrices to calculate protection factors
        # This is a placeholder - implement actual netHDX calculation here
        avg_contacts = np.mean(features.contact_matrices, axis=0)
        log_pf = np.log10(np.sum(avg_contacts, axis=1)).tolist()

        return NetHDX_output_features(log_Pf=log_pf, k_ints=None)
