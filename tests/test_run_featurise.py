from MDAnalysis import Universe

from jaxent.config.base import FeaturiserSettings
from jaxent.datatypes import Experiment_Ensemble
from jaxent.featurise import run_featurise
from jaxent.forwardmodels.models import BV_model, BV_model_Config


def test_run_featurise():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"

    test_universe = Universe(topology_path)

    ensemble = Experiment_Ensemble([test_universe], [BV_model(bv_config)])

    features = run_featurise(ensemble, featuriser_settings)

    assert len(features) == 1

    print(features)


if __name__ == "__main__":
    test_run_featurise()
