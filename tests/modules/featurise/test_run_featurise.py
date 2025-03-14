from MDAnalysis import Universe

from jaxent.featurise import run_featurise
from jaxent.interfaces.builder import Experiment_Builder
from jaxent.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.types.config import FeaturiserSettings


def test_run_featurise():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "./tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"

    test_universe = Universe(topology_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feat_top = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features)


def test_run_featurise_ensemble():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "./tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "./tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feat_top = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features)


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    # set env XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise jax will preallocate 75% of memory
    test_run_featurise()
    # test_quick_optimiser()
    # test_uptake_optimiser()
    # test_run_featurise_ensemble()
