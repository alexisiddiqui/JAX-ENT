from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config


def test_run_featurise():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feat_top = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features)


def test_run_featurise_ensemble():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
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
