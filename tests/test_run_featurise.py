import os

import jax.numpy as jnp
from MDAnalysis import Universe

from jaxent.config.base import FeaturiserSettings, OptimiserSettings
from jaxent.datatypes import (
    Experiment_Ensemble,
    Experimental_Dataset,
    HDX_protection_factor,
    Simulation,
    Simulation_Parameters,
)
from jaxent.featurise import run_featurise
from jaxent.forwardmodels.models import BV_input_features, BV_model, BV_model_Config
from jaxent.lossfn.base import hdx_pf_l2_loss, hdx_pf_mae_loss
from jaxent.optimise import run_optimise


def test_run_featurise():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"

    test_universe = Universe(topology_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features)


def test_run_featurise_ensemble():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features)


def test_quick_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    test_prediction = simulation.forward()
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    fake_pfs = [
        HDX_protection_factor(protection_factor=10, top=None) for _ in range(features_length)
    ]

    # print(fake_pfs)
    #
    dataset = Experimental_Dataset(data=fake_pfs)

    print(dataset.y_true)
    print(dataset.y_true.shape)

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset,),
        config=opt_settings,
        forward_models=models,
        loss_functions=[hdx_pf_l2_loss],
    )


def test_regularised_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    test_prediction = simulation.forward()
    print(test_prediction[0])

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    fake_pfs = [
        HDX_protection_factor(protection_factor=10, top=None) for _ in range(features_length)
    ]

    # print(fake_pfs)
    #
    dataset = Experimental_Dataset(data=fake_pfs)
    # extract the protection factors from the unoptimsied prediction log_Pfs
    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=None) for pf in test_prediction[0].log_Pf
    ]

    pf_prior = Experimental_Dataset(data=pf_prior)

    print(dataset.y_true)
    print(dataset.y_true.shape)

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior),
        config=opt_settings,
        forward_models=models,
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss],
    )


if __name__ == "__main__":
    # set env XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise jax will preallocate 75% of memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # test_run_featurise()
    # test_quick_optimiser()
    test_regularised_optimiser()
    # test_run_featurise_ensemble()
