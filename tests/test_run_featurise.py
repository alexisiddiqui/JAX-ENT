import jax.numpy as jnp
from MDAnalysis import Universe

from jaxent.config.base import FeaturiserSettings, OptimiserSettings
from jaxent.datatypes import (
    Experiment_Ensemble,
    Experimental_Dataset,
    HDX_peptide,
    HDX_protection_factor,
    Simulation,
    Simulation_Parameters,
)
from jaxent.featurise import run_featurise
from jaxent.forwardmodels.models import (
    BV_input_features,
    BV_model,
    BV_model_Config,
    linear_BV_model,
    linear_BV_model_Config,
)
from jaxent.lossfn.base import (
    hdx_pf_l2_loss,
    hdx_pf_mae_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_monotonicity_loss,
)
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
    uptake_config = linear_BV_model_Config(num_timepoints=3)

    models = [
        BV_model(bv_config),
        BV_model(bv_config),
        linear_BV_model(uptake_config),
        linear_BV_model(uptake_config),
    ]

    ensemble = Experiment_Ensemble(universes, models)

    features = run_featurise(ensemble, featuriser_settings, forward_models=models)
    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    bv_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        model_parameters=[
            bv_config.forward_parameters,
            bv_config.forward_parameters,
            uptake_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=bv_params)

    simulation.initialise()
    test_prediction = simulation.forward()
    print("test prediction")
    print(test_prediction)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    fake_pfs = [
        HDX_protection_factor(protection_factor=10, top=None) for _ in range(features_length)
    ]

    fake_uptake = [HDX_peptide(top=None, dfrac=[0.0001, 0.5, 1]) for _ in range(features_length)]

    # print(fake_pfs)

    updake_dataset = Experimental_Dataset(data=fake_uptake)
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
        data_to_fit=(dataset, pf_prior, updake_dataset, None),
        config=opt_settings,
        forward_models=models,
        loss_functions=[
            hdx_pf_l2_loss,
            hdx_pf_mae_loss,
            hdx_uptake_l2_loss,
            hdx_uptake_monotonicity_loss,
        ],
    )


def test_uptake_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]
    uptake_config = linear_BV_model_Config(num_timepoints=3)

    models = [linear_BV_model(uptake_config), linear_BV_model(uptake_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features = run_featurise(ensemble, featuriser_settings, forward_models=models)
    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    # bv_params = Simulation_Parameters(
    #     frame_weights=jnp.ones(trajectory_length) / trajectory_length,
    #     model_parameters=[bv_config.forward_parameters],
    #     forward_model_weights=jnp.ones(1),
    # )
    linear_bv_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        model_parameters=[uptake_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
    )

    params = [linear_bv_params]

    simulation = Simulation(forward_models=models, input_features=features, params=linear_bv_params)

    simulation.initialise()
    test_prediction = simulation.forward()
    print("test prediction")
    print(test_prediction)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    fake_pfs = [
        HDX_protection_factor(protection_factor=10, top=None) for _ in range(features_length)
    ]

    fake_uptake = [HDX_peptide(top=None, dfrac=[0.0001, 0.5, 1]) for _ in range(features_length)]

    # print(fake_pfs)

    updake_dataset = Experimental_Dataset(data=fake_uptake)
    #
    dataset = Experimental_Dataset(data=fake_pfs)
    # extract the protection factors from the unoptimsied prediction log_Pfs
    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=None) for pf in test_prediction[0].uptake
    ]

    pf_prior = Experimental_Dataset(data=pf_prior)

    print(dataset.y_true)
    print(dataset.y_true.shape)

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=[updake_dataset, None],
        config=opt_settings,
        forward_models=models,
        loss_functions=[hdx_uptake_l2_loss, hdx_uptake_monotonicity_loss],
    )


if __name__ == "__main__":
    import os

    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    # set env XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise jax will preallocate 75% of memory
    # test_run_featurise()
    # test_quick_optimiser()
    # test_uptake_optimiser()
    # test_run_featurise_ensemble()

    # try running on jax cpu
    with jax.default_device(jax.devices("gpu")[0]):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        test_regularised_optimiser()
