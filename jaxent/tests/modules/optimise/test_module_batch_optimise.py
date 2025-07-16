import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)


# def test_batch_optimise():
#     """Test batch optimization with multiple datasets."""
#     bv_config = BV_model_Config()
#     featuriser_settings = FeaturiserSettings(name="BV", batch_size=10)  # Use batch processing

#     topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
#     trajectory_path = (
#         "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
#     )
#     test_universe = Universe(topology_path, trajectory_path)

#     universes = [test_universe]
#     models = [BV_model(bv_config)]
#     ensemble = Experiment_Builder(universes, models)

#     features, feature_topology = run_featurise(ensemble, featuriser_settings)
#     assert len(features) == len(models)

#     BV_features: BV_input_features = features[0]
#     trajectory_length = BV_features.features_shape[1]

#     params = Simulation_Parameters(
#         frame_weights=jnp.ones(trajectory_length) / trajectory_length,
#         frame_mask=jnp.ones(trajectory_length),
#         model_parameters=[bv_config.forward_parameters],
#         forward_model_weights=jnp.ones(1),
#         forward_model_scaling=jnp.ones(1),
#         normalise_loss_functions=jnp.ones(1),
#     )

#     simulation = Simulation(forward_models=models, input_features=features, params=params)
#     simulation.initialise()

#     opt_settings = OptimiserSettings(name="test_batch", n_steps=10)

#     # Create experimental dataset
#     top_segments = Partial_Topology.find_common_residues(
#         universes, exclude_selection="(resname PRO or resid 1) "
#     )[0]

#     exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in feature_topology[0]]

#     dataset = ExpD_Dataloader(data=exp_data)

#     splitter = DataSplitter(
#         dataset,
#         random_seed=42,
#         ensemble=universes,
#         common_residues=set(feature_topology[0]),
#     )
#     train_data, val_data = splitter.random_split()

#     # Create sparse maps
#     train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)
#     val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

#     dataset.train = Dataset(
#         data=train_data,
#         y_true=jnp.array([data.extract_features() for data in train_data]),
#         residue_feature_ouput_mapping=train_sparse_map,
#     )

#     dataset.val = Dataset(
#         data=val_data,
#         y_true=jnp.array([data.extract_features() for data in val_data]),
#         residue_feature_ouput_mapping=val_sparse_map,
#     )

#     # Run batch optimization
#     opt_simulation = run_optimise(
#         simulation,
#         data_to_fit=(dataset,),
#         config=opt_settings,
#         forward_models=models,
#         indexes=[0],
#         loss_functions=[hdx_pf_l2_loss],
#     )

#     # Verify results
#     assert opt_simulation is not None
#     result_simulation, history = opt_simulation
#     assert result_simulation is not None
#     assert history is not None

#     del simulation
#     jax.clear_caches()


# if __name__ == "__main__":
#     test_batch_optimise()
