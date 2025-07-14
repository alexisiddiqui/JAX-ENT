import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from MDAnalysis import Universe

from jaxent.src.custom_types.config import (
    FeaturiserSettings,
    Optimisable_Parameters,
    OptimiserSettings,
)
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_input_features, BV_model
from jaxent.src.opt.losses import (
    HDX_uptake_KL_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_MAE_loss,
    max_entropy_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import (
    run_optimise,
)
from jaxent.tests.plots.datasplitting import plot_split_visualization
from jaxent.tests.plots.optimisation import (
    plot_frame_weights_heatmap,
    plot_loss_components,
    plot_total_losses,
)


# Ensure output directory exists
def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    output_dir = "tests/_plots/module_optimise"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# Add these visualizations to the test function
def visualize_optimization_results(train_data, val_data, exp_data, opt_simulation):
    """
    Create and display all visualization plots.
    """
    simulation, history = opt_simulation

    # Create all plots
    split_fig = plot_split_visualization(train_data, val_data, exp_data)
    total_loss_fig = plot_total_losses(history)
    loss_fig = plot_loss_components(history)
    weights_fig = plot_frame_weights_heatmap(history)

    # Display plots
    figs = [split_fig, total_loss_fig, loss_fig, weights_fig]
    fig_names = ["split", "total_loss", "loss", "weights"]
    output_dir = ensure_output_dir()
    for fig, names in zip(figs, fig_names):
        output_path = os.path.join(output_dir, f"{names}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)


def test_uptake_optimiser():
    bv_config = BV_model_Config(num_timepoints=3)

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)
    segs_data = (
        "/home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/BPTI/BPTI_residue_segs_trimmed.txt"
    )
    dfrac_data = "/home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/BPTI/BPTI_expt_dfracs_clean_trimmed.dat"

    with open(segs_data, "r") as f:
        segs_text = [line.strip() for line in f.readlines()]
        segs = [line.split() for line in segs_text]

    segs = [[start, end] for start, end in segs]

    exp_residues = [int(seg[1]) for seg in segs]

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    exp_topology = [top for top in feature_topology[0] if top.residue_end in exp_residues]

    print("filtered_topology", [top.residue_end for top in exp_topology])
    print("filtered_topology", [top.residue_start for top in exp_topology])

    # breakpoint()

    with open(dfrac_data, "r") as f:
        # skip first line and then read in vals
        dfrac_text = [line.strip() for line in f.readlines()[1:]]
        dfracs = [line.split() for line in dfrac_text]

    dfracs = [jnp.array(line, dtype=float) for line in dfracs]
    dfrac_dict = {res: dfrac for res, dfrac in zip(exp_residues, dfracs)}
    # print("feature_topology", [top.fragment_index for top in feature_topology[0]])
    # print(feature_topology)
    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    # features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(4),
        forward_model_scaling=jnp.array([1.0, 10.0, 10, 10]),
        normalise_loss_functions=jnp.zeros(4),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].uptake)
    print(test_prediction[0].uptake.shape)

    prior_pfs = test_prediction[0].uptake.T

    # print("prior_pfs", prior_pfs)

    prior_data = [
        HDX_peptide(dfrac=_prior_df, top=top) for _prior_df, top in zip(prior_pfs, exp_topology)
    ]

    print("prior_data", prior_data)

    opt_settings = OptimiserSettings(name="test", n_steps=1000, convergence=1e-5, tolerance=1e-5)

    # create fake experimental dataset

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [HDX_peptide(dfrac=_dfrac, top=top) for _dfrac, top in zip(dfracs, exp_topology)]
    # print("exp_data", exp_data)
    dataset = ExpD_Dataloader(data=exp_data)
    # top_segments = [data.top for data in dataset.data]
    # print("top_segments", [top.fragment_index for top in top_segments])
    # print("top_segments", [top.residue_start for top in top_segments])

    print("type", type(feature_topology[0]))
    # create random split
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=universes,
        common_residues=set(feature_topology[0]),
    )
    train_data, val_data = splitter.random_split()
    # print("train_data", [data.top.fragment_index for data in train_data])
    # print("train_data", [data.top.residue_start for data in train_data])

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

    dataset.train = Dataset(
        data=train_data,
        y_true=jnp.array([data.extract_features() for data in train_data]),
        residue_feature_ouput_mapping=train_sparse_map,
    )

    dataset.val = Dataset(
        data=val_data,
        y_true=jnp.array([data.extract_features() for data in val_data]),
        residue_feature_ouput_mapping=val_sparse_map,
    )

    dataset.test = Dataset(
        data=exp_data,
        y_true=jnp.array([data.extract_features() for data in exp_data]),
        residue_feature_ouput_mapping=test_sparse_map,
    )

    optimiser = OptaxOptimizer(
        parameter_masks={Optimisable_Parameters.frame_weights},
    )

    prior_dataset = ExpD_Dataloader(data=prior_data)

    prior_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_data)
    prior_dataset.train = Dataset(
        data=prior_data,
        y_true=jnp.array([data.extract_features() for data in prior_data]),
        residue_feature_ouput_mapping=prior_sparse_map,
    )
    prior_dataset.val = Dataset(
        data=prior_data,
        y_true=jnp.array([data.extract_features() for data in prior_data]),
        residue_feature_ouput_mapping=prior_sparse_map,
    )

    prior_dataset.test = Dataset(
        data=prior_data,
        y_true=jnp.array([data.extract_features() for data in prior_data]),
        residue_feature_ouput_mapping=prior_sparse_map,
    )

    opt_simulation = run_optimise(
        simulation,
        optimizer=optimiser,
        data_to_fit=(
            dataset,
            simulation.params,
            prior_dataset,
            prior_dataset,
        ),
        config=opt_settings,
        forward_models=models,
        indexes=[
            0,
            0,
            0,
            0,
        ],
        # initialise=True,
        loss_functions=[
            hdx_uptake_l2_loss,
            max_entropy_loss,
            hdx_uptake_MAE_loss,
            HDX_uptake_KL_loss,
        ],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)

    del simulation
    jax.clear_caches()  # Add this line


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # test_underscore_optimiser()
    # test_quick_optimiser()
    test_uptake_optimiser()
