import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Datapoint
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config


def create_synthetic_data(
    ref_structures: list[Universe],
    ref_ratios: list[float],
    trajectory_path: str,
    topology_path: str,
    bv_config=BV_model_Config(num_timepoints=3),
) -> Sequence[ExpD_Datapoint]:
    """
    Create synthetic data based on reference structures and their ratios,
    using the entire ensemble and weighting by state clusters.

    Parameters:
    -----------
    ref_structures : list
        List of reference structure Universes
    ref_ratios : list
        List of ratios for each state [open_ratio, closed_ratio]
    trajectory_path : str
        Path to trajectory file
    topology_path : str
        Path to topology file
    bv_config : BV_model_Config
        Configuration for BV model

    Returns:
    --------
    synthetic_data : list
        List of synthetic HDX peptide datapoints
    """
    print("Creating synthetic data with ensemble clustering...")

    # 1. Compute RMSD between each frame and both reference structures
    reference_paths = [ref.filename for ref in ref_structures]
    rmsd_values = compute_rmsd_to_references(trajectory_path, topology_path, reference_paths)

    # 2. Cluster frames based on minimum RMSD
    cluster_assignments = np.argmin(rmsd_values, axis=1)

    # 3. Create frame weights based on state ratios (60:40 for open:closed)
    n_frames = len(cluster_assignments)
    frame_weights = np.zeros(n_frames)

    # Assign weights to each cluster based on the desired ratio
    for cluster_idx, ratio in enumerate(ref_ratios):
        # Get frames in this cluster
        cluster_frames = np.where(cluster_assignments == cluster_idx)[0]
        if len(cluster_frames) == 0:
            print(f"Warning: No frames found in cluster {cluster_idx}")
            continue

        # Assign equal weights to frames in this cluster, totaling to the desired ratio
        frame_weights[cluster_frames] = ratio / len(cluster_frames)

    # Verify weights sum to 1.0
    frame_weights = frame_weights / np.sum(frame_weights)
    print(f"Created weights for {n_frames} frames, sum: {np.sum(frame_weights)}")
    print(
        f"Open state frames: {np.sum(cluster_assignments == 0)}, weight: {np.sum(frame_weights[cluster_assignments == 0])}"
    )
    print(
        f"Closed state frames: {np.sum(cluster_assignments == 1)}, weight: {np.sum(frame_weights[cluster_assignments == 1])}"
    )

    # 4. Featurize the trajectory
    featuriser_settings = FeaturiserSettings(name="synthetic", batch_size=None)
    models = [BV_model(bv_config)]

    # Create ensemble using the trajectory
    ensemble = Experiment_Builder([Universe(topology_path, trajectory_path)], models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    print(f"Features shape: {features[0].features_shape}")

    # 5. Create synthetic data using the frame weights
    params = Simulation_Parameters(
        frame_weights=jnp.array(frame_weights).reshape(1, -1, 1),
        frame_mask=jnp.ones(n_frames).reshape(1, -1, 1),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.zeros(1, dtype=jnp.float32),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)
    simulation.initialise()
    simulation.forward(params)
    output = simulation.outputs[0]
    print(f"Output shape: {output.uptake.shape}")

    # Create synthetic data
    synthetic_data = [
        HDX_peptide(dfrac=output.uptake[0, i], top=feature_topology[0][i])
        for i in range(output.uptake.shape[1])
    ]

    del simulation
    jax.clear_caches()
    print("Synthetic data created.")
    return synthetic_data
