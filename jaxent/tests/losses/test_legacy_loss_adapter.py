import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    HAS_PLOTTING_LIBS = True
except ImportError:
    HAS_PLOTTING_LIBS = False
    print(
        "Warning: matplotlib, seaborn, or pandas not available. Performance plots will not be generated."
    )

import jax
import jax.numpy as jnp
import pytest

from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import Dataset
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import Partial_Topology

# Import real classes
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters

# Import LossRegistry from legacy adapter
from jaxent.src.opt.loss.legacy import LossRegistry

# Import only jax_pairwise_cosine_similarity directly (not a legacy loss)
from jaxent.src.opt.losses import jax_pairwise_cosine_similarity
from jaxent.src.utils.jit_fn import jit_Guard

# Global dictionary to store performance results
PERFORMANCE_RESULTS = defaultdict(lambda: defaultdict(dict))

# Define different input size configurations
INPUT_SIZE_CONFIGS = {
    "small": {"num_residues": 10, "num_timepoints": 5, "num_frames": 2},
    "medium": {"num_residues": 50, "num_timepoints": 10, "num_frames": 10},
    # "large": {"num_residues": 100, "num_timepoints": 20, "num_frames": 200},
    # "xlarge": {"num_residues": 200, "num_timepoints": 30, "num_frames": 500},
}


# Create mock experimental datapoints for testing
def create_mock_exp_datapoint(residues, timepoints, fragment_idx, chain="A"):
    """Create a mock experimental datapoint with realistic structure."""
    # Create mock topology
    topology = Partial_Topology(
        chain=chain, residues=list(range(1, residues + 1)), fragment_index=fragment_idx
    )

    # Create mock experimental data based on key type
    if timepoints == 1:
        # Protection factor data - single value per residue
        exp_data = jax.random.uniform(jax.random.PRNGKey(fragment_idx), (residues,))
        key = m_key("HDX_resPF")
    else:
        # Uptake data - one value per timepoint (HDX_peptide structure)
        # Each peptide covers multiple residues but has one uptake value per timepoint
        exp_data = jax.random.uniform(jax.random.PRNGKey(fragment_idx), (timepoints,))
        key = m_key("HDX_peptide")

    # Create a minimal ExpD_Datapoint-like object
    class MockExpDatapoint:
        def __init__(self, data, topology, key):
            self.data = data
            self.top = topology
            self.key = key

        def extract_features(self):
            # For uptake data (HDX_peptide), reshape to (n_timepoints, 1)
            # For protection factor data, return as-is (single value)
            if self.key == m_key("HDX_peptide"):
                return self.data.reshape(-1, 1)
            else:
                return self.data.flatten()

    return MockExpDatapoint(exp_data, topology, key)


def create_dummy_data_with_size(config_name, config):
    """Create dummy data with specified size configuration."""
    key = jax.random.PRNGKey(0)

    # Data for HDX_peptide/HDX_protection_factor losses
    num_residues = config["num_residues"]
    num_timepoints = config["num_timepoints"]
    num_frames = config["num_frames"]
    prediction_index = 0

    # Create real BV input features
    heavy_contacts = jax.random.uniform(key, (num_residues, num_frames))
    acceptor_contacts = jax.random.uniform(key, (num_residues, num_frames))
    input_features = BV_input_features(
        heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts
    )

    # Create feature topology (one per residue)
    feature_topology = []
    for i in range(num_residues):
        topology = Partial_Topology(chain="A", residues=[i + 1], fragment_index=i)
        feature_topology.append(topology)

    # Create mock simulation outputs
    log_Pf_data = jax.random.normal(key, (num_residues,))
    uptake_data = jax.random.uniform(key, (num_timepoints, num_residues), minval=0.0, maxval=1.0)

    # Use real output features
    pf_output = BV_output_features(log_Pf=log_Pf_data)
    uptake_output = uptake_BV_output_features(uptake=uptake_data)

    # Frame weights/mask for simulation
    frame_weights = jax.random.uniform(key, (num_frames,))
    frame_weights = frame_weights / jnp.sum(frame_weights)
    frame_mask = jnp.array([1.0 if i % 2 == 0 else 0.0 for i in range(num_frames)])

    # Real Simulation_Parameters
    sim_params = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=frame_mask,
        model_parameters=[BV_Model_Parameters()],
        forward_model_weights=jnp.ones((1,)),
        normalise_loss_functions=jnp.ones((1,)),
        forward_model_scaling=jnp.ones((1,)),
    )

    # Real Simulation instance
    sim = Simulation(input_features=[input_features], forward_models=[], params=sim_params)
    sim.length = num_frames
    sim._input_features = [input_features]
    sim.outputs = (pf_output, uptake_output)

    # Create experimental datapoints with proper shapes
    exp_datapoints_pf = []
    exp_datapoints_uptake = []

    # Create protection factor datapoints (one per residue, single value each)
    for i in range(num_residues):
        exp_datapoint = create_mock_exp_datapoint(1, 1, i, "A")  # 1 residue, 1 timepoint
        exp_datapoints_pf.append(exp_datapoint)

    # Create uptake datapoints - one fragment covering all residues for all timepoints
    exp_datapoint_uptake = create_mock_exp_datapoint(num_residues, num_timepoints, 0, "A")
    exp_datapoints_uptake = [exp_datapoint_uptake]

    # Create dataloaders for both types
    class MockDataloader:
        def __init__(self, exp_data, is_uptake=False):
            # Create datasets with proper mappings for testing
            y_true_list = []
            for data in exp_data:
                features = data.extract_features()
                y_true_list.append(features)

            # Stack to create y_true with shape (n_fragments, features_per_fragment)
            y_true = jnp.array(y_true_list)

            # Create mapping matrix based on data type
            n_fragments = len(exp_data)

            if is_uptake:
                # For uptake: map full model predictions to single fragment
                mapping_matrix = jnp.ones((1, num_residues)) / num_residues
            else:
                # For protection factors: identity mapping
                mapping_matrix = jnp.eye(min(n_fragments, num_residues), num_residues)

            from jax.experimental import sparse

            sparse_map = sparse.bcoo_fromdense(mapping_matrix)

            self.train = Dataset(
                data=exp_data, y_true=y_true, residue_feature_ouput_mapping=sparse_map
            )
            self.val = Dataset(
                data=exp_data, y_true=y_true, residue_feature_ouput_mapping=sparse_map
            )

    # Create dataloaders with proper expected shapes
    mock_dataloader_pf = MockDataloader(exp_datapoints_pf, is_uptake=False)
    mock_dataloader_uptake = MockDataloader(exp_datapoints_uptake, is_uptake=True)

    # Data for frame weight consistency losses
    consistency_dataset = jax.random.uniform(key, (num_frames, num_frames), minval=0.0, maxval=2.0)
    consistency_dataset = (consistency_dataset + consistency_dataset.T) / 2.0

    return {
        "model": sim,
        "initialised_model": sim,
        "dataset_pf": mock_dataloader_pf,
        "dataset_uptake": mock_dataloader_uptake,
        "sim_params_dataset": sim_params,
        "consistency_dataset": consistency_dataset,
        "prediction_index": prediction_index,
        "num_frames": num_frames,
        "num_residues": num_residues,
        "num_timepoints": num_timepoints,
        "pf_output": pf_output,
        "uptake_output": uptake_output,
        "config_name": config_name,
    }


# Create fixtures for each size configuration
@pytest.fixture(params=list(INPUT_SIZE_CONFIGS.keys()))
def dummy_data_sized(request):
    config_name = request.param
    config = INPUT_SIZE_CONFIGS[config_name]
    return create_dummy_data_with_size(config_name, config)


# List of all legacy loss functions to test using LossRegistry
ALL_LEGACY_LOSS_NAMES = [
    "legacy_hdx_pf_l2_loss",
    "legacy_hdx_pf_mae_loss",
    "legacy_max_entropy_loss",
    "legacy_maxent_convexKL_loss",
    "legacy_maxent_JSD_loss",
    # "legacy_maxent_W1_loss",
    # "legacy_maxent_ESS_loss",
    # "legacy_minent_ESS_loss",
    # "legacy_maxent_L2_loss",
    # "legacy_maxent_L1_loss",
    # "legacy_sparse_max_entropy_loss",
    # "legacy_mask_L0_loss",
    # "legacy_hdx_uptake_l1_loss",
    # "legacy_hdx_uptake_abs_loss",
    # "legacy_hdx_uptake_mean_centred_l1_loss",
    # "legacy_hdx_uptake_mean_centred_l2_loss",
    # "legacy_hdx_uptake_monotonicity_loss",
    # "legacy_frame_weight_consistency_loss",
    # "legacy_exp_frame_weight_consistency_loss",
    # "legacy_L1_frame_weight_consistency_loss",
    # "legacy_normalised_frame_weight_consistency_loss",
    # "legacy_convex_KL_frame_weight_consistency_loss",
    # "legacy_cosine_frame_weight_consistency_loss",
    # "legacy_corr_frame_weight_consistency_loss",
    # "legacy_hdx_uptake_mean_centred_MSE_loss",
    # "legacy_hdxer_MSE_loss",
    # "legacy_hdxer_mcMSE_loss",
    # "legacy_hdx_uptake_mean_centred_MAE_loss",
    # "legacy_hdx_uptake_l2_loss",
    # "legacy_hdx_uptake_MAE_loss",
    # "legacy_hdx_uptake_MSE_loss",
    # "legacy_hdx_uptake_MAE_loss_vectorized",
    # "legacy_HDX_uptake_KL_loss",
    # "legacy_HDX_uptake_convex_KL_loss",
]

ALL_LOSS_FUNCTIONS = {name: LossRegistry.get(name) for name in ALL_LEGACY_LOSS_NAMES}


def get_jit_static_args(loss_name):
    """Return the static_argnums for JIT compilation based on loss function type."""
    # Use legacy names for matching
    if loss_name in [
        "legacy_hdx_pf_l2_loss",
        "legacy_hdx_pf_mae_loss",
        "legacy_hdx_uptake_l1_loss",
        "legacy_hdx_uptake_abs_loss",
        "legacy_hdx_uptake_mean_centred_l1_loss",
        "legacy_hdx_uptake_mean_centred_l2_loss",
        "legacy_hdx_uptake_mean_centred_MSE_loss",
        "legacy_hdxer_MSE_loss",
        "legacy_hdxer_mcMSE_loss",
        "legacy_hdx_uptake_mean_centred_MAE_loss",
        "legacy_hdx_uptake_l2_loss",
        "legacy_hdx_uptake_MAE_loss",
        "legacy_hdx_uptake_MSE_loss",
        "legacy_hdx_uptake_MAE_loss_vectorized",
        "legacy_HDX_uptake_KL_loss",
        "legacy_HDX_uptake_convex_KL_loss",
    ]:
        return [0, 1, 2]
    elif loss_name == "legacy_hdx_uptake_monotonicity_loss":
        return [0, 2]
    elif loss_name in [
        "legacy_max_entropy_loss",
        "legacy_maxent_convexKL_loss",
        "legacy_maxent_JSD_loss",
        "legacy_maxent_W1_loss",
        "legacy_maxent_ESS_loss",
        "legacy_minent_ESS_loss",
        "legacy_maxent_L2_loss",
        "legacy_maxent_L1_loss",
        "legacy_sparse_max_entropy_loss",
        "legacy_mask_L0_loss",
    ]:
        return [0]
    elif loss_name in [
        "legacy_frame_weight_consistency_loss",
        "legacy_exp_frame_weight_consistency_loss",
        "legacy_L1_frame_weight_consistency_loss",
        "legacy_normalised_frame_weight_consistency_loss",
        "legacy_convex_KL_frame_weight_consistency_loss",
        "legacy_cosine_frame_weight_consistency_loss",
        "legacy_corr_frame_weight_consistency_loss",
    ]:
        return [0, 2]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def prepare_loss_function_args(loss_name, dummy_data):
    """Prepare arguments for loss function based on its signature."""
    # Use legacy names for matching
    model = dummy_data["model"]
    initialised_model = dummy_data["initialised_model"]
    dataset_pf = dummy_data["dataset_pf"]
    dataset_uptake = dummy_data["dataset_uptake"]
    sim_params_dataset = dummy_data["sim_params_dataset"]
    consistency_dataset = dummy_data["consistency_dataset"]
    prediction_index = dummy_data["prediction_index"]

    # Update model outputs and choose correct dataset based on loss function type
    if any(keyword in loss_name for keyword in ["uptake", "HDX_uptake"]):
        model.outputs = [dummy_data["uptake_output"]]
        initialised_model.outputs = [dummy_data["uptake_output"]]
        dataset = dataset_uptake
    elif loss_name in ["legacy_hdxer_MSE_loss", "legacy_hdxer_mcMSE_loss"]:
        model.outputs = [dummy_data["uptake_output"]]
        initialised_model.outputs = [dummy_data["uptake_output"]]
        dataset = dataset_uptake
    else:
        model.outputs = [dummy_data["pf_output"]]
        initialised_model.outputs = [dummy_data["pf_output"]]
        dataset = dataset_pf

    # Determine which arguments to pass based on the loss function signature
    if loss_name in [
        "legacy_hdx_pf_l2_loss",
        "legacy_hdx_pf_mae_loss",
        "legacy_hdx_uptake_l1_loss",
        "legacy_hdx_uptake_abs_loss",
        "legacy_hdx_uptake_mean_centred_l1_loss",
        "legacy_hdx_uptake_mean_centred_l2_loss",
        "legacy_hdx_uptake_mean_centred_MSE_loss",
        "legacy_hdxer_MSE_loss",
        "legacy_hdxer_mcMSE_loss",
        "legacy_hdx_uptake_mean_centred_MAE_loss",
        "legacy_hdx_uptake_l2_loss",
        "legacy_hdx_uptake_MAE_loss",
        "legacy_hdx_uptake_MSE_loss",
        "legacy_hdx_uptake_MAE_loss_vectorized",
        "legacy_HDX_uptake_KL_loss",
        "legacy_HDX_uptake_convex_KL_loss",
    ]:
        return (model, dataset, prediction_index)
    elif loss_name == "legacy_hdx_uptake_monotonicity_loss":
        return (model, None, prediction_index)
    elif loss_name in [
        "legacy_max_entropy_loss",
        "legacy_maxent_convexKL_loss",
        "legacy_maxent_JSD_loss",
        "legacy_maxent_W1_loss",
        "legacy_maxent_ESS_loss",
        "legacy_minent_ESS_loss",
        "legacy_maxent_L2_loss",
        "legacy_maxent_L1_loss",
        "legacy_sparse_max_entropy_loss",
        "legacy_mask_L0_loss",
    ]:
        return (initialised_model, sim_params_dataset, None)
    elif loss_name in [
        "legacy_frame_weight_consistency_loss",
        "legacy_exp_frame_weight_consistency_loss",
        "legacy_L1_frame_weight_consistency_loss",
        "legacy_normalised_frame_weight_consistency_loss",
        "legacy_convex_KL_frame_weight_consistency_loss",
        "legacy_cosine_frame_weight_consistency_loss",
        "legacy_corr_frame_weight_consistency_loss",
    ]:
        return (model, consistency_dataset, prediction_index)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def time_function_execution(func, args, num_runs=10, warmup_runs=3):
    """Time function execution with proper warmup."""
    # Warmup runs
    for _ in range(warmup_runs):
        _ = func(*args)

    # Actual timing
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = func(*args)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "times": times,
    }


@jit_Guard.clear_caches_after()
@pytest.mark.parametrize("loss_name", ALL_LOSS_FUNCTIONS.keys())
@pytest.mark.parametrize("use_jit", [True, False])
def test_loss_function_performance_comprehensive(dummy_data_sized, loss_name, use_jit):
    """Test performance of loss functions with different sizes and JIT configurations."""
    loss_func = ALL_LOSS_FUNCTIONS[loss_name]
    args = prepare_loss_function_args(loss_name, dummy_data_sized)
    config_name = dummy_data_sized["config_name"]

    if use_jit:
        static_argnums = get_jit_static_args(loss_name)

        # wrap legacy adapter object into a plain Python function for JAX jit
        def _callable_loss(*args):
            return loss_func(*args)

        jitted_loss_func = jax.jit(_callable_loss, static_argnums=static_argnums)
        func_to_test = jitted_loss_func
        jit_label = "jit"
    else:
        func_to_test = loss_func
        jit_label = "no_jit"

    # Time the function execution
    timing_results = time_function_execution(func_to_test, args, num_runs=10)

    # Store results globally for summary
    PERFORMANCE_RESULTS[loss_name][config_name][jit_label] = timing_results

    # Print intermediate results
    config = INPUT_SIZE_CONFIGS[config_name]
    print(
        f"{loss_name} [{config_name}: {config['num_residues']}r, {config['num_frames']}f] "
        f"[{jit_label}]: {timing_results['mean']:.3e}s ± {timing_results['std']:.3e}s"
    )

    # Debug: Print current results count
    total_results = sum(
        len(config_results) * len(jit_results)
        for config_results in PERFORMANCE_RESULTS.values()
        for jit_results in config_results.values()
    )
    print(f"  -> Total results collected so far: {total_results}")

    # Basic performance assertion
    assert timing_results["mean"] < 10.0  # Should complete in reasonable time


def create_performance_heatmaps(results_dict, output_dir="_legacy_lossAdapter_performance"):
    """Create heatmaps showing performance across different configurations."""
    if not HAS_PLOTTING_LIBS:
        print("Skipping heatmap generation - plotting libraries not available")
        return
    # save to directory next to this file
    output_dir = Path(__file__).parent / output_dir

    Path(output_dir).mkdir(exist_ok=True)

    # Prepare data for heatmaps
    loss_functions = list(results_dict.keys())
    configs = list(INPUT_SIZE_CONFIGS.keys())

    if not loss_functions or not configs:
        print("No data available for heatmap generation")
        return

    # Create separate heatmaps for JIT and non-JIT
    for jit_type in ["jit", "no_jit"]:
        # Create matrix for mean execution times
        mean_times = np.zeros((len(loss_functions), len(configs)))

        for i, loss_name in enumerate(loss_functions):
            for j, config_name in enumerate(configs):
                if (
                    config_name in results_dict[loss_name]
                    and jit_type in results_dict[loss_name][config_name]
                ):
                    mean_times[i, j] = results_dict[loss_name][config_name][jit_type]["mean"]
                else:
                    mean_times[i, j] = np.nan

        # Create heatmap
        plt.figure(figsize=(12, 20))
        mask = np.isnan(mean_times)

        # Use log scale for better visualization
        log_mean_times = np.log10(mean_times + 1e-8)  # Add small value to avoid log(0)

        sns.heatmap(
            log_mean_times,
            annot=mean_times,
            fmt=".3e",
            mask=mask,
            xticklabels=configs,
            yticklabels=loss_functions,
            cmap="viridis",
            cbar_kws={"label": "Log10(Execution Time [s])"},
        )

        plt.title(f"Loss Function Performance Heatmap ({jit_type.upper()})")
        plt.xlabel("Input Size Configuration")
        plt.ylabel("Loss Function")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        output_file = f"{output_dir}/performance_heatmap_{jit_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {output_file}")

    # Create speedup heatmap (JIT vs no-JIT)
    speedup_matrix = np.zeros((len(loss_functions), len(configs)))

    for i, loss_name in enumerate(loss_functions):
        for j, config_name in enumerate(configs):
            if (
                config_name in results_dict[loss_name]
                and "jit" in results_dict[loss_name][config_name]
                and "no_jit" in results_dict[loss_name][config_name]
            ):
                no_jit_time = results_dict[loss_name][config_name]["no_jit"]["mean"]
                jit_time = results_dict[loss_name][config_name]["jit"]["mean"]
                speedup_matrix[i, j] = no_jit_time / jit_time
            else:
                speedup_matrix[i, j] = np.nan

    plt.figure(figsize=(12, 20))
    mask = np.isnan(speedup_matrix)

    sns.heatmap(
        speedup_matrix,
        annot=True,
        fmt=".3e",
        mask=mask,
        xticklabels=configs,
        yticklabels=loss_functions,
        cmap="RdYlGn",
        center=1.0,
        cbar_kws={"label": "Speedup Factor (no-JIT / JIT)"},
    )

    plt.title("JIT Speedup Heatmap")
    plt.xlabel("Input Size Configuration")
    plt.ylabel("Loss Function")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_file = f"{output_dir}/speedup_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved speedup heatmap: {output_file}")


def create_performance_summary_table(results_dict, output_dir="_legacy_lossAdapter_performance"):
    """Create a detailed summary table of performance results."""
    output_dir = Path(__file__).parent / output_dir

    Path(output_dir).mkdir(exist_ok=True)

    # Prepare summary data
    summary_data = []

    for loss_name, config_results in results_dict.items():
        for config_name, jit_results in config_results.items():
            config = INPUT_SIZE_CONFIGS[config_name]

            row_data = {
                "Loss Function": loss_name,
                "Config": config_name,
                "Residues": config["num_residues"],
                "Timepoints": config["num_timepoints"],
                "Frames": config["num_frames"],
                "Input Size": config["num_residues"] * config["num_frames"],
            }

            for jit_type in ["no_jit", "jit"]:
                if jit_type in jit_results:
                    results = jit_results[jit_type]
                    row_data[f"{jit_type.replace('_', '-').title()} Mean (s)"] = results["mean"]
                    row_data[f"{jit_type.replace('_', '-').title()} Std (s)"] = results["std"]
                else:
                    row_data[f"{jit_type.replace('_', '-').title()} Mean (s)"] = np.nan
                    row_data[f"{jit_type.replace('_', '-').title()} Std (s)"] = np.nan

            # Calculate speedup if both available
            if "no_jit" in jit_results and "jit" in jit_results:
                speedup = jit_results["no_jit"]["mean"] / jit_results["jit"]["mean"]
                row_data["Speedup Factor"] = speedup
            else:
                row_data["Speedup Factor"] = np.nan

            summary_data.append(row_data)

    # Convert to DataFrame-like format and save as CSV
    if HAS_PLOTTING_LIBS:
        df = pd.DataFrame(summary_data)
        output_file = f"{output_dir}/performance_summary.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved performance summary: {output_file}")
        return df
    else:
        # Manual CSV writing if pandas not available
        if summary_data:
            output_file = f"{output_dir}/performance_summary.csv"
            with open(output_file, "w") as f:
                # Write header
                headers = list(summary_data[0].keys())
                f.write(",".join(headers) + "\n")

                # Write data
                for row in summary_data:
                    row_values = [str(row[h]) for h in headers]
                    f.write(",".join(row_values) + "\n")
            print(f"Saved performance summary: {output_file}")
        return summary_data


def finalize_performance_analysis():
    """Generate final performance analysis plots and summaries."""
    print("\nDEBUG: finalize_performance_analysis called")
    print(f"DEBUG: PERFORMANCE_RESULTS has {len(PERFORMANCE_RESULTS)} loss functions")

    if not PERFORMANCE_RESULTS:
        print("No performance results to analyze!")
        return

    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 80)

    # Create visualizations
    print("Creating performance heatmaps...")
    create_performance_heatmaps(PERFORMANCE_RESULTS)

    print("Creating performance summary table...")
    create_performance_summary_table(PERFORMANCE_RESULTS)

    # Print top-level summary statistics
    print(
        f"\nTested {len(PERFORMANCE_RESULTS)} loss functions across {len(INPUT_SIZE_CONFIGS)} size configurations"
    )
    print(f"Input size configurations: {list(INPUT_SIZE_CONFIGS.keys())}")

    # Find best and worst performing functions
    all_times = []
    for loss_name, config_results in PERFORMANCE_RESULTS.items():
        for config_name, jit_results in config_results.items():
            for jit_type, results in jit_results.items():
                all_times.append((loss_name, config_name, jit_type, results["mean"]))

    if all_times:
        all_times.sort(key=lambda x: x[3])  # Sort by time

        print(
            f"\nFastest execution: {all_times[0][0]} ({all_times[0][1]}, {all_times[0][2]}): {all_times[0][3]:.6f}s"
        )
        print(
            f"Slowest execution: {all_times[-1][0]} ({all_times[-1][1]}, {all_times[-1][2]}): {all_times[-1][3]:.6f}s"
        )

        # Calculate average speedup from JIT
        speedups = []
        for loss_name, config_results in PERFORMANCE_RESULTS.items():
            for config_name, jit_results in config_results.items():
                if "no_jit" in jit_results and "jit" in jit_results:
                    speedup = jit_results["no_jit"]["mean"] / jit_results["jit"]["mean"]
                    speedups.append(speedup)

        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"\nAverage JIT speedup: {avg_speedup:.2f}x")
            print(f"Max JIT speedup: {max(speedups):.2f}x")
            print(f"Min JIT speedup: {min(speedups):.2f}x")

    print("\nPerformance plots saved to: _legacy_lossAdapter_performance/")
    print("- performance_heatmap_jit.png")
    print("- performance_heatmap_no_jit.png")
    print("- speedup_heatmap.png")
    print("- performance_summary.csv")


# Manual trigger for generating performance analysis
def generate_performance_plots():
    """Manually trigger performance plot generation (call this after tests complete)."""
    finalize_performance_analysis()


# Standalone function to run performance analysis
if __name__ == "__main__":
    print("Running standalone performance analysis...")
    # Note: This will only work if PERFORMANCE_RESULTS has been populated by running tests first
    generate_performance_plots()


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, right before returning the exit status."""
    print(f"DEBUG: pytest_sessionfinish called with exitstatus {exitstatus}")
    finalize_performance_analysis()


# Simple test to trigger plot generation
def test_generate_final_performance_plots():
    """Final test that generates all performance plots and summaries."""
    # This test runs last and generates the final output
    print("\nFinal test - generating performance analysis...")
    generate_performance_plots()


# Helper function for jax_pairwise_cosine_similarity
def test_jax_pairwise_cosine_similarity():
    key = jax.random.PRNGKey(1)
    array = jax.random.normal(key, (5, 3))
    similarity_matrix = jax_pairwise_cosine_similarity(array)
    assert similarity_matrix.shape == (5, 5)
    assert jnp.all(similarity_matrix >= 0) and jnp.all(similarity_matrix <= 2)
    assert jnp.allclose(jnp.diag(similarity_matrix), 2.0)  # Self-similarity should be 1+1=2


# Standalone function to run performance analysis
if __name__ == "__main__":
    print("Running standalone performance analysis...")
    # Note: This will only work if PERFORMANCE_RESULTS has been populated by running tests first
    generate_performance_plots()
