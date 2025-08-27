"""
This script contains a small set of experiments for testing projections and normalisation functions.
There are 3 main experiments:
1. Testing the projection function on a vector that already exists on the probability simplex.
2. Testing the projection function on a vector of natural numbers.
3. Testing the projections functions on a vector of random real numbers.
Projection functions:
 - optax.projections.projection_simplex
 - simple rescaling plus epsilon (1e-8)
Plots:
- Histograms original vs projected vectors
- Summary statistics original vs projected vectors
 - min, max, mean, std, sum
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax.projections
from scipy.stats import pearsonr

# Set up output directory
output_dir = os.path.dirname(__file__) + os.path.basename(__file__).replace(".py", "_outputs/")
os.makedirs(output_dir, exist_ok=True)

# Configuration
SEED = 42
VECTOR_SIZE = 100
EPSILON = 1e-30
NUM_RUNS = 25  # Number of experiment runs


@dataclass
class VectorStats:
    """Container for vector statistics."""

    min: float
    max: float
    mean: float
    std: float
    sum: float

    def to_dict(self):
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "sum": self.sum,
        }


@dataclass
class AggregatedStats:
    """Container for aggregated statistics across multiple runs."""

    min_mean: float
    min_std: float
    max_mean: float
    max_std: float
    mean_mean: float
    mean_std: float
    std_mean: float
    std_std: float
    sum_mean: float
    sum_std: float

    # Additional metrics
    simplex_validity_rate: float = 0.0
    l2_distance_mean: float = 0.0
    l2_distance_std: float = 0.0
    correlation_mean: float = 0.0
    correlation_std: float = 0.0
    kl_divergence_mean: float = 0.0
    kl_divergence_std: float = 0.0


def compute_stats(vec: jnp.ndarray) -> VectorStats:
    """Compute summary statistics for a vector."""
    return VectorStats(
        min=float(jnp.min(vec)),
        max=float(jnp.max(vec)),
        mean=float(jnp.mean(vec)),
        std=float(jnp.std(vec)),
        sum=float(jnp.sum(vec)),
    )


def rescale_projection(vec: jnp.ndarray, epsilon: float = EPSILON) -> jnp.ndarray:
    """Simple rescaling projection with epsilon for numerical stability."""
    # Add epsilon to avoid division by zero
    vec_positive = jnp.maximum(vec, 0)  # Ensure non-negative
    vec_sum = jnp.sum(vec_positive)
    if vec_sum > epsilon:
        return vec_positive / vec_sum
    else:
        raise ValueError("Cannot normalize vector with near-zero L1 norm.")


def softmax_projection(vec: jnp.ndarray) -> jnp.ndarray:
    """Softmax projection to probability simplex."""
    return jax.nn.softmax(vec)


def l1_normalization(vec: jnp.ndarray, epsilon: float = EPSILON) -> jnp.ndarray:
    """L1 normalization with epsilon for stability."""
    vec_abs = jnp.abs(vec)
    vec_sum = jnp.sum(vec_abs)
    if vec_sum > epsilon:
        return vec_abs / vec_sum
    else:
        raise ValueError("Cannot normalize vector with near-zero L1 norm.")


def optax_simplex_projection(vec: jnp.ndarray) -> jnp.ndarray:
    """Wrapper for optax simplex projection."""
    return optax.projections.projection_simplex(vec)


def optax_safe_simplex_projection(vec: jnp.ndarray) -> jnp.ndarray:
    """Wrapper for optax simplex projection."""
    vec = jnp.abs(vec)
    return optax.projections.projection_simplex(vec)


def optax_L1_ball_projection(vec: jnp.ndarray, radius: float = 1.0) -> jnp.ndarray:
    """Wrapper for optax L1 ball projection."""
    vec = jnp.abs(vec)

    return optax.projections.projection_l1_ball(vec, radius)


def optax_l2_ball_projection(vec: jnp.ndarray, radius: float = 1.0) -> jnp.ndarray:
    """Wrapper for optax L2 ball projection."""
    vec = jnp.abs(vec)
    return optax.projections.projection_l2_ball(vec, radius)


# Dictionary of projection functions
PROJECTION_FUNCTIONS: Dict[str, Callable] = {
    "optax_l2_ball": optax_l2_ball_projection,
    "optax_L1_ball": optax_L1_ball_projection,
    "optax_safe_simplex": optax_safe_simplex_projection,
    "optax_simplex": optax_simplex_projection,
    "rescale": rescale_projection,
    "softmax": softmax_projection,
    "l1_norm": l1_normalization,
}


def generate_test_vectors(key: jax.random.PRNGKey, size: int) -> Dict[str, jnp.ndarray]:
    """Generate test vectors for experiments."""
    key1, key2, key3 = jax.random.split(key, 3)

    vectors = {}

    # 1. Vector already on probability simplex
    simplex_vec = jax.random.dirichlet(key1, jnp.ones(size))
    vectors["simplex"] = simplex_vec

    # 2. Vector of natural numbers (integers from 1 to 100)
    natural_vec = jax.random.randint(key2, shape=(size,), minval=1, maxval=101)
    vectors["natural"] = natural_vec.astype(jnp.float32)

    # 3. Vector of random real numbers (including negative)
    random_vec = jax.random.normal(key3, shape=(size,)) * 10
    vectors["random"] = random_vec

    return vectors


def validate_simplex_projection(vec: jnp.ndarray, tolerance: float = 1e-4) -> bool:
    """Validate if a vector is on the probability simplex."""
    is_non_negative = jnp.all(vec >= 0)
    sum_to_one = jnp.abs(jnp.sum(vec) - 1.0) < tolerance
    return bool(is_non_negative and sum_to_one)


def aggregate_statistics(stats_list: List[VectorStats]) -> AggregatedStats:
    """Aggregate statistics across multiple runs."""
    stats_array = np.array([[s.min, s.max, s.mean, s.std, s.sum] for s in stats_list])

    return AggregatedStats(
        min_mean=np.mean(stats_array[:, 0]),
        min_std=np.std(stats_array[:, 0]),
        max_mean=np.mean(stats_array[:, 1]),
        max_std=np.std(stats_array[:, 1]),
        mean_mean=np.mean(stats_array[:, 2]),
        mean_std=np.std(stats_array[:, 2]),
        std_mean=np.mean(stats_array[:, 3]),
        std_std=np.std(stats_array[:, 3]),
        sum_mean=np.mean(stats_array[:, 4]),
        sum_std=np.std(stats_array[:, 4]),
    )


def compute_correlation(vec1: jnp.ndarray, vec2: jnp.ndarray) -> float:
    """Compute Pearson correlation coefficient between two vectors."""
    # Convert to numpy for scipy
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    # Handle edge cases
    if np.std(vec1_np) == 0 or np.std(vec2_np) == 0:
        return 0.0

    corr, _ = pearsonr(vec1_np, vec2_np)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_kl_divergence(p: jnp.ndarray, q: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence KL(p||q) between two probability distributions."""
    # Add epsilon to avoid log(0) or division by zero
    p = p + epsilon
    q = p + epsilon
    # Ensure both are probability distributions (sum to 1)
    p = p / jnp.sum(p)

    q = q / jnp.sum(q)

    # Compute KL divergence
    # KL(p||q) = sum(p * log(p/q))
    kl_div = jnp.sum(p * jnp.log(p / q))
    return float(kl_div)


def run_single_experiment(
    vector: jnp.ndarray, projection_funcs: Dict[str, Callable]
) -> Dict[str, any]:
    """Run a single experiment iteration with all projection functions."""
    results = {
        "original_stats": compute_stats(vector),
        "proj_stats": {},
        "simplex_validity": {},
        "l2_distances": {},
        "correlations": {},
        "kl_divergences": {},
    }

    for proj_name, proj_func in projection_funcs.items():
        projected = proj_func(vector)
        results["proj_stats"][proj_name] = compute_stats(projected)
        results["simplex_validity"][proj_name] = validate_simplex_projection(projected)

        # rescale to probability simplex using l1 normalization for distance calculation
        normalised_vector = l1_normalization(vector)
        results["l2_distances"][proj_name] = float(jnp.linalg.norm(normalised_vector - projected))
        results["correlations"][proj_name] = compute_correlation(normalised_vector, projected)
        results["kl_divergences"][proj_name] = compute_kl_divergence(normalised_vector, projected)

    return results


def run_multiple_experiments(
    experiment_name: str,
    vector_generator: Callable,
    projection_funcs: Dict[str, Callable],
    num_runs: int,
    base_key: jax.random.PRNGKey,
) -> Dict:
    """Run multiple experiment iterations and aggregate results."""
    print(f"\nRunning {experiment_name} ({num_runs} iterations)...")

    all_results = []
    keys = jax.random.split(base_key, num_runs)

    for i, key in enumerate(keys):
        vector = vector_generator(key)
        results = run_single_experiment(vector, projection_funcs)
        all_results.append(results)

    # Aggregate results
    aggregated = {
        "experiment_name": experiment_name,
        "num_runs": num_runs,
        "original": {},
        "projections": {},
    }

    # Aggregate original stats
    original_stats = [r["original_stats"] for r in all_results]
    aggregated["original"] = aggregate_statistics(original_stats)

    # Aggregate projection stats
    for proj_name in projection_funcs.keys():
        proj_stats = [r["proj_stats"][proj_name] for r in all_results]
        validity_rates = [r["simplex_validity"][proj_name] for r in all_results]
        l2_distances = [r["l2_distances"][proj_name] for r in all_results]
        correlations = [r["correlations"][proj_name] for r in all_results]
        kl_divergences = [r["kl_divergences"][proj_name] for r in all_results]

        agg_stats = aggregate_statistics(proj_stats)
        agg_stats.simplex_validity_rate = np.mean(validity_rates) * 100
        agg_stats.l2_distance_mean = np.mean(l2_distances)
        agg_stats.l2_distance_std = np.std(l2_distances)
        agg_stats.correlation_mean = np.mean(correlations)
        agg_stats.correlation_std = np.std(correlations)
        agg_stats.kl_divergence_mean = np.mean(kl_divergences)
        agg_stats.kl_divergence_std = np.std(kl_divergences)

        aggregated["projections"][proj_name] = agg_stats

    # Store raw results for visualization
    aggregated["raw_results"] = all_results

    return aggregated


def plot_aggregated_statistics(aggregated_results: Dict[str, Dict], output_path: str):
    """Plot aggregated statistics with error bars."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    metrics = ["min", "max", "mean", "std", "sum", "l2_distance", "correlation", "kl_divergence"]
    metric_labels = [
        "Min Value",
        "Max Value",
        "Mean Value",
        "Std Dev",
        "Sum",
        "L2 Distance from Original",
        "Correlation with Original",
        "KL Divergence from Original",
    ]

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        data_to_plot = []
        error_bars = []
        labels = []
        colors = []

        for exp_name, exp_data in aggregated_results.items():
            # Original (except for distance metrics)
            if metric not in ["l2_distance", "correlation", "kl_divergence"]:
                orig_stats = exp_data["original"]
                data_to_plot.append(getattr(orig_stats, f"{metric}_mean"))
                error_bars.append(getattr(orig_stats, f"{metric}_std"))
                labels.append(f"{exp_name}\n(original)")
                colors.append("blue")

            # Projections
            for proj_name, proj_stats in exp_data["projections"].items():
                if metric in ["l2_distance", "correlation", "kl_divergence"]:
                    data_to_plot.append(getattr(proj_stats, f"{metric}_mean"))
                    error_bars.append(getattr(proj_stats, f"{metric}_std"))
                else:
                    data_to_plot.append(getattr(proj_stats, f"{metric}_mean"))
                    error_bars.append(getattr(proj_stats, f"{metric}_std"))
                labels.append(f"{exp_name}\n({proj_name})")
                colors.append(
                    "green"
                    if proj_name == "optax_simplex"
                    else "red"
                    if proj_name == "rescale"
                    else "purple"
                    if proj_name == "softmax"
                    else "orange"
                )

        x_pos = np.arange(len(labels))
        bars = ax.bar(
            x_pos,
            data_to_plot,
            yerr=error_bars,
            capsize=3,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_xlabel("Experiment / Method", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} Across Experiments", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Aggregated Statistics Across All Experiments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_validity_rates(aggregated_results: Dict[str, Dict], output_path: str):
    """Plot simplex validity rates for each method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = list(aggregated_results.keys())
    methods = list(next(iter(aggregated_results.values()))["projections"].keys())

    x = np.arange(len(experiments))
    width = 0.2

    colors = ["green", "red", "purple", "orange"]
    for i, method in enumerate(methods):
        validity_rates = [
            aggregated_results[exp]["projections"][method].simplex_validity_rate
            for exp in experiments
        ]
        ax.bar(
            x + i * width,
            validity_rates,
            width,
            label=method,
            color=colors[i % len(colors)],
            alpha=0.7,
            edgecolor="black",
        )

    ax.set_xlabel("Experiment Type", fontsize=12)
    ax.set_ylabel("Simplex Validity Rate (%)", fontsize=12)
    ax.set_title(
        "Probability Simplex Validity Rates Across Methods", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 105])

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution_samples(
    aggregated_results: Dict[str, Dict], output_path: str, sample_runs: int = 5
):
    """Plot sample distributions from multiple runs."""
    fig, axes = plt.subplots(3, len(PROJECTION_FUNCTIONS) + 1, figsize=(20, 12))

    experiments = ["simplex", "natural", "random"]
    exp_titles = ["Simplex Vector", "Natural Numbers", "Random Real Numbers"]

    for exp_idx, (exp_key, exp_title) in enumerate(zip(experiments, exp_titles)):
        raw_results = aggregated_results[exp_key]["raw_results"][:sample_runs]

        # Original distributions
        ax = axes[exp_idx, 0]
        for run_idx, result in enumerate(raw_results):
            # Generate vector for this run (we need to regenerate since we don't store them)
            key = jax.random.PRNGKey(SEED + run_idx)
            vectors = generate_test_vectors(key, VECTOR_SIZE)
            vector = vectors[exp_key]

            ax.hist(np.array(vector), bins=30, alpha=0.3, label=f"Run {run_idx + 1}")

        ax.set_title(f"{exp_title}\nOriginal", fontsize=10, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Projected distributions
        for proj_idx, proj_name in enumerate(PROJECTION_FUNCTIONS.keys()):
            ax = axes[exp_idx, proj_idx + 1]

            for run_idx, result in enumerate(raw_results):
                key = jax.random.PRNGKey(SEED + run_idx)
                vectors = generate_test_vectors(key, VECTOR_SIZE)
                vector = vectors[exp_key]
                projected = PROJECTION_FUNCTIONS[proj_name](vector)

                ax.hist(np.array(projected), bins=30, alpha=0.3, label=f"Run {run_idx + 1}")

            ax.set_title(f"{exp_title}\n{proj_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Distribution Samples ({sample_runs} runs per experiment)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_detailed_report(aggregated_results: Dict[str, Dict], output_path: str):
    """Create a detailed text report of all results."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PROJECTION FUNCTIONS EXPERIMENT REPORT\n")
        f.write(f"Number of runs per experiment: {NUM_RUNS}\n")
        f.write(f"Vector size: {VECTOR_SIZE}\n")
        f.write("=" * 80 + "\n\n")

        for exp_name, exp_data in aggregated_results.items():
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Experiment: {exp_name.upper()}\n")
            f.write(f"{'=' * 60}\n\n")

            # Original stats
            orig = exp_data["original"]
            f.write("Original Vector Statistics (averaged over runs):\n")
            f.write(f"  Min:  {orig.min_mean:.6e} ± {orig.min_std:.6e}\n")
            f.write(f"  Max:  {orig.max_mean:.6e} ± {orig.max_std:.6e}\n")
            f.write(f"  Mean: {orig.mean_mean:.6e} ± {orig.mean_std:.6e}\n")
            f.write(f"  Std:  {orig.std_mean:.6e} ± {orig.std_std:.6e}\n")
            f.write(f"  Sum:  {orig.sum_mean:.6e} ± {orig.sum_std:.6e}\n\n")

            # Projection stats
            for proj_name, proj_stats in exp_data["projections"].items():
                f.write(f"\n{proj_name} Projection:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Min:  {proj_stats.min_mean:.6e} ± {proj_stats.min_std:.6e}\n")
                f.write(f"  Max:  {proj_stats.max_mean:.6e} ± {proj_stats.max_std:.6e}\n")
                f.write(f"  Mean: {proj_stats.mean_mean:.6e} ± {proj_stats.mean_std:.6e}\n")
                f.write(f"  Std:  {proj_stats.std_mean:.6e} ± {proj_stats.std_std:.6e}\n")
                f.write(f"  Sum:  {proj_stats.sum_mean:.6e} ± {proj_stats.sum_std:.6e}\n")
                f.write(f"  Simplex Validity Rate: {proj_stats.simplex_validity_rate:.2e}%\n")
                f.write(
                    f"  L2 Distance from Original: {proj_stats.l2_distance_mean:.6e} ± {proj_stats.l2_distance_std:.6e}\n"
                )
                f.write(
                    f"  Correlation with Original: {proj_stats.correlation_mean:.6e} ± {proj_stats.correlation_std:.6e}\n"
                )
                f.write(
                    f"  KL Divergence from Original: {proj_stats.kl_divergence_mean:.6e} ± {proj_stats.kl_divergence_std:.6e}\n"
                )

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write("Simplex Validity Rates:\n")
        f.write("-" * 40 + "\n")
        for exp_name in aggregated_results.keys():
            f.write(f"\n{exp_name}:\n")
            for proj_name, proj_stats in aggregated_results[exp_name]["projections"].items():
                f.write(f"  {proj_name:15s}: {proj_stats.simplex_validity_rate:6.2e}%\n")

        f.write("\n\nMean L2 Distances from Original:\n")
        f.write("-" * 40 + "\n")
        for exp_name in aggregated_results.keys():
            f.write(f"\n{exp_name}:\n")
            for proj_name, proj_stats in aggregated_results[exp_name]["projections"].items():
                f.write(f"  {proj_name:15s}: {proj_stats.l2_distance_mean:.6e}\n")

        f.write("\n\nMean Correlations with Original:\n")
        f.write("-" * 40 + "\n")
        for exp_name in aggregated_results.keys():
            f.write(f"\n{exp_name}:\n")
            for proj_name, proj_stats in aggregated_results[exp_name]["projections"].items():
                f.write(f"  {proj_name:15s}: {proj_stats.correlation_mean:.6e}\n")

        f.write("\n\nMean KL Divergences from Original:\n")
        f.write("-" * 40 + "\n")
        for exp_name in aggregated_results.keys():
            f.write(f"\n{exp_name}:\n")
            for proj_name, proj_stats in aggregated_results[exp_name]["projections"].items():
                f.write(f"  {proj_name:15s}: {proj_stats.kl_divergence_mean:.6e}\n")


def main():
    """Main execution function."""
    print("=" * 80)
    print("PROJECTION FUNCTION EXPERIMENTS WITH MULTIPLE RUNS")
    print(f"Configuration: {NUM_RUNS} runs per experiment, vector size = {VECTOR_SIZE}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Initialize random key
    base_key = jax.random.PRNGKey(SEED)

    # Define experiment configurations
    experiments = {
        "simplex": (
            "Simplex Vector",
            lambda key: generate_test_vectors(key, VECTOR_SIZE)["simplex"],
        ),
        "natural": (
            "Natural Numbers",
            lambda key: generate_test_vectors(key, VECTOR_SIZE)["natural"],
        ),
        "random": (
            "Random Real Numbers",
            lambda key: generate_test_vectors(key, VECTOR_SIZE)["random"],
        ),
    }

    # Run all experiments
    aggregated_results = {}

    # Split the base key for each experiment
    experiment_keys = jax.random.split(base_key, len(experiments))

    for (exp_key, (exp_name, vector_gen)), key in zip(experiments.items(), experiment_keys):
        results = run_multiple_experiments(
            exp_name, vector_gen, PROJECTION_FUNCTIONS, NUM_RUNS, key
        )
        aggregated_results[exp_key] = results

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Aggregated statistics with error bars
    stats_plot_path = os.path.join(output_dir, "aggregated_statistics.png")
    plot_aggregated_statistics(aggregated_results, stats_plot_path)
    print(f"  Aggregated statistics plot saved to: {stats_plot_path}")

    # 2. Validity rates comparison
    validity_plot_path = os.path.join(output_dir, "validity_rates.png")
    plot_validity_rates(aggregated_results, validity_plot_path)
    print(f"  Validity rates plot saved to: {validity_plot_path}")

    # 3. Sample distributions
    dist_plot_path = os.path.join(output_dir, "distribution_samples.png")
    plot_distribution_samples(aggregated_results, dist_plot_path)
    print(f"  Distribution samples plot saved to: {dist_plot_path}")

    # 4. Detailed text report
    report_path = os.path.join(output_dir, "detailed_report.txt")
    create_detailed_report(aggregated_results, report_path)
    print(f"  Detailed report saved to: {report_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    for exp_name, exp_data in aggregated_results.items():
        print(f"\n{exp_name.upper()} Experiment:")
        print("-" * 40)
        print("Simplex Validity Rates:")
        for proj_name, proj_stats in exp_data["projections"].items():
            print(f"  {proj_name:15s}: {proj_stats.simplex_validity_rate:6.2e}%")

        print("\nMean L2 Distance from Original:")
        for proj_name, proj_stats in exp_data["projections"].items():
            print(
                f"  {proj_name:15s}: {proj_stats.l2_distance_mean:.6e} ± {proj_stats.l2_distance_std:.6e}"
            )

        print("\nMean Correlation with Original:")
        for proj_name, proj_stats in exp_data["projections"].items():
            print(
                f"  {proj_name:15s}: {proj_stats.correlation_mean:.6e} ± {proj_stats.correlation_std:.6e}"
            )

        print("\nMean KL Divergence from Original:")
        for proj_name, proj_stats in exp_data["projections"].items():
            print(
                f"  {proj_name:15s}: {proj_stats.kl_divergence_mean:.6e} ± {proj_stats.kl_divergence_std:.6e}"
            )

    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
