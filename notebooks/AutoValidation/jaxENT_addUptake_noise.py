import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_hdx_data_from_file(file_path: str) -> Dict[int, List[float]]:
    """Read deuterium uptake data from file"""
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment="#",
        header=None,
    )

    result_dict = {}
    for _, row in df.iterrows():
        residue_end = int(row[1])  # Second column is residue_end
        times = row[2:7].tolist()  # Columns 2-6 are time values
        result_dict[residue_end] = times

    return result_dict


def add_noise_to_dfrac(
    original_data: Dict[int, List[float]], noise_sd: float, random_seed: int = 42
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Add stochastic noise to dfrac data proportional to uptake value

    Returns:
        Tuple containing:
        - Dictionary of noisy data
        - Dictionary of actual noise added
    """
    rng = np.random.RandomState(random_seed)
    noisy_data = {}
    noise_added = {}

    for residue_end, uptake_values in original_data.items():
        base_dfrac = np.asarray(uptake_values)

        # Scale noise with the magnitude of the uptake (same method as original script)
        # scaled_noise_sd = noise_sd * (base_dfrac) / 0.6
        scaled_noise_sd = noise_sd * np.ones_like(base_dfrac) / 1

        # Generate unique noise for each peptide and timepoint
        noise = rng.normal(0, scaled_noise_sd, size=base_dfrac.shape)
        noisy_dfrac = base_dfrac + noise

        # Ensure values stay within physical bounds (0 to 1 for deuterium fractions)
        noisy_dfrac = np.clip(noisy_dfrac, 0.0, 1.0)

        # Store the noisy data and actual noise added
        noisy_data[residue_end] = noisy_dfrac.tolist()
        noise_added[residue_end] = noise.tolist()

    return noisy_data, noise_added


def save_dfrac_data(data: Dict[int, List[float]], file_path: str):
    """Save deuterium uptake data to file in the same format as original"""
    with open(file_path, "w") as f:
        for residue_end, uptake_values in sorted(data.items()):
            values_str = " ".join([f"{val:.6f}" for val in uptake_values])
            f.write(f"0 {residue_end} {values_str}\n")


def plot_uptake_curves(
    original_data: Dict[int, List[float]],
    noisy_data: Dict[int, List[float]],
    timepoints: List[float],
    noise_sd: float,
    output_path: str,
):
    """Plot mean uptake curves for original and noisy data with standard deviation"""
    plt.figure(figsize=(12, 6))

    # Convert to dataframes for easier plotting
    original_df = pd.DataFrame(original_data).T
    original_df.columns = timepoints

    noisy_df = pd.DataFrame(noisy_data).T
    noisy_df.columns = timepoints

    # Calculate mean uptake at each timepoint
    original_mean = original_df.mean()
    noisy_mean = noisy_df.mean()

    # Calculate standard deviation
    original_std = original_df.std()
    noisy_std = noisy_df.std()

    # Plot mean uptake curves
    plt.plot(timepoints, original_mean, "b-", label="Original", linewidth=2)
    plt.plot(timepoints, noisy_mean, "r-", label=f"Noise SD = {noise_sd}", linewidth=2)

    # Add shaded regions for standard deviation
    plt.fill_between(
        timepoints,
        original_mean - original_std,
        original_mean + original_std,
        color="blue",
        alpha=0.2,
    )
    plt.fill_between(
        timepoints, noisy_mean - noisy_std, noisy_mean + noisy_std, color="red", alpha=0.2
    )

    plt.xlabel("Time (min)")
    plt.ylabel("Deuterium Uptake Fraction")
    plt.title(f"Mean Uptake Curves (Noise SD = {noise_sd})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")  # Use log scale for time which is common for HDX-MS data
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uptake_heatmap(
    data: Dict[int, List[float]], timepoints: List[float], title: str, output_path: str
):
    """Create heatmap of deuterium uptake across residues and timepoints"""
    df = pd.DataFrame(data).T
    df.columns = timepoints
    df.index.name = "Residue End"

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, cmap="viridis", vmin=0, vmax=1, xticklabels=timepoints)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Residue End Position")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sd_difference_heatmap(
    original_data: Dict[int, List[float]],
    noisy_data: Dict[int, List[float]],
    timepoints: List[float],
    noise_sd: float,
    output_path: str,
):
    """Create heatmap showing the absolute difference between original and noisy data"""
    original_df = pd.DataFrame(original_data).T
    original_df.columns = timepoints

    noisy_df = pd.DataFrame(noisy_data).T
    noisy_df.columns = timepoints

    # Calculate absolute difference
    diff_df = (noisy_df - original_df).abs()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(diff_df, cmap="Reds", vmin=0, vmax=noise_sd * 2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Residue End Position")
    ax.set_title(f"Absolute Difference (Noise SD = {noise_sd})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sd_boxplots(
    original_data: Dict[int, List[float]],
    all_noisy_data: Dict[float, Dict[int, List[float]]],
    timepoints: List[float],
    output_dir: str,
):
    """Create boxplots comparing SD distributions for different noise levels"""

    # Create dataframe for boxplot data
    data_for_boxplot = []

    # Calculate SD across timepoints for each peptide (peptide axis)
    original_df = pd.DataFrame(original_data).T
    original_df.columns = timepoints
    original_sd_by_peptide = original_df.std(axis=1)

    for sd_val in original_sd_by_peptide:
        data_for_boxplot.append(
            {"Standard Deviation": sd_val, "Dataset": "Original", "Direction": "By Peptide"}
        )

    # Calculate noisy data SDs by peptide
    for noise_sd, noisy_data in sorted(all_noisy_data.items()):
        if noise_sd == 0:  # Skip noise_sd=0 as it's identical to original
            continue

        noisy_df = pd.DataFrame(noisy_data).T
        noisy_df.columns = timepoints
        noisy_sd_by_peptide = noisy_df.std(axis=1)

        for sd_val in noisy_sd_by_peptide:
            data_for_boxplot.append(
                {
                    "Standard Deviation": sd_val,
                    "Dataset": f"SD={noise_sd}",
                    "Direction": "By Peptide",
                }
            )

    # Calculate SD across peptides for each timepoint (timepoint axis)
    original_sd_by_timepoint = original_df.std(axis=0)

    for sd_val in original_sd_by_timepoint:
        data_for_boxplot.append(
            {"Standard Deviation": sd_val, "Dataset": "Original", "Direction": "By Timepoint"}
        )

    for noise_sd, noisy_data in sorted(all_noisy_data.items()):
        if noise_sd == 0:  # Skip noise_sd=0 as it's identical to original
            continue

        noisy_df = pd.DataFrame(noisy_data).T
        noisy_df.columns = timepoints
        noisy_sd_by_timepoint = noisy_df.std(axis=0)

        for sd_val in noisy_sd_by_timepoint:
            data_for_boxplot.append(
                {
                    "Standard Deviation": sd_val,
                    "Dataset": f"SD={noise_sd}",
                    "Direction": "By Timepoint",
                }
            )

    # Create dataframe for plotting
    boxplot_df = pd.DataFrame(data_for_boxplot)

    # Plot SD distribution by peptide
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x="Dataset",
        y="Standard Deviation",
        data=boxplot_df[boxplot_df["Direction"] == "By Peptide"],
        order=["Original"] + [f"SD={sd}" for sd in sorted(all_noisy_data.keys()) if sd > 0],
    )
    plt.title("Standard Deviation Distribution (Across Timepoints for Each Peptide)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "sd_boxplots_by_peptide.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot SD distribution by timepoint
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x="Dataset",
        y="Standard Deviation",
        data=boxplot_df[boxplot_df["Direction"] == "By Timepoint"],
        order=["Original"] + [f"SD={sd}" for sd in sorted(all_noisy_data.keys()) if sd > 0],
    )
    plt.title("Standard Deviation Distribution (Across Peptides for Each Timepoint)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "sd_boxplots_by_timepoint.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_sd_by_peptide_comparison(
    original_data: Dict[int, List[float]],
    all_noisy_data: Dict[float, Dict[int, List[float]]],
    timepoints: List[float],
    output_dir: str,
):
    """Plot SD for each peptide across timepoints for different noise levels"""
    # Get residue numbers
    residues = sorted(original_data.keys())

    # Calculate SD for original data across timepoints
    original_df = pd.DataFrame(original_data).T
    original_df.columns = timepoints
    original_sd = original_df.std(axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(residues, original_sd, "k-", label="Original", linewidth=2)

    # Add lines for each noise level
    noise_levels = sorted([sd for sd in all_noisy_data.keys() if sd > 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels)))

    for i, noise_sd in enumerate(noise_levels):
        noisy_data = all_noisy_data[noise_sd]
        noisy_df = pd.DataFrame(noisy_data).T
        noisy_df.columns = timepoints
        noisy_sd_values = noisy_df.std(axis=1)

        plt.plot(residues, noisy_sd_values, color=colors[i], label=f"SD={noise_sd}", alpha=0.7)

    plt.xlabel("Residue End Position")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation by Peptide for Different Noise Levels")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "sd_by_peptide_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_actual_vs_target_noise(
    all_noise_added: Dict[float, Dict[int, List[float]]], output_dir: str
):
    """Plot actual noise SD against target noise SD"""
    actual_sds = []
    target_sds = []

    for target_sd, noise_data in sorted(all_noise_added.items()):
        noise_df = pd.DataFrame(noise_data).T

        # Calculate actual SD of added noise
        actual_sd = noise_df.values.std()
        actual_sds.append(actual_sd)
        target_sds.append(target_sd)

    plt.figure(figsize=(8, 6))
    plt.scatter(target_sds, actual_sds, s=80)

    # Add labels for each point
    for i, target_sd in enumerate(target_sds):
        plt.annotate(
            f"SD={target_sd}",
            (target_sds[i], actual_sds[i]),
            xytext=(10, 5),
            textcoords="offset points",
        )

    # Add line of equality
    max_val = max(max(target_sds), max(actual_sds))
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

    plt.xlabel("Target Noise SD")
    plt.ylabel("Actual Noise SD")
    plt.title("Actual vs Target Noise Standard Deviation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "actual_vs_target_noise.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    # Define input file path and output directory
    input_file = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data/mixed_60-40_artificial_expt_resfracs.dat"
    output_dir = "hdx_noise_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Define timepoints from the original script
    timepoints = [0.167, 1, 10, 60, 120]

    # Define noise levels to test (same as in original script)
    noise_sds = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Read original data
    original_data = extract_hdx_data_from_file(input_file)
    print(f"Read data for {len(original_data)} peptides")

    # Plot original data heatmap
    plot_uptake_heatmap(
        original_data,
        timepoints,
        "Original Uptake Heatmap",
        os.path.join(output_dir, "original_heatmap.png"),
    )

    # Store all noisy data and added noise
    all_noisy_data = {}
    all_noise_added = {}

    # Process each noise level
    for noise_sd in noise_sds:
        if noise_sd == 0:  # Skip noise_sd=0 as it's identical to original
            all_noisy_data[noise_sd] = original_data
            continue

        print(f"Processing noise SD = {noise_sd}")

        # Add noise to data using the same method as the original script
        noisy_data, noise_added = add_noise_to_dfrac(original_data, noise_sd)
        all_noisy_data[noise_sd] = noisy_data
        all_noise_added[noise_sd] = noise_added

        # Save noisy data in the same format as original
        noise_output_file = os.path.join(output_dir, f"noisy_dfrac_sd_{noise_sd}.dat")
        save_dfrac_data(noisy_data, noise_output_file)
        print(f"Saved noisy data to {noise_output_file}")

        # Plot uptake curves with error bands
        plot_uptake_curves(
            original_data,
            noisy_data,
            timepoints,
            noise_sd,
            os.path.join(output_dir, f"uptake_curves_sd_{noise_sd}.png"),
        )

        # Plot heatmap for this noise level
        plot_uptake_heatmap(
            noisy_data,
            timepoints,
            f"Noisy Uptake Heatmap (SD = {noise_sd})",
            os.path.join(output_dir, f"noisy_heatmap_sd_{noise_sd}.png"),
        )

        # Plot difference heatmap to visualize noise impact
        plot_sd_difference_heatmap(
            original_data,
            noisy_data,
            timepoints,
            noise_sd,
            os.path.join(output_dir, f"difference_heatmap_sd_{noise_sd}.png"),
        )

    # Create summary plots comparing all noise levels
    plot_sd_boxplots(original_data, all_noisy_data, timepoints, output_dir)
    plot_actual_vs_target_noise(all_noise_added, output_dir)
    plot_sd_by_peptide_comparison(original_data, all_noisy_data, timepoints, output_dir)

    print("Analysis complete. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
