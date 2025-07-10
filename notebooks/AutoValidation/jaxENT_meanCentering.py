import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


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


def plot_publication_ready_heatmaps(
    data: Dict[int, List[float]], timepoints: List[float], output_dir: str, noise_sd: float = None
):
    """
    Create publication-ready heatmaps of deuterium uptake
    with large labels and both raw and mean-centered data
    """
    # Convert to dataframe for easier processing
    df = pd.DataFrame(data).T
    df.columns = timepoints
    df.index.name = "Residue End"

    # Create a custom red-white-blue colormap for mean-centered data
    colors = [
        "#053061",
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#D1E5F0",
        "#FFFFFF",
        "#FDDBC7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
        "#67001F",
    ]
    cmap_name = "custom_diverging"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # ============= RAW UPTAKE HEATMAP =============
    plt.figure(figsize=(12, 10))

    # Create the heatmap with large font sizes
    ax = sns.heatmap(
        df,
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=timepoints,
        cbar_kws={"label": "Deuterium Uptake Fraction"},
    )

    # Set larger font sizes
    title_prefix = "Deuterium Uptake Fraction"
    if noise_sd is not None:
        title_prefix += f" (Noise SD = {noise_sd})"

    plt.title(title_prefix, fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Time (min)", fontsize=18, fontweight="bold", labelpad=15)
    plt.ylabel("Residue End Position", fontsize=18, fontweight="bold", labelpad=15)

    # Increase tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Adjust colorbar fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Deuterium Uptake Fraction", size=16, weight="bold", labelpad=15)

    # Save high resolution figure
    title_suffix = "_raw"
    if noise_sd is not None:
        title_suffix = f"_noise_sd_{noise_sd}_raw"

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"uptake_heatmap{title_suffix}.png"), dpi=600, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, f"uptake_heatmap{title_suffix}.pdf"), bbox_inches="tight")
    plt.close()

    # ============= MEAN-CENTERED HEATMAP =============
    # Calculate timepoint-wise mean across peptides
    time_means = df.mean(axis=0)

    # Subtract the mean of each timepoint from all peptides
    centered_df = df.copy()
    for col in centered_df.columns:
        centered_df[col] = centered_df[col] - time_means[col]

    # Find symmetrical vmin/vmax for diverging colormap
    abs_max = max(abs(centered_df.min().min()), abs(centered_df.max().max()))

    plt.figure(figsize=(12, 10))

    # Create the mean-centered heatmap with large font sizes
    ax = sns.heatmap(
        centered_df,
        cmap=cm,
        vmin=-abs_max,
        vmax=abs_max,
        xticklabels=timepoints,
        cbar_kws={"label": "Mean-Centered Deuterium Uptake"},
    )

    # Set larger font sizes
    title_prefix = "Mean-Centered Deuterium Uptake"
    if noise_sd is not None:
        title_prefix += f" (Noise SD = {noise_sd})"

    plt.title(title_prefix, fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Time (min)", fontsize=18, fontweight="bold", labelpad=15)
    plt.ylabel("Residue End Position", fontsize=18, fontweight="bold", labelpad=15)

    # Increase tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Adjust colorbar fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Deviation from Timepoint Mean", size=16, weight="bold", labelpad=15)

    # Save high resolution figure
    title_suffix = "_mean_centered"
    if noise_sd is not None:
        title_suffix = f"_noise_sd_{noise_sd}_mean_centered"

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"uptake_heatmap{title_suffix}.png"), dpi=600, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, f"uptake_heatmap{title_suffix}.pdf"), bbox_inches="tight")
    plt.close()


def main():
    # Define input file path and output directory
    input_file = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data/mixed_60-40_artificial_expt_resfracs.dat"
    input_file = (
        "/home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_segdfrac.dat"
    )

    output_dir = "hdx_publication_ready_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    # Define timepoints from the original script
    timepoints = [0.167, 1, 10, 60, 120]

    # Define noise levels to test (same as in original script)
    noise_sds = [0]

    # Read original data
    original_data = extract_hdx_data_from_file(input_file)
    print(f"Read data for {len(original_data)} peptides")

    # Create publication-ready heatmaps for the original data
    print("Generating publication-ready heatmaps for original data...")
    plot_publication_ready_heatmaps(original_data, timepoints, output_dir)

    # Process each noise level
    for noise_sd in noise_sds:
        if noise_sd == 0:  # Skip noise_sd=0 as it's identical to original
            continue

        print(f"Processing noise SD = {noise_sd}")

        # Add noise to data using the same method as the original script
        noisy_data, _ = add_noise_to_dfrac(original_data, noise_sd)

        # Create publication-ready heatmaps for the noisy data
        print(f"Generating publication-ready heatmaps for noise SD = {noise_sd}...")
        plot_publication_ready_heatmaps(noisy_data, timepoints, output_dir, noise_sd)

    print("Analysis complete. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
