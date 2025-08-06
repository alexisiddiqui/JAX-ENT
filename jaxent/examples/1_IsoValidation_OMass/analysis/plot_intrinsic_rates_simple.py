"""
This script is to plot intrinsic rates

HDXer = /home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/TeaA/bi_tri_modal/data/TeaA_ISO_bi/Benchmark/RW_bench/TeaA_ISO_bi_RW_bench_R3_k_sequence/train_TeaA_ISO_bi_1/out__train_TeaA_ISO_bi_1Intrinsic_rates.dat

# ResID  Intrinsic rate / min^-1
      2    163227.81828442




jaxENT = /home/alexi/Documents/JAX-ENT/jaxent/examples/1_IsoValidation/fitting/jaxENT/_featurise/features_iso_bi.npz

jnp.savez(
    features_path,
    heavy_contacts=features.heavy_contacts,
    acceptor_contacts=features.acceptor_contacts,
    k_ints=features.k_ints,
)
Partial_Topology.save_list_to_json(
    feature_topology,
    topology_save_path,
    )


HDXer rates are aligned by the resIDs in the .dat file while jaxENT rates are aligned by the topology.


This script plots the intrinisic rates for each residue in the protein, comparing the rates from HDXer and jaxENT - saving the plots to a file inside "_intrinsic_rates/".

It then computes some summary statistics on the differences between the two sets of rates.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Assuming Partial_Topology is importable from jaxent.src.interfaces.topology
import jaxent.src.interfaces.topology as pt
from jaxent.src.models.HDX.BV.features import BV_input_features


def plot_intrinsic_rates():
    """
    This script plots the intrinsic rates for each residue in the protein,
    comparing the rates from HDXer and jaxENT, and computes summary statistics.
    """

    # 1. Define file paths
    # hdxer_rates_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/TeaA/bi_tri_modal/data/TeaA_ISO_bi/Benchmark/RW_bench/TeaA_ISO_bi_RW_bench_R3_k_sequence/train_TeaA_ISO_bi_1/out__train_TeaA_ISO_bi_1Intrinsic_rates.dat"
    hdxer_kint_path = "../data/out__train_TeaA_auto_VAL_1Intrinsic_rates.dat"
    hdxer_rates_path = os.path.join(os.path.dirname(__file__), hdxer_kint_path)
    jaxent_features_path = "../fitting/jaxENT/_featurise/features_iso_bi.npz"
    jaxent_topology_path = "../fitting/jaxENT/_featurise/topology_iso_bi.json"
    jaxent_features_path = os.path.join(os.path.dirname(__file__), jaxent_features_path)
    jaxent_topology_path = os.path.join(os.path.dirname(__file__), jaxent_topology_path)

    output_dir = os.path.join(os.path.dirname(__file__), "_intrinsic_rates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "intrinsic_rates_comparison.png"
    stats_path = output_dir / "intrinsic_rates_comparison_stats.txt"

    print(f"Loading HDXer rates from: {hdxer_rates_path}")
    # 2. Load HDXer data
    hdxer_rates = {}
    try:
        with open(hdxer_rates_path, "r") as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            resid = int(parts[0])
                            rate = float(parts[1])
                            hdxer_rates[resid] = rate
                        except ValueError:
                            print(f"Skipping malformed line in HDXer data: {line.strip()}")
        print(f"Loaded {len(hdxer_rates)} HDXer rates.")
    except FileNotFoundError:
        print(f"Error: HDXer rates file not found at {hdxer_rates_path}")
        return
    except Exception as e:
        print(f"Error reading HDXer rates file: {e}")
        return

    print(f"Loaded {len(hdxer_rates)} HDXer rates.")

    print(f"Loading jaxENT features from: {jaxent_features_path}")
    print(f"Loading jaxENT topology from: {jaxent_topology_path}")
    # 3. Load jaxENT data
    jaxent_k_ints = None
    try:
        data = BV_input_features.load(jaxent_features_path)

        jaxent_k_ints = data.k_ints
        jaxENT_heavy_contacts = data.heavy_contacts
        jaxENT_acceptor_contacts = data.acceptor_contacts

        print(f"Loaded jaxENT k_ints with shape: {jaxent_k_ints.shape}")
        print(
            f"jaxENT features shape: heavy_contacts={jaxENT_heavy_contacts.shape}, acceptor_contacts={jaxENT_acceptor_contacts.shape}"
        )
        # breakpoint()
    except FileNotFoundError:
        print(f"Error: jaxENT features file not found at {jaxent_features_path}")
        return
    except Exception as e:
        print(f"Error loading jaxENT features: {e}")
        return

    jaxent_topologies = None
    try:
        jaxent_topologies = pt.PTSerialiser.load_list_from_json(jaxent_topology_path)
        print(f"Loaded {len(jaxent_topologies)} jaxENT topologies.")
    except FileNotFoundError:
        print(f"Error: jaxENT topology file not found at {jaxent_topology_path}")
        return
    except Exception as e:
        print(f"Error loading jaxENT topology: {e}")
        return

    if jaxent_k_ints is None or jaxent_topologies is None:
        print("Failed to load all jaxENT data. Exiting.")
        return

    # Create a mapping from residue ID to k_int from jaxENT data
    jaxent_rates = {}
    if len(jaxent_k_ints) != len(jaxent_topologies):
        print(
            "Warning: Number of k_ints does not match number of topologies. This may lead to incorrect mapping."
        )

    for i, topo in enumerate(jaxent_topologies):
        if i < len(jaxent_k_ints):
            # Assuming each topology corresponds to a single residue for k_ints
            if len(topo.residues) == 1:
                # jaxent_rates[topo.residues[0]] = jaxent_k_ints[i - 1]
                # jaxent_rates[topo.residues[0] + 1] = jaxent_k_ints[i]
                jaxent_rates[topo.residues[0]] = jaxent_k_ints[i]

            else:
                # If a topology represents multiple residues, we need a strategy.
                # For intrinsic rates, it's usually per residue.
                # For now, we'll skip multi-residue topologies for k_ints.
                print(
                    f"Skipping multi-residue topology {topo.fragment_name} for intrinsic rates mapping."
                )
        else:
            print(f"Warning: No k_int found for topology {topo.fragment_name} at index {i}.")

    print(f"Loaded {len(jaxent_rates)} jaxENT rates.")

    # 4. Align and Compare
    aligned_hdxer_rates = []
    aligned_jaxent_rates = []
    aligned_resids = []

    for resid, hdxer_rate in hdxer_rates.items():
        if resid in jaxent_rates:
            aligned_hdxer_rates.append(hdxer_rate)
            aligned_jaxent_rates.append(jaxent_rates[resid])
            aligned_resids.append(resid)

    if not aligned_resids:
        print(
            "No common residues found between HDXer and jaxENT data. Cannot plot or compute statistics."
        )
        return

    print(f"Found {len(aligned_resids)} common residues for comparison.")

    # Convert to numpy arrays for easier calculations
    aligned_hdxer_rates = np.array(aligned_hdxer_rates)
    aligned_jaxent_rates = np.array(aligned_jaxent_rates)

    # 5. Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(aligned_hdxer_rates, aligned_jaxent_rates, alpha=0.7)
    plt.plot(
        [min(aligned_hdxer_rates), max(aligned_hdxer_rates)],
        [min(aligned_hdxer_rates), max(aligned_hdxer_rates)],
        "r--",
        label="y=x",
    )  # Diagonal line for reference

    plt.xlabel("HDXer Intrinsic Rate (min^-1)")
    plt.ylabel("jaxENT Intrinsic Rate (min^-1)")
    plt.title("Comparison of Intrinsic Rates: HDXer vs jaxENT")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Plotting intrinsic rates vs. residue ID
    plot_residue_path = output_dir / "intrinsic_rates_vs_residue.png"
    plt.figure(figsize=(10, 6))
    plt.plot(aligned_resids, aligned_hdxer_rates, "o-", label="HDXer Rates", alpha=0.7)
    plt.plot(aligned_resids, aligned_jaxent_rates, "x-", label="jaxENT Rates", alpha=0.7)

    plt.xlabel("Residue ID")
    plt.ylabel("Intrinsic Rate (min^-1)")
    plt.title("Intrinsic Rates vs. Residue ID: HDXer vs jaxENT")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.yscale("log")  # Intrinsic rates can vary over orders of magnitude
    plt.tight_layout()
    plt.savefig(plot_residue_path)
    plt.close()
    print(f"Residue-wise plot saved to: {plot_residue_path}")

    # 6. Summary Statistics
    differences = aligned_jaxent_rates - aligned_hdxer_rates
    abs_differences = np.abs(differences)

    # Avoid division by zero for percentage difference
    percentage_differences = np.where(
        aligned_hdxer_rates != 0, (abs_differences / aligned_hdxer_rates) * 100, np.nan
    )

    stats_output = f"""
Intrinsic Rate Comparison Summary Statistics:
---------------------------------------------
Number of common residues: {len(aligned_resids)}

Mean Absolute Difference: {np.nanmean(abs_differences):.4f} min^-1
Median Absolute Difference: {np.nanmedian(abs_differences):.4f} min^-1
Standard Deviation of Absolute Differences: {np.nanstd(abs_differences):.4f} min^-1

Mean Percentage Absolute Difference: {np.nanmean(percentage_differences):.2f}%
Median Percentage Absolute Difference: {np.nanmedian(percentage_differences):.2f}%
"""
    print(stats_output)
    with open(stats_path, "w") as f:
        f.write(stats_output)
    print(f"Summary statistics saved to: {stats_path}")


if __name__ == "__main__":
    plot_intrinsic_rates()
