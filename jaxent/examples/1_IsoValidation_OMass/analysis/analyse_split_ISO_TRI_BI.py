"""
analyse_split_ISO_TRI_BI.py

Analyses and plots the data splits for the ISO TRI BI dataset.
Delegates all logic to jaxent.examples.common (loading, plotting).

Usage:
    python jaxent/examples/1_IsoValidation_OMass/analysis/analyse_split_ISO_TRI_BI.py
"""

import os
import pandas as pd
from jaxent.examples.common import loading, plotting


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "fitting", "jaxENT", "_datasplits")
    output_base_dir = os.path.join(os.path.dirname(__file__), "_analysis_split_iso_tri_bi")
    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.exists(base_dir):
        print(f"Data directory not found: {base_dir}")
        print("Please run the `splitdata_ISO.py` script first.")
        return

    split_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for split_type in split_types:
        split_type_dir = os.path.join(base_dir, split_type)
        output_dir = os.path.join(output_base_dir, split_type)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- Analysing split type: {split_type} ---")

        split_dirs = [
            os.path.join(split_type_dir, d)
            for d in os.listdir(split_type_dir)
            if os.path.isdir(os.path.join(split_type_dir, d)) and d.startswith("split_")
        ]

        if not split_dirs:
            print(f"No splits found for type {split_type}, skipping.")
            continue

        all_peptides_data = []

        for split_dir in split_dirs:
            train_top, train_dfrac, val_top, val_dfrac = loading.load_split_data(split_dir)
            split_name = os.path.basename(split_dir)

            plotting.plot_uptake_heatmap(
                train_top, train_dfrac,
                f"Training Set Uptake - {split_name}",
                os.path.join(output_dir, f"uptake_train_{split_name}.png"),
                split_type="train",
            )
            plotting.plot_uptake_heatmap(
                val_top, val_dfrac,
                f"Validation Set Uptake - {split_name}",
                os.path.join(output_dir, f"uptake_val_{split_name}.png"),
                split_type="validation",
            )
            plotting.plot_combined_uptake_comparison(
                train_top, train_dfrac, val_top, val_dfrac, split_name, output_dir
            )

            for p in train_top:
                all_peptides_data.append({
                    "peptide_index": p.fragment_index, "split_dir": split_name,
                    "split_type": "train", "present": 1,
                })
            for p in val_top:
                all_peptides_data.append({
                    "peptide_index": p.fragment_index, "split_dir": split_name,
                    "split_type": "validation", "present": 1,
                })

        if not all_peptides_data:
            print(f"No data found in split directories for type {split_type}.")
            continue

        all_splits_df = pd.DataFrame(all_peptides_data)
        plotting.plot_split_distributions(all_splits_df, output_dir)
        plotting.plot_split_heatmap(all_splits_df, output_dir)
        plotting.plot_enhanced_split_heatmap(all_splits_df, output_dir)
        plotting.plot_gap_analysis(all_splits_df, output_dir)

        print(f"Analysis for {split_type} complete. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
