#!/usr/bin/env python3
"""
Calculates the mean recovery percentage of conformational states based on clustering assignments.

This script reads clustering data, which assigns each frame of a molecular dynamics trajectory
to a specific cluster. It then maps these clusters to predefined conformational states (e.g., Folded, PUF1, PUF2).

Using a target ratio for each state (defined in a JSON file), the script calculates the
current proportion of each state found in the clustering data. The "recovery percentage" is
then calculated for each state, defined as:

    (current_proportion / target_proportion) * 100

This provides a measure of how well the observed conformational sampling matches the expected
thermodynamic populations.

The script assumes a uniform weighting for each frame unless a weights file is provided.
"""

import argparse
import json
import pandas as pd
import numpy as np
import os


def calculate_recovery_percentage(cluster_assignments, weights, target_ratios, state_mapping):
    """
    Calculates the current proportions and recovery percentages for conformational states.

    Args:
        cluster_assignments (pd.Series): A pandas Series where each value is the cluster label for a frame.
        weights (np.ndarray): An array of weights, one for each frame. The sum of weights should be 1.0.
        target_ratios (dict): A dictionary mapping state names to their target fractional populations.
        state_mapping (dict): A dictionary mapping cluster labels (int) to state names (str).

    Returns:
        dict: A dictionary containing the results, with current proportions and recovery percentages for each state.
    """
    results = {}
    total_frames = len(cluster_assignments)
    if total_frames == 0:
        return {"error": "No frames found in cluster assignments."}

    # Invert the state_mapping to group clusters by state
    state_to_clusters = {}
    for cluster_id, state_name in state_mapping.items():
        if state_name not in state_to_clusters:
            state_to_clusters[state_name] = []
        state_to_clusters[state_name].append(cluster_id)

    # Calculate current proportions
    current_proportions = {state: 0.0 for state in target_ratios}
    for state_name, cluster_ids in state_to_clusters.items():
        # Create a boolean mask for frames belonging to any of the clusters for the current state
        state_mask = cluster_assignments.isin(cluster_ids)
        # Sum the weights of these frames (use numpy boolean indexing)
        current_proportions[state_name] = float(np.sum(weights[state_mask.to_numpy()]))

    # Calculate recovery percentages
    recovery_percentages = {}
    for state_name, target in target_ratios.items():
        current = current_proportions.get(state_name, 0.0)
        if target > 0:
            recovery = (current / target) * 100
        else:
            recovery = 0.0 if current == 0.0 else float('inf')
        recovery_percentages[state_name] = recovery

    results = {
        "target_proportions": target_ratios,
        "current_proportions": current_proportions,
        "recovery_percentages": recovery_percentages,
    }
    return results

def calculate_recovery_JSD(cluster_assignments, weights, target_ratios, state_mapping):
    """
    Compute a single Jensen-Shannon divergence between the observed current proportions
    (over the provided states) and the target_ratios. Uses log base 2 so JS is in [0,1].
    Returns the JS divergence (float). If the current distribution is all zeros, returns np.nan.
    JSD Recovery% can be computed as (1 - sqrt(JS)) * 100.
    """
    # Invert mapping: state -> cluster ids
    state_to_clusters = {}
    for cluster_id, state_name in state_mapping.items():
        state_to_clusters.setdefault(state_name, []).append(cluster_id)

    # Compute current proportions (weighted)
    current_proportions = {state: 0.0 for state in target_ratios}
    for state_name, cluster_ids in state_to_clusters.items():
        state_mask = cluster_assignments.isin(cluster_ids)
        current_proportions[state_name] = float(np.sum(weights[state_mask.to_numpy()]))

    # Order states consistently with target_ratios
    states = list(target_ratios.keys())
    P = np.array([current_proportions.get(s, 0.0) for s in states], dtype=float)
    Q = np.array([target_ratios.get(s, 0.0) for s in states], dtype=float)

    # Normalize distributions to sum to 1 (if possible)
    sumP = P.sum()
    sumQ = Q.sum()
    if sumP > 0:
        P = P / sumP
    else:
        # No observed probability mass for these states -> undefined JS
        return np.nan

    if sumQ > 0:
        Q = Q / sumQ
    else:
        # Invalid target distribution
        return np.nan

    # Jensen-Shannon divergence (base 2)
    M = 0.5 * (P + Q)

    def kld(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    js = 0.5 * (kld(P, M) + kld(Q, M))
    return float(js)

def main():
    """Main function to parse arguments and run the recovery calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate mean recovery percentage for conformational states from clustering data."
    )
    parser.add_argument(
        "--cluster_assignments_csv",
        type=str,
        default="jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters/global_frame_to_cluster_ensemble.csv",
        help="Path to the CSV file with cluster assignments for each frame.",
    )
    parser.add_argument(
        "--state_ratios_json",
        type=str,
        default="jaxent/examples/2_CrossValidation/analysis/state_ratios.json",
        help="Path to the JSON file with target state ratios.",
    )
    args = parser.parse_args()

    # --- Configuration ---
    # Mapping from cluster ID to conformational state name
    state_mapping = {
        0: "Folded",
        1: "PUF1",
        2: "PUF2",
    }

    # --- Load Data ---
    try:
        assignments_df = pd.read_csv(args.cluster_assignments_csv)
    except FileNotFoundError:
        print(f"Error: Cluster assignments file not found at {args.cluster_assignments_csv}")
        return

    try:
        with open(args.state_ratios_json, 'r') as f:
            ratios_data = json.load(f)
        # Extract the fractional populations
        target_ratios = {
            "Folded": ratios_data["fractional_populations"]["folded"]["fraction"],
            "PUF1": ratios_data["fractional_populations"]["PUF1"]["fraction"],
            "PUF2": ratios_data["fractional_populations"]["PUF2"]["fraction"],
        }
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading or parsing state ratios file {args.state_ratios_json}: {e}")
        return

    # --- Process Each Ensemble Individually ---
    ensemble_groups = assignments_df.groupby("ensemble_name")

    for ensemble_name, ensemble_df in ensemble_groups:
        # --- Prepare Inputs for the current ensemble ---
        cluster_assignments = ensemble_df["cluster_label"]
        n_frames = len(cluster_assignments)

        if n_frames == 0:
            continue

        # Assume uniform weights for each frame within the ensemble
        weights = np.full(n_frames, 1/n_frames)

        # --- Calculate Recovery ---
        recovery_results = calculate_recovery_percentage(cluster_assignments, weights, target_ratios, state_mapping)
        js_divergence = calculate_recovery_JSD(cluster_assignments, weights, target_ratios, state_mapping)

        # --- Print Results ---
        if "error" in recovery_results:
            print(f"An error occurred for ensemble {ensemble_name}: {recovery_results['error']}")
            continue

        print(f"--- State Recovery Analysis for Ensemble: {ensemble_name} ---")
        print(f"Source: {os.path.basename(args.cluster_assignments_csv)}")
        print(f"Total Frames: {n_frames}\n")

        print(f"{'State':<10} | {'Target (%)':<12} | {'Current (%)':<12} | {'Recovery (%)':<12}")
        print("-" * 55)

        for state in target_ratios:
            target_p = recovery_results["target_proportions"][state] * 100
            current_p = recovery_results["current_proportions"][state] * 100
            recovery_p = recovery_results["recovery_percentages"][state]
            print(f"{state:<10} | {target_p:<12.2f} | {current_p:<12.2f} | {recovery_p:<12.1f}")

        print("-" * 55)
        # Print ensemble-level Jensen-Shannon divergence (and distance)
        if np.isnan(js_divergence):
            print("Jensen-Shannon divergence (states): n/a (no observed mass for listed states)")
        else:
            js_dist = float(np.sqrt(js_divergence))
            print(f"Jensen-Shannon divergence (base 2): {js_divergence:.6f}")
            print(f"Jensen-Shannon distance (sqrt(JS)): {js_dist:.6f}")
        print()  # Add a blank line for spacing between ensembles


if __name__ == "__main__":
    main()
