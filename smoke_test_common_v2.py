
import numpy as np
import pandas as pd
from jaxent.examples.common import loading, analysis

def test_scoring_metrics():
    print("Testing scoring metrics...")
    p = np.array([0.1, 0.2, 0.7])
    q = np.array([0.1, 0.2, 0.7])
    exp = np.array([0.1, 0.2, 0.7])
    
    dmse = analysis.calculate_dMSE(p, q, exp)
    print(f"dMSE (same): {dmse}")
    
    work = analysis.calculate_work_metrics(p, q)
    print(f"Work metrics (same): {work}")
    
    # Test JSD recovery
    cluster_assignments = np.array([0, 0, 1, 1, 2, 2])
    weights = np.array([0.5, 0.5, 0, 0, 0, 0]) # All in cluster 0
    target_ratios = {"Folded": 1.0, "PUF": 0.0}
    state_mapping = {0: "Folded", 1: "PUF", 2: "PUF"}
    
    recovery = analysis.calculate_recovery_percentage(cluster_assignments, weights, target_ratios, state_mapping)
    print(f"Recovery (100%): {recovery}%")

def test_loading_stubs():
    print("\nTesting loading function signatures...")
    print(f"load_all_optimization_results_2d as attr: {hasattr(loading, 'load_all_optimization_results_2d')}")
    print(f"save_split_data as attr: {hasattr(loading, 'save_split_data')}")

if __name__ == "__main__":
    test_scoring_metrics()
    test_loading_stubs()
