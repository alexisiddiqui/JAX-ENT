"""
jaxent.src.analysis.PCA
~~~~~~~~~~~~~~~~~~~~~~~
Shared PCA computation and plotting utilities.

Used by:
- jaxent.cli.efficient_k_cluster (single-trajectory kCluster workflow)
- jaxent.cli.ipca                (multi-trajectory iPCA workflow)
"""

from jaxent.src.analysis.PCA.core import (
    calculate_distances_and_perform_pca,
    calculate_multi_traj_pca,
    calculate_pairwise_rmsd,
    perform_pca_on_distances,
)
from jaxent.src.analysis.PCA.plots import (
    create_publication_plots,
    plot_combined_density,
    plot_combined_scatter,
    plot_condition_replicates,
)

__all__ = [
    # core
    "calculate_pairwise_rmsd",
    "perform_pca_on_distances",
    "calculate_distances_and_perform_pca",
    "calculate_multi_traj_pca",
    # plots
    "create_publication_plots",
    "plot_combined_scatter",
    "plot_combined_density",
    "plot_condition_replicates",
]
