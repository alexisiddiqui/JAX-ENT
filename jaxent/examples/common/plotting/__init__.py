"""
Shared plotting functions for JAX-ENT example scripts.

All functions accept an optional ``PlotStyle`` dataclass instead of relying on
hardcoded colour/marker dicts.  The ``setup_publication_style()`` call replaces
the ``sns.set_style("ticks"); sns.set_context(...)`` block repeated in ~12 scripts.
"""

from .style import setup_publication_style, _DEFAULT_STYLE
from .heatmaps import plot_convergence_maxent_heatmaps, plot_metric_heatmap
from .scores import plot_model_score_heatmaps, create_violin_plots
from .comparisons import plot_best_model_comparisons
from .distributions import plot_weight_distribution_lines
from .sweeps import plot_2d_heatmaps_grid, plot_1d_slices_2d_sweep
from .splits import plot_split_distributions, plot_enhanced_split_heatmap, plot_split_heatmap
from .uptake import plot_uptake_heatmap, plot_combined_uptake_comparison
from .gaps import plot_gap_analysis

__all__ = [
    "setup_publication_style",
    "_DEFAULT_STYLE",
    "plot_convergence_maxent_heatmaps",
    "plot_metric_heatmap",
    "plot_model_score_heatmaps",
    "create_violin_plots",
    "plot_best_model_comparisons",
    "plot_weight_distribution_lines",
    "plot_2d_heatmaps_grid",
    "plot_1d_slices_2d_sweep",
    "plot_split_distributions",
    "plot_enhanced_split_heatmap",
    "plot_split_heatmap",
    "plot_uptake_heatmap",
    "plot_combined_uptake_comparison",
    "plot_gap_analysis",
]
