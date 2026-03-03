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
from .distributions import plot_weight_distribution_lines, plot_weight_distribution_maxent_panels, plot_weight_distribution_convergence_panels
from .convergence import plot_loss_convergence, plot_split_variability, plot_loss_convergence_2d, plot_split_variability_2d
from .kld import plot_kld_between_splits, plot_sequential_maxent_kld
from .sweeps import plot_2d_heatmaps_grid, plot_1d_slices_2d_sweep, plot_best_hyperparameters
from .volcano import plot_volcano_kl_recovery, plot_volcano_kl_recovery_averaged
from .recovery import (
    plot_metric_vs_regularization_strength,
    plot_metric_maxent_comparison,
    plot_recovery_vs_regularization_strength,
    plot_maxent_comparison,
)
from .splits import plot_split_distributions, plot_enhanced_split_heatmap, plot_split_heatmap
from .uptake import plot_uptake_heatmap, plot_combined_uptake_comparison
from .gaps import plot_gap_analysis
from .mlm import (
    plot_coefficient_comparison,
    plot_partial_r2_comparison,
    plot_stability_comparison,
    plot_eta_and_ftest,
    plot_scatter_and_distributions,
    plot_model_selection_performance,
    plot_correlations_bar_charts,
)
from .selected_models import (
    plot_score_panel,
    plot_minimax_panel,
    plot_rank_panel,
    plot_fixed_effects,
    plot_aggregated_analysis,
    plot_p_values,
)

__all__ = [
    "setup_publication_style",
    "_DEFAULT_STYLE",
    "plot_convergence_maxent_heatmaps",
    "plot_metric_heatmap",
    "plot_model_score_heatmaps",
    "create_violin_plots",
    "plot_best_model_comparisons",
    "plot_weight_distribution_lines",
    "plot_weight_distribution_maxent_panels",
    "plot_weight_distribution_convergence_panels",
    "plot_kld_between_splits",
    "plot_sequential_maxent_kld",
    "plot_loss_convergence",
    "plot_split_variability",
    "plot_loss_convergence_2d",
    "plot_split_variability_2d",
    "plot_2d_heatmaps_grid",
    "plot_1d_slices_2d_sweep",
    "plot_best_hyperparameters",
    "plot_volcano_kl_recovery",
    "plot_volcano_kl_recovery_averaged",
    "plot_metric_vs_regularization_strength",
    "plot_metric_maxent_comparison",
    "plot_recovery_vs_regularization_strength",
    "plot_maxent_comparison",
    "plot_split_distributions",
    "plot_enhanced_split_heatmap",
    "plot_split_heatmap",
    "plot_uptake_heatmap",
    "plot_combined_uptake_comparison",
    "plot_gap_analysis",
    "plot_coefficient_comparison",
    "plot_partial_r2_comparison",
    "plot_stability_comparison",
    "plot_eta_and_ftest",
    "plot_scatter_and_distributions",
    "plot_model_selection_performance",
    "plot_correlations_bar_charts",
    "plot_score_panel",
    "plot_minimax_panel",
    "plot_rank_panel",
    "plot_fixed_effects",
    "plot_aggregated_analysis",
    "plot_p_values",
]
