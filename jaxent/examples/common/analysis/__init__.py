"""
Shared analysis functions for JAX-ENT example scripts.

Consolidates the most-duplicated analysis computations. All functions are pure
numpy/pandas — no JAX-ENT library imports needed.
"""

from .stats import kl_divergence, effective_sample_size
from .loss_trajectories import extract_loss_trajectories, extract_loss_trajectories_2d
from .scoring import compute_model_scores, select_best_models, calculate_work_metrics
from .clustering import (
    calculate_cluster_ratios,
    calculate_recovery_JSD,
    calculate_recovery_percentage,
    calculate_dMSE,
    analyze_conformational_recovery,
)
from .validation import (
    _is_valid_array,
    _align_uptake_shape,
    get_experimental_uptake,
    calculate_mse,
)
from .convergence import find_best_convergence_threshold
from .weights import (
    extract_frame_weights_kl,
    extract_final_weights,
    extract_final_weights_2d,
    extract_weights_over_convergence_steps,
)
from .pairwise_kld import (
    _sym_kld_pairs, 
    compute_pairwise_kld_between_splits, 
    compute_sequential_maxent_kld,
    compute_pairwise_kld_between_splits_2d
)
from .mlm import (
    prepare_metric_columns,
    multiple_regression_analysis,
    stability_analysis_by_ensemble,
    stability_analysis_by_split,
    mixed_effects_analysis,
    compute_model_selection_performance,
    compute_correlations,
    run_analysis_on_subset,
)
from .selected_models import (
    SCORE_METRIC_COLORS,
    _metric_color_cache,
    _metric_palette_cycle,
    get_metric_color,
    p_to_stars,
    ttest_from_stats,
    calculate_kendalls_w,
    get_metric_order,
    load_and_process_data,
    aggregate_df,
    compute_minimax_df,
    run_fixed_effects_for_score,
    run_fixed_effects_analysis,
    calculate_concordance_maps,
    save_gt_scores,
)

__all__ = [
    "kl_divergence",
    "effective_sample_size",
    "extract_loss_trajectories",
    "extract_loss_trajectories_2d",
    "compute_model_scores",
    "select_best_models",
    "calculate_work_metrics",
    "calculate_cluster_ratios",
    "calculate_recovery_JSD",
    "calculate_recovery_percentage",
    "analyze_conformational_recovery",
    "calculate_dMSE",
    "_is_valid_array",
    "_align_uptake_shape",
    "get_experimental_uptake",
    "calculate_mse",
    "find_best_convergence_threshold",
    "extract_frame_weights_kl",
    "extract_final_weights",
    "extract_final_weights_2d",
    "extract_weights_over_convergence_steps",
    "_sym_kld_pairs",
    "compute_pairwise_kld_between_splits",
    "compute_sequential_maxent_kld",
    "compute_pairwise_kld_between_splits_2d",
    "prepare_metric_columns",
    "multiple_regression_analysis",
    "stability_analysis_by_ensemble",
    "stability_analysis_by_split",
    "mixed_effects_analysis",
    "compute_model_selection_performance",
    "compute_correlations",
    "run_analysis_on_subset",
    "SCORE_METRIC_COLORS",
    "_metric_color_cache",
    "_metric_palette_cycle",
    "get_metric_color",
    "p_to_stars",
    "ttest_from_stats",
    "calculate_kendalls_w",
    "get_metric_order",
    "load_and_process_data",
    "aggregate_df",
    "compute_minimax_df",
    "run_fixed_effects_for_score",
    "run_fixed_effects_analysis",
    "calculate_concordance_maps",
    "save_gt_scores",
]
