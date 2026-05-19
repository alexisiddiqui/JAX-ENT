from __future__ import annotations

import re as _re
import seaborn as sns
from ..config import PlotStyle

METRIC_DISPLAY_NAMES: dict[str, str] = {
    # Loss
    "val_loss": "Validation Loss",
    "train_loss": "Training Loss",
    # MSE variants (loss function names)
    "mcMSE": "MC-MSE",
    "MSE": "MSE",
    "Sigma_MSE": r"$\Sigma$-MSE",
    "d_mse": r"$\Delta$MSE",
    # MSE column variants (MLM predictor columns)
    "train_mse": "Train MSE",
    "val_mse": "Validation MSE",
    "test_mse": "Test MSE",
    # Delta-MSE column variants
    "d_mse_train": r"$\Delta$MSE (Train)",
    "d_mse_val": r"$\Delta$MSE (Val)",
    "d_mse_test": r"$\Delta$MSE (Test)",
    # Information-theoretic
    "kl_divergence": r"$D_\mathrm{KL}$",
    "js_divergence": "JS Divergence",
    "js_distance": "JS Distance",
    # Work metrics (base names)
    "work_scale": "Work Scale",
    "work_shape": "Work Shape",
    "work_density": "Work Density",
    "work_fitting": "Work Fitting",
    "work_magnitude": "Work Magnitude",
    "work_opt": "Work (Opt)",
    # Work metrics with _kj unit suffix
    "work_scale_kj": "Work Scale (kJ)",
    "work_shape_kj": "Work Shape (kJ)",
    "work_density_kj": "Work Density (kJ)",
    "work_fitting_kj": "Work Fitting (kJ)",
    "work_magnitude_kj": "Work Magnitude (kJ)",
    # Recovery
    "recovery_percent": "Recovery (%)",
    "jsd_recovery": "JSD Recovery",
    "open_recovery": "Open Recovery",
    "unfolded_recovery": "Unfolded Recovery",
    "Folded_recovery": "Folded Recovery",
    "PUF1_recovery": "PUF1 Recovery",
    "PUF2_recovery": "PUF2 Recovery",
    # Auxiliary
    "maxent_value": "MaxEnt Value",
    "convergence_value": "Convergence Value",
    "bv_reg_value": "BV Reg Value",
    "effective_sample_size": "Effective Sample Size",
    "spearman_mean": "Spearman Correlation",
}

_STRIP_SUFFIXES = ("_percentile", "_rank", "_transformed", "_mean", "_std")
_CLUSTER_RE = _re.compile(r"^cluster_(-?\d+)(?:_(?:percentile|rank|transformed))?$")


def _cluster_label(m: "_re.Match[str]", style: PlotStyle | None) -> str:
    idx = m.group(1)
    if idx == "-1":
        return "Unclustered"
    if style is not None and style.cluster_name_mapping:
        name = style.cluster_name_mapping.get(int(idx)) or style.cluster_name_mapping.get(str(idx))
        if name:
            return name
    return f"Cluster {idx}"


def _get_metric_display_name(metric: str, style: PlotStyle | None = None) -> str:
    """Map a raw metric column name to a human-readable display label.

    Handles ``_percentile``, ``_rank``, ``_transformed``, ``_mean``, ``_std``
    suffixes by stripping them and looking up the base name.
    ``cluster_N`` patterns map to the configured state name or ``"Cluster N"``.
    """
    mapping = dict(METRIC_DISPLAY_NAMES)
    if style is not None and style.metric_name_mapping:
        mapping.update(style.metric_name_mapping)

    if metric in mapping:
        return mapping[metric]

    # Strip one known suffix and retry against mapping and cluster regex
    for suffix in _STRIP_SUFFIXES:
        if metric.endswith(suffix):
            base = metric[: -len(suffix)]
            if base in mapping:
                return mapping[base]
            m = _CLUSTER_RE.match(base)
            if m:
                return _cluster_label(m, style)
            break  # only strip one suffix

    # cluster_N or cluster_N_<suffix>
    m = _CLUSTER_RE.match(metric)
    if m:
        return _cluster_label(m, style)

    return metric


_DEFAULT_STYLE = PlotStyle(
    ensemble_colors={},
    loss_markers={"mcMSE": "o", "MSE": "s", "Sigma_MSE": "^"},
    split_type_colors={
        "r": "fuchsia",
        "s": "black",
        "R3": "green",
        "sequence_cluster": "green",
        "Sp": "grey",
        "random": "#9467bd",
    },
    split_name_mapping={
        "r": "Random",
        "s": "Sequence",
        "R3": "Non-Redundant",
        "sequence_cluster": "Non-Redundant",
        "Sp": "Spatial",
        "spatial": "Spatial",
        "random": "Random",
    },
)

def setup_publication_style() -> None:
    """Apply publication-ready matplotlib/seaborn style.

    Replaces ~12 identical inline blocks across example scripts.
    """
    sns.set_style("ticks")
    sns.set_context(
        "paper",
        rc={
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 10,
        },
    )

def _get_style(style: PlotStyle | None) -> PlotStyle:
    return style if style is not None else _DEFAULT_STYLE
