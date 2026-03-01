from __future__ import annotations

import seaborn as sns
from ..config import PlotStyle

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
