from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PlotStyle
from .style import _get_style, _DEFAULT_STYLE

def plot_best_model_comparisons(
    df: pd.DataFrame,
    output_dir: str,
    style: PlotStyle | None = None,
) -> None:
    """Plot bar charts comparing best models across metrics (val_loss, KL, recovery)."""
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("val_loss", "Validation Loss", True),
        ("kl_divergence", "KL Divergence", True),
        ("recovery_percent", "Recovery %", False),
    ]

    available_metrics = [(col, label, log) for col, label, log in metrics if col in df.columns]
    if not available_metrics:
        print("No metrics available for comparison plots")
        return

    split_types = df["split_type"].unique() if "split_type" in df.columns else [None]
    colors = style.split_type_colors or _DEFAULT_STYLE.split_type_colors

    for col, label, use_log in available_metrics:
        fig, ax = plt.subplots(figsize=style.figsize_wide)
        plot_df = df.copy()
        if use_log:
            plot_df[col] = np.log10(plot_df[col].clip(lower=1e-300))
            label = f"log10({label})"

        sns.barplot(data=plot_df, x="ensemble", y=col, hue="split_type", ax=ax, palette=colors)
        ax.set_ylabel(label, fontweight="bold")
        ax.set_xlabel("Ensemble", fontweight="bold")
        ax.set_title(f"Best Models: {label}", fontweight="bold")
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"best_models_{col}.png"), dpi=style.dpi, bbox_inches="tight")
        plt.close()
