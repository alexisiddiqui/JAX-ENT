from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_convergence_column(df: pd.DataFrame, preferred: str, fallback: list[str]) -> str | None:
    if preferred in df.columns:
        return preferred
    for col in fallback:
        if col in df.columns:
            return col
    return None


def plot_volcano_kl_recovery(
    kl_ess_df,
    recovery_df,
    convergence_rates,
    output_dir,
    *,
    recovery_col: str = "recovery_percent",
    baseline_col: str | None = None,
    target_value: float = 100.0,
    target_label: str = "Target\n(100%)",
    fold_change_col: str = "recovery_fold_change",
    current_col: str = "current_recovery",
    baseline_output_col: str = "baseline_recovery",
    xlabel_suffix: str = "Recovery vs Unweighted Baseline",
    title_keyword: str = "Recovery",
    xlim: tuple[float, float] | None = None,
    save_target_csv: bool = False,
) -> pd.DataFrame | None:
    """Plot volcano plot with KL divergence vs recovery fold change."""
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Creating volcano plot...")
    print(f"KL/ESS DataFrame shape: {kl_ess_df.shape}")
    print(f"Recovery DataFrame shape: {recovery_df.shape}")

    recovery_filtered = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(recovery_filtered) == 0:
        print("No recovery data available for volcano plot")
        return None

    if len(kl_ess_df) == 0:
        print("No KL divergence data available for volcano plot")
        return None

    kl_ess_df = kl_ess_df.copy()
    recovery_filtered = recovery_filtered.copy()

    kl_conv_col = _find_convergence_column(
        kl_ess_df,
        preferred="convergence_threshold_step",
        fallback=["step", "convergence_step"],
    )
    recovery_conv_col = _find_convergence_column(
        recovery_filtered,
        preferred="convergence_step",
        fallback=["step"],
    )

    if kl_conv_col is None:
        print(
            "Warning: No convergence step column found in KL data. "
            f"Available columns: {kl_ess_df.columns.tolist()}"
        )
        return None

    if recovery_conv_col is None:
        print(
            "Warning: No convergence step column found in recovery data. "
            f"Available columns: {recovery_filtered.columns.tolist()}"
        )
        return None

    kl_ess_df[kl_conv_col] = pd.to_numeric(kl_ess_df[kl_conv_col], errors="coerce")
    recovery_filtered[recovery_conv_col] = pd.to_numeric(
        recovery_filtered[recovery_conv_col], errors="coerce"
    )

    merge_columns = ["split_type", "ensemble", "loss_function", "maxent_value", "split"]
    left_cols = merge_columns + [kl_conv_col]
    right_cols = merge_columns + [recovery_conv_col]

    merged_df = pd.merge(
        kl_ess_df,
        recovery_filtered,
        left_on=left_cols,
        right_on=right_cols,
        how="inner",
    )

    if len(merged_df) == 0:
        print("No exact matches found, trying to merge by taking final convergence steps...")

        kl_final = kl_ess_df.groupby(merge_columns).last().reset_index()
        recovery_final = recovery_filtered.groupby(merge_columns).last().reset_index()

        merged_df = pd.merge(
            kl_final,
            recovery_final,
            on=merge_columns,
            how="inner",
            suffixes=("_kl", "_recovery"),
        )

        if recovery_conv_col in merged_df.columns:
            merged_df["plot_convergence_step"] = merged_df[recovery_conv_col]
        elif f"{recovery_conv_col}_recovery" in merged_df.columns:
            merged_df["plot_convergence_step"] = merged_df[f"{recovery_conv_col}_recovery"]
        else:
            merged_df["plot_convergence_step"] = 1
    else:
        merged_df["plot_convergence_step"] = merged_df[recovery_conv_col]

    if len(merged_df) == 0:
        print("No matching data found for volcano plot after all merge attempts")
        return None

    print(f"Merged {len(merged_df)} data points for volcano plot")

    unweighted_recovery = recovery_df[recovery_df["loss_function"] == "Original"].copy()
    print(f"Found {len(unweighted_recovery)} unweighted baseline records")

    baseline_recoveries = {}
    for ensemble in unweighted_recovery["ensemble"].unique():
        baseline_data = unweighted_recovery[unweighted_recovery["ensemble"] == ensemble]
        if baseline_data.empty:
            baseline_recoveries[ensemble] = 0.0
            continue

        if baseline_col and baseline_col in baseline_data.columns:
            baseline_recoveries[ensemble] = baseline_data[baseline_col].iloc[0]
        elif recovery_col in baseline_data.columns:
            baseline_recoveries[ensemble] = baseline_data[recovery_col].iloc[0]
        else:
            baseline_recoveries[ensemble] = 0.0

    fold_change_data = []
    for _, row in merged_df.iterrows():
        ensemble = row["ensemble"]
        baseline_recovery = baseline_recoveries.get(ensemble, 0.0)

        if baseline_recovery > 0:
            if baseline_col and baseline_col in row.index and pd.notna(row[baseline_col]):
                current_recovery = row[baseline_col]
            elif recovery_col in row.index and pd.notna(row[recovery_col]):
                current_recovery = row[recovery_col]
            else:
                current_recovery = 0.0
            fold_change = current_recovery / baseline_recovery
            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
        else:
            fold_change = 1.0
            log2_fold_change = 0.0
            current_recovery = row.get(recovery_col, 0.0)

        fold_change_data.append(
            {
                **row.to_dict(),
                fold_change_col: fold_change,
                "log2_fold_change": log2_fold_change,
                baseline_output_col: baseline_recovery,
                current_col: current_recovery,
            }
        )

    if not fold_change_data:
        print("No fold change data could be calculated")
        return None

    volcano_df = pd.DataFrame(fold_change_data)

    target_fold_changes = {}
    for ensemble in volcano_df["ensemble"].unique():
        baseline_recovery = baseline_recoveries.get(ensemble, 0.0)
        if baseline_recovery > 0:
            target_fold_change = target_value / baseline_recovery
            target_fold_changes[ensemble] = np.log2(target_fold_change)
        else:
            target_fold_changes[ensemble] = 0

    split_types = volcano_df["split_type"].unique()
    for split_type in split_types:
        print(f"  Creating volcano plot for split type: {split_type}")
        split_data = volcano_df[volcano_df["split_type"] == split_type]

        if len(split_data) == 0:
            continue

        ensembles = sorted(split_data["ensemble"].unique())
        loss_functions = sorted(split_data["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Volcano Plot: KL Divergence vs {title_keyword} Fold Change - {split_type}",
            fontsize=16,
            fontweight="bold",
        )

        maxent_values = sorted(split_data["maxent_value"].unique())
        if len(maxent_values) > 1:
            log_maxent = np.log10([max(1, val) for val in maxent_values])
            norm = plt.Normalize(vmin=min(log_maxent), vmax=max(log_maxent))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        conv_steps = sorted(split_data["plot_convergence_step"].dropna().unique())
        max_size = 150
        min_size = 30

        size_map = {}
        for i, step in enumerate(conv_steps):
            step_int = int(step)
            if step_int > 0 and step_int <= len(convergence_rates):
                size = min_size + (i / max(1, len(conv_steps) - 1)) * (max_size - min_size)
                size_map[step] = size
            else:
                size_map[step] = min_size

        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "P", "X"]

        all_x_data = split_data["log2_fold_change"]
        all_y_data = split_data["kl_divergence"]

        x_margin = (all_x_data.max() - all_x_data.min()) * 0.05
        y_margin = (all_y_data.max() - all_y_data.min()) * 0.05

        global_xlim = [all_x_data.min() - x_margin, all_x_data.max() + x_margin]
        global_ylim = [all_y_data.min() - y_margin, all_y_data.max() + y_margin]

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_data = split_data[
                    (split_data["ensemble"] == ensemble)
                    & (split_data["loss_function"] == loss_func)
                ]

                if len(combo_data) > 0:
                    splits = sorted(combo_data["split"].unique())

                    for k, split_idx in enumerate(splits):
                        split_data_subset = combo_data[combo_data["split"] == split_idx]

                        if len(split_data_subset) > 0:
                            x_sub = split_data_subset["log2_fold_change"]
                            y_sub = split_data_subset["kl_divergence"]

                            colors_sub = []
                            for maxent_val in split_data_subset["maxent_value"]:
                                if len(maxent_values) > 1:
                                    colors_sub.append(cmap(norm(np.log10(max(1, maxent_val)))))
                                else:
                                    colors_sub.append("blue")

                            sizes_sub = [
                                size_map.get(step, min_size)
                                for step in split_data_subset["plot_convergence_step"]
                            ]

                            ax.scatter(
                                x_sub,
                                y_sub,
                                c=colors_sub,
                                s=sizes_sub,
                                marker=markers[k % len(markers)],
                                alpha=0.7,
                                edgecolors="black",
                                linewidth=0.5,
                                label=f"Replicate {split_idx}",
                            )

                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

                    if ensemble in target_fold_changes:
                        target_x = target_fold_changes[ensemble]
                        ax.axvline(
                            x=target_x, color="orange", linestyle=":", alpha=0.8, linewidth=2
                        )
                        ax.text(
                            target_x,
                            ax.get_ylim()[1] * 0.95,
                            target_label,
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    if xlim is not None:
                        ax.set_xlim(xlim)
                    else:
                        ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    ax.text(
                        0.98,
                        0.98,
                        "High KL\nHigh Recovery",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        fontsize=8,
                    )

                    ax.text(
                        0.02,
                        0.98,
                        "High KL\nLow Recovery",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                        fontsize=8,
                    )

                    ax.set_xlabel(f"Log2 Fold Change ({xlabel_suffix})")
                    ax.set_ylabel("KL Divergence")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.grid(True, alpha=0.3)

                    if i == 0 and j == 0:
                        legend_elements = []

                        for k, split_idx in enumerate(splits[:6]):
                            legend_elements.append(
                                plt.Line2D(
                                    [0],
                                    [0],
                                    marker=markers[k % len(markers)],
                                    color="w",
                                    markerfacecolor="gray",
                                    markersize=8,
                                    label=f"Replicate {split_idx}",
                                    markeredgecolor="black",
                                    markeredgewidth=0.5,
                                )
                            )

                        legend_elements.append(
                            plt.Line2D([0], [0], color="red", linestyle="--", label="No Fold Change")
                        )
                        legend_elements.append(
                            plt.Line2D(
                                [0],
                                [0],
                                color="orange",
                                linestyle=":",
                                label=target_label.replace("\n", " "),
                            )
                        )

                        ax.legend(
                            handles=legend_elements,
                            bbox_to_anchor=(1.05, 1),
                            loc="upper left",
                            fontsize=8,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    if ensemble in target_fold_changes:
                        target_x = target_fold_changes[ensemble]
                        ax.axvline(
                            x=target_x, color="orange", linestyle=":", alpha=0.8, linewidth=2
                        )
                        ax.text(
                            target_x,
                            ax.get_ylim()[1] * 0.95,
                            target_label,
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

        plt.tight_layout()

        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(split_output_dir, "volcano_plot_kl_recovery.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    volcano_df["target_log2_fold_change"] = volcano_df["ensemble"].map(target_fold_changes)

    volcano_df_path = os.path.join(output_dir, "volcano_plot_data.csv")
    volcano_df.to_csv(volcano_df_path, index=False)
    print(f"Volcano plot dataset saved to: {volcano_df_path}")

    if save_target_csv:
        target_df = pd.DataFrame(
            [
                {"ensemble": ensemble, "target_log2_fold_change": target_fc}
                for ensemble, target_fc in target_fold_changes.items()
            ]
        )
        target_path = os.path.join(output_dir, "target_fold_changes.csv")
        target_df.to_csv(target_path, index=False)
        print(f"Target fold changes saved to: {target_path}")

    return volcano_df


def plot_volcano_kl_recovery_averaged(
    kl_ess_df,
    recovery_df,
    convergence_rates,
    output_dir,
    **volcano_kwargs,
) -> None:
    """Plot averaged volcano plot with error bars."""
    plt.style.use("seaborn-v0_8-whitegrid")

    kwargs = {
        "recovery_col": "recovery_percent",
        "baseline_col": None,
        "target_value": 100.0,
        "target_label": "Target\n(100%)",
        "fold_change_col": "recovery_fold_change",
        "current_col": "current_recovery",
        "baseline_output_col": "baseline_recovery",
        "xlabel_suffix": "Recovery vs Unweighted Baseline",
        "title_keyword": "Recovery",
        "xlim": None,
        "save_target_csv": False,
    }
    kwargs.update(volcano_kwargs)

    current_col = kwargs["current_col"]
    baseline_output_col = kwargs["baseline_output_col"]
    target_label = kwargs["target_label"]
    title_keyword = kwargs["title_keyword"]
    xlabel_suffix = kwargs["xlabel_suffix"]

    print("Creating averaged volcano plot with error bars...")
    volcano_df = plot_volcano_kl_recovery(
        kl_ess_df,
        recovery_df,
        convergence_rates,
        output_dir,
        **kwargs,
    )

    if volcano_df is None or len(volcano_df) == 0:
        print("No volcano data available for averaged plot")
        return

    grouping_cols = [
        "split_type",
        "ensemble",
        "loss_function",
        "maxent_value",
        "plot_convergence_step",
    ]

    averaged_df = (
        volcano_df.groupby(grouping_cols)
        .agg(
            {
                "log2_fold_change": ["mean", "std", "count"],
                "kl_divergence": ["mean", "std", "count"],
                current_col: ["mean", "std"],
                baseline_output_col: "first",
                "target_log2_fold_change": "first",
            }
        )
        .reset_index()
    )

    averaged_df.columns = [
        "split_type",
        "ensemble",
        "loss_function",
        "maxent_value",
        "plot_convergence_step",
        "log2_fold_change_mean",
        "log2_fold_change_std",
        "log2_fold_change_count",
        "kl_divergence_mean",
        "kl_divergence_std",
        "kl_divergence_count",
        f"{current_col}_mean",
        f"{current_col}_std",
        baseline_output_col,
        "target_log2_fold_change",
    ]

    averaged_df["log2_fold_change_std"] = averaged_df["log2_fold_change_std"].fillna(0)
    averaged_df["kl_divergence_std"] = averaged_df["kl_divergence_std"].fillna(0)
    averaged_df[f"{current_col}_std"] = averaged_df[f"{current_col}_std"].fillna(0)

    split_types = averaged_df["split_type"].unique()
    for split_type in split_types:
        print(f"  Creating averaged volcano plot for split type: {split_type}")
        split_data = averaged_df[averaged_df["split_type"] == split_type]

        if len(split_data) == 0:
            continue

        ensembles = sorted(split_data["ensemble"].unique())
        loss_functions = sorted(split_data["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Averaged Volcano Plot: KL Divergence vs {title_keyword} Fold Change - {split_type}",
            fontsize=16,
            fontweight="bold",
        )

        maxent_values = sorted(split_data["maxent_value"].unique())
        if len(maxent_values) > 1:
            log_maxent = np.log10([max(1, val) for val in maxent_values])
            norm = plt.Normalize(vmin=min(log_maxent), vmax=max(log_maxent))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        all_x_data = split_data["log2_fold_change_mean"]
        all_y_data = split_data["kl_divergence_mean"]

        x_with_error = np.concatenate(
            [
                all_x_data + split_data["log2_fold_change_std"],
                all_x_data - split_data["log2_fold_change_std"],
            ]
        )
        y_with_error = np.concatenate(
            [
                all_y_data + split_data["kl_divergence_std"],
                all_y_data - split_data["kl_divergence_std"],
            ]
        )

        x_margin = (x_with_error.max() - x_with_error.min()) * 0.05
        y_margin = (y_with_error.max() - y_with_error.min()) * 0.05

        global_xlim = [x_with_error.min() - x_margin, x_with_error.max() + x_margin]
        global_ylim = [y_with_error.min() - y_margin, y_with_error.max() + y_margin]

        conv_steps = sorted(split_data["plot_convergence_step"].dropna().unique())
        max_size = 150
        min_size = 30

        size_map = {}
        for i, step in enumerate(conv_steps):
            step_int = int(step)
            if step_int > 0 and step_int <= len(convergence_rates):
                size = min_size + (i / max(1, len(conv_steps) - 1)) * (max_size - min_size)
                size_map[step] = size
            else:
                size_map[step] = min_size

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_data = split_data[
                    (split_data["ensemble"] == ensemble)
                    & (split_data["loss_function"] == loss_func)
                ]

                if len(combo_data) > 0:
                    x_vals = combo_data["log2_fold_change_mean"]
                    y_vals = combo_data["kl_divergence_mean"]
                    x_errs = combo_data["log2_fold_change_std"]
                    y_errs = combo_data["kl_divergence_std"]

                    colors_vals = []
                    for maxent_val in combo_data["maxent_value"]:
                        if len(maxent_values) > 1:
                            colors_vals.append(cmap(norm(np.log10(max(1, maxent_val)))))
                        else:
                            colors_vals.append("blue")

                    sizes_vals = [
                        size_map.get(step, min_size) for step in combo_data["plot_convergence_step"]
                    ]

                    ax.scatter(
                        x_vals,
                        y_vals,
                        c=colors_vals,
                        s=sizes_vals,
                        marker="o",
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=3,
                    )

                    ax.errorbar(
                        x_vals,
                        y_vals,
                        xerr=x_errs,
                        yerr=y_errs,
                        fmt="none",
                        ecolor="black",
                        alpha=0.5,
                        capsize=2,
                        capthick=1,
                        zorder=2,
                    )

                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

                    target_log2_fold_change = combo_data["target_log2_fold_change"].iloc[0]
                    if pd.notna(target_log2_fold_change):
                        ax.axvline(
                            x=target_log2_fold_change,
                            color="orange",
                            linestyle=":",
                            alpha=0.8,
                            linewidth=2,
                        )
                        ax.text(
                            target_log2_fold_change,
                            ax.get_ylim()[1] * 0.95,
                            target_label,
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    if kwargs["xlim"] is not None:
                        ax.set_xlim(kwargs["xlim"])
                    else:
                        ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    ax.text(
                        0.98,
                        0.98,
                        "High KL\nHigh Recovery",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        fontsize=8,
                    )

                    ax.text(
                        0.02,
                        0.98,
                        "High KL\nLow Recovery",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                        fontsize=8,
                    )

                    ax.set_xlabel(f"Log2 Fold Change ({xlabel_suffix})")
                    ax.set_ylabel("KL Divergence")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()

        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(split_output_dir, "volcano_plot_kl_recovery_averaged.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    averaged_df_path = os.path.join(output_dir, "volcano_plot_averaged_data.csv")
    averaged_df.to_csv(averaged_df_path, index=False)
    print(f"Averaged volcano plot dataset saved to: {averaged_df_path}")
