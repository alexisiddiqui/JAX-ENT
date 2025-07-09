import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Define component names globally for consistency
COMPONENT_NAMES = ["ExpL2", "MaxENT", "MAE", "MRE"]


def ensure_output_dir(name="reanalysis"):
    """Create the output directory if it doesn't exist."""
    output_dir = f"plots/{name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_results(json_path):
    """Load results from a JSON file."""
    print(f"Loading results from {json_path}")
    with open(json_path, "r") as f:
        results = json.load(f)
    return results


def plot_individual_result(result, plots_dir, combination_index):
    """Plot the individual optimization result with focus on validation loss components."""
    scaling_values = result["forward_model_scaling"]

    # Create figure with 2 rows
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Row 1: Validation Loss Components
    ax1 = fig.add_subplot(gs[0, :])

    # Plot validation loss components
    component_count = len(result["final_val_components"])

    # Check if we have unscaled history data
    if "unscaled_val_history" in result:
        for i in range(component_count):
            component_history = [float(step[i]) for step in result["unscaled_val_history"]]
            component_label = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
            ax1.plot(component_history, label=f"Val {component_label}")
    else:
        # Fall back to plotting total validation loss if component history isn't available
        ax1.plot(result["loss_history"]["val"], label="Total Validation Loss")

    # Add labels and title
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Validation Loss Component Value")
    ax1.set_title(f"Validation Loss Components for Scaling: {scaling_values}")
    ax1.legend()

    # Add text with scaling values and component information
    val_components_text = ", ".join(
        [f"{float(comp):.4f}" for comp in result["best_val_components"]]
    )
    component_labels = []
    for i in range(component_count):
        comp_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
        comp_value = float(result["best_val_components"][i])
        component_labels.append(f"{comp_name}: {comp_value:.4f}")

    textstr = "\n".join(
        [
            f"Scaling: {scaling_values}",
            f"{component_labels[0]}",
            f"All Components: [{val_components_text}]",
            f"Steps: {result['actual_steps']}",
            f"Time: {float(result['execution_time']) if 'execution_time' in result else 'N/A'} s",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax1.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # Row 2: Pairwise analysis of validation loss components
    # Create a mock dataframe for component analysis if unscaled history isn't available
    if "unscaled_val_history" in result:
        val_comp_histories = {}
        for i in range(component_count):
            comp_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
            val_comp_histories[comp_name] = [
                float(step[i]) for step in result["unscaled_val_history"]
            ]

        df_val_history = pd.DataFrame(val_comp_histories)

        # If multiple components, show pairwise relationships
        if component_count >= 2:
            # Left plot: ExpL2 vs MaxENT
            ax2 = fig.add_subplot(gs[1, 0])
            sns.scatterplot(data=df_val_history, x=COMPONENT_NAMES[0], y=COMPONENT_NAMES[1], ax=ax2)
            ax2.set_title(f"{COMPONENT_NAMES[0]} vs {COMPONENT_NAMES[1]}")

            # Right plot: ExpL2 vs another component or histogram
            ax3 = fig.add_subplot(gs[1, 1])
            if component_count >= 3:
                sns.scatterplot(
                    data=df_val_history, x=COMPONENT_NAMES[0], y=COMPONENT_NAMES[2], ax=ax3
                )
                ax3.set_title(f"{COMPONENT_NAMES[0]} vs {COMPONENT_NAMES[2]}")
            else:
                # Only two components, show distribution of ExpL2
                sns.histplot(data=df_val_history, x=COMPONENT_NAMES[0], kde=True, ax=ax3)
                ax3.set_title(f"Distribution of {COMPONENT_NAMES[0]}")
        else:
            # Only one component, show its distribution
            ax2 = fig.add_subplot(gs[1, :])
            sns.histplot(data=df_val_history, x=COMPONENT_NAMES[0], kde=True, ax=ax2)
            ax2.set_title(f"Distribution of {COMPONENT_NAMES[0]}")
    else:
        # If unscaled history isn't available, show the total validation loss
        ax2 = fig.add_subplot(gs[1, :])
        total_val_loss = result["loss_history"]["val"]
        ax2.plot(total_val_loss)
        ax2.set_title("Total Validation Loss Over Optimization Steps")
        ax2.set_xlabel("Optimization Step")
        ax2.set_ylabel("Total Validation Loss")

    # Save figure
    fig_path = os.path.join(plots_dir, f"combination_{combination_index}.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create additional heatmap/correlation matrix if multiple components and unscaled history is available
    if component_count > 1 and "unscaled_val_history" in result:
        fig2 = plt.figure(figsize=(10, 8))
        # Create correlation matrix
        corr_matrix = df_val_history.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix of Validation Loss Components")
        fig2_path = os.path.join(plots_dir, f"val_corr_matrix_{combination_index}.png")
        fig2.tight_layout()
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)


def create_summary_plots(all_results, plots_dir):
    """Create summary plots focusing on validation loss components."""
    # Extract validation components from results
    validation_components = []
    for result in all_results:
        validation_components.append([float(comp) for comp in result["best_val_components"]])

    # Get component count
    component_count = len(validation_components[0])

    # Create dataframe for components
    df_columns = [
        COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
        for i in range(component_count)
    ]
    df_components = pd.DataFrame(validation_components, columns=df_columns)

    # Add scaling parameters and index
    df_components["Scaling"] = [str(result["forward_model_scaling"]) for result in all_results]
    df_components["Config"] = range(len(all_results))

    # Add individual scaling parameter columns for plotting
    for i in range(len(all_results[0]["forward_model_scaling"])):
        df_components[f"Scale_{i}"] = [
            float(result["forward_model_scaling"][i]) for result in all_results
        ]

    # Sort by first component (ExpL2)
    sorted_by_comp0 = df_components.sort_values(COMPONENT_NAMES[0]).reset_index(drop=True)

    # Create parameter variation plots
    create_2d_surface_plots(df_components, COMPONENT_NAMES[0], plots_dir)
    create_1d_slice_plots(df_components, COMPONENT_NAMES[0], plots_dir)

    # Plot 1: Bar chart of top configurations by ExpL2
    plt.figure(figsize=(12, 8))
    top_n = min(10, len(sorted_by_comp0))

    # Get indices of top configurations
    top_configs = sorted_by_comp0.iloc[:top_n]

    # Create bar chart for ExpL2
    plt.bar(range(top_n), top_configs[COMPONENT_NAMES[0]], color="steelblue")
    plt.xlabel("Configuration")
    plt.ylabel(f"Value of {COMPONENT_NAMES[0]}")
    plt.title(f"Top Configurations by {COMPONENT_NAMES[0]} Validation Loss Component")
    plt.xticks(range(top_n), top_configs["Scaling"], rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    plt.savefig(
        os.path.join(plots_dir, f"top_by_{COMPONENT_NAMES[0]}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Comparison of all validation components for top configurations
    fig, ax = plt.subplots(figsize=(14, 10))

    # Set width based on number of components
    width = 0.8 / component_count
    x = np.arange(top_n)

    # Plot each component
    for i in range(component_count):
        comp_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
        comp_values = top_configs[comp_name]
        ax.bar(x + (i - component_count / 2 + 0.5) * width, comp_values, width, label=comp_name)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Component Value")
    ax.set_title("Validation Loss Components for Top Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(top_configs["Scaling"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plots_dir, "component_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Pairwise scatter plots of validation components
    if component_count >= 2:
        fig, axes = plt.subplots(1, component_count - 1, figsize=(5 * component_count, 6))

        # Handle case of only 2 components (1 comparison)
        if component_count == 2:
            axes = [axes]

        # Create scatter plots: ExpL2 vs each other component
        for i in range(1, component_count):
            comp1_name = COMPONENT_NAMES[0]
            comp2_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"

            axes[i - 1].scatter(df_components[comp1_name], df_components[comp2_name])
            axes[i - 1].set_xlabel(comp1_name)
            axes[i - 1].set_ylabel(comp2_name)
            axes[i - 1].set_title(f"{comp1_name} vs {comp2_name}")

            # Add config numbers as annotations
            for j, config in enumerate(df_components["Config"]):
                axes[i - 1].annotate(
                    str(config),
                    (df_components[comp1_name].iloc[j], df_components[comp2_name].iloc[j]),
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "component_pairplots.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Plot 4: Modified table of all configurations sorted by ExpL2
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(all_results) * 0.4 + 2))
    ax.axis("off")

    # Determine number of scaling parameters
    num_scaling_params = len(all_results[0]["forward_model_scaling"])

    # Define colors for scaling parameters (using matplotlib color names)
    scaling_colors = ["blue", "purple", "orange", "green"]
    # Ensure we have enough colors (cycle if needed)
    while len(scaling_colors) < num_scaling_params:
        scaling_colors.extend(scaling_colors[: num_scaling_params - len(scaling_colors)])

    # Prepare table data
    table_data = []
    for idx, row in sorted_by_comp0.iterrows():
        # Extract individual scaling parameters
        scaling_values = []
        for i in range(num_scaling_params):
            scaling_values.append(f"{float(row[f'Scale_{i}']):.3g}")

        # Add to table data (excluding "Other Components" and "Sum")
        row_data = [
            idx + 1,  # Rank
            int(row["Config"]),  # Config #
        ]
        # Add individual scaling parameters
        row_data.extend(scaling_values)
        # Add ExpL2
        row_data.append(f"{float(row[COMPONENT_NAMES[0]]):.3g}")

        table_data.append(row_data)

    # Create column labels
    col_labels = ["Rank", "Config #"]
    for i in range(num_scaling_params):
        col_labels.append(f"Scale_{i}")
    col_labels.append(COMPONENT_NAMES[0])  # ExpL2

    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Apply colors to the scaling parameter columns
    for i in range(num_scaling_params):
        col_idx = i + 2  # +2 because first two columns are Rank and Config #
        color = scaling_colors[i]

        # Apply color to header
        table[(0, col_idx)].set_facecolor(color)
        table[(0, col_idx)].set_text_props(color="white")

        # Apply lighter color to cells in this column
        for row_idx in range(1, len(table_data) + 1):
            alpha = 0.3  # Lighter color for data cells
            rgb_color = plt.cm.colors.to_rgba(color, alpha)
            table[(row_idx, col_idx)].set_facecolor(rgb_color)

    plt.savefig(os.path.join(plots_dir, "configurations_table.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4.5: Table with normalized scaling constants (sum to 1)
    fig, ax = plt.subplots(figsize=(12, len(all_results) * 0.4 + 2))
    ax.axis("off")

    # Prepare table data with normalized scaling constants
    normalized_table_data = []
    for idx, row in sorted_by_comp0.iterrows():
        # Extract and normalize scaling parameters to sum to 1
        scaling_values_raw = [float(row[f"Scale_{i}"]) for i in range(num_scaling_params)]
        scaling_sum = sum(scaling_values_raw)

        # Handle the case where sum is zero
        if scaling_sum == 0:
            normalized_scaling = [0] * num_scaling_params
        else:
            normalized_scaling = [val / scaling_sum for val in scaling_values_raw]

        normalized_scaling_formatted = [f"{val:.3g}" for val in normalized_scaling]

        # Add to table data
        row_data = [
            idx + 1,  # Rank
            int(row["Config"]),  # Config #
        ]
        # Add normalized scaling parameters
        row_data.extend(normalized_scaling_formatted)
        # Add ExpL2
        row_data.append(f"{float(row[COMPONENT_NAMES[0]]):.3g}")

        normalized_table_data.append(row_data)

    # Create the normalized table
    normalized_table = ax.table(
        cellText=normalized_table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Format table
    normalized_table.auto_set_font_size(False)
    normalized_table.set_fontsize(9)
    normalized_table.scale(1, 1.5)

    # Apply colors to the scaling parameter columns (same as before)
    for i in range(num_scaling_params):
        col_idx = i + 2  # +2 because first two columns are Rank and Config #
        color = scaling_colors[i]

        # Apply color to header
        normalized_table[(0, col_idx)].set_facecolor(color)
        normalized_table[(0, col_idx)].set_text_props(color="white")

        # Apply lighter color to cells in this column
        for row_idx in range(1, len(normalized_table_data) + 1):
            alpha = 0.3  # Lighter color for data cells
            rgb_color = plt.cm.colors.to_rgba(color, alpha)
            normalized_table[(row_idx, col_idx)].set_facecolor(rgb_color)

    # Add a title to the figure indicating normalization
    plt.figtext(
        0.5,
        0.95,
        "Scaling Constants Normalized to Sum to 1",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.savefig(
        os.path.join(plots_dir, "configurations_table_normalized.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 5: Heatmap of correlation between components
    if component_count > 1:
        plt.figure(figsize=(8, 6))
        component_cols = [
            COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
            for i in range(component_count)
        ]
        corr_matrix = df_components[component_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Between Validation Loss Components")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "component_correlation.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def create_2d_surface_plots(df, target_component, plots_dir):
    """
    Create 2D surface plots showing how the target component varies with pairs of parameters.

    Parameters:
        df: DataFrame containing the component values and scaling parameters
        target_component: Name of the component to visualize (e.g., "ExpL2")
        plots_dir: Directory to save the plots
    """
    scale_dims = [col for col in df.columns if col.startswith("Scale_")]
    n_dims = len(scale_dims)

    if n_dims < 2:
        return  # Need at least 2 dimensions for surface plots

    # Create plots for each pair of dimensions
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            dim1 = scale_dims[i]
            dim2 = scale_dims[j]

            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot with color representing the target component
            scatter = ax.scatter(
                df[dim1], df[dim2], c=df[target_component], cmap="viridis", s=100, alpha=0.7
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(target_component)

            # Label points with their configuration index
            for idx, row in df.iterrows():
                ax.annotate(
                    str(idx),
                    (row[dim1], row[dim2]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            # Set labels and title
            ax.set_xlabel(f"Scaling Parameter {i}")
            ax.set_ylabel(f"Scaling Parameter {j}")
            ax.set_title(f"{target_component} vs Scaling Parameters {i} and {j}")

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.6)

            # Save plot
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"{target_component}_surface_{i}_{j}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # If we have enough points, try to create a contour plot
            if len(df) >= 10:
                try:
                    # Create a figure for contour plot
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Create a grid for contour plot
                    x_unique = sorted(df[dim1].unique())
                    y_unique = sorted(df[dim2].unique())

                    if len(x_unique) > 1 and len(y_unique) > 1:
                        # Create meshgrid
                        X, Y = np.meshgrid(x_unique, y_unique)
                        Z = np.zeros_like(X)

                        # Fill Z values
                        for x_idx, x_val in enumerate(x_unique):
                            for y_idx, y_val in enumerate(y_unique):
                                # Find rows matching these x,y values
                                matches = df[(df[dim1] == x_val) & (df[dim2] == y_val)]
                                if not matches.empty:
                                    Z[y_idx, x_idx] = matches[target_component].values[0]

                        # Create contour plot
                        contour = ax.contourf(X, Y, Z, cmap="viridis", levels=15)
                        cbar = plt.colorbar(contour, ax=ax)
                        cbar.set_label(target_component)

                        # Set labels and title
                        ax.set_xlabel(f"Scaling Parameter {i}")
                        ax.set_ylabel(f"Scaling Parameter {j}")
                        ax.set_title(f"{target_component} Contour vs Parameters {i} and {j}")

                        # Save contour plot
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(plots_dir, f"{target_component}_contour_{i}_{j}.png"),
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()
                except Exception as e:
                    print(f"Could not create contour plot for params {i} and {j}: {e}")
            else:
                # Enhanced scatter plot for fewer than 10 points
                fig, ax = plt.subplots(figsize=(10, 8))

                # Create scatter plot with larger points
                scatter = ax.scatter(
                    df[dim1],
                    df[dim2],
                    c=df[target_component],
                    cmap="viridis",
                    s=200,
                    alpha=0.8,
                    edgecolors="black",
                )

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(target_component)

                # Annotate each point with its value
                for idx, row in df.iterrows():
                    ax.annotate(
                        f"{row[target_component]:.3f}",
                        (row[dim1], row[dim2]),
                        textcoords="offset points",
                        xytext=(0, -15),
                        ha="center",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    )

                    # Also annotate with the config number
                    ax.annotate(
                        f"#{idx}",
                        (row[dim1], row[dim2]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=9,
                        weight="bold",
                    )

                # Set labels and title
                ax.set_xlabel(f"Scaling Parameter {i}")
                ax.set_ylabel(f"Scaling Parameter {j}")
                ax.set_title(f"{target_component} vs Scaling Parameters {i} and {j} (Sparse Data)")

                # Add grid
                ax.grid(True, linestyle="--", alpha=0.6)

                # Save plot
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"{target_component}_annotated_scatter_{i}_{j}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()


def create_1d_slice_plots(df, target_component, plots_dir):
    """
    Create 1D slice plots showing how the target component varies with each parameter.

    Parameters:
        df: DataFrame containing the component values and scaling parameters
        target_component: Name of the component to visualize (e.g., "ExpL2")
        plots_dir: Directory to save the plots
    """
    scale_dims = [col for col in df.columns if col.startswith("Scale_")]
    n_dims = len(scale_dims)

    # Find the configuration with the minimum target component value
    best_config = df.loc[df[target_component].idxmin()]

    # Create plots for each dimension
    for i in range(n_dims):
        dim = scale_dims[i]

        # Group by this dimension, aggregating by mean for the target component
        grouped = df.groupby(dim)[target_component].agg(["mean", "min", "max", "count"])
        grouped = grouped.reset_index()

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the mean with error bars for min/max
        ax.errorbar(
            grouped[dim],
            grouped["mean"],
            yerr=[grouped["mean"] - grouped["min"], grouped["max"] - grouped["mean"]],
            fmt="o-",
            capsize=5,
            label=f"Mean {target_component} (with min/max range)",
        )

        # Highlight the best value
        ax.axvline(
            x=best_config[dim],
            color="r",
            linestyle="--",
            label=f"Best value: {best_config[dim]:.2f}",
        )

        # Add point counts as annotations
        for _, row in grouped.iterrows():
            ax.annotate(
                f"n={int(row['count'])}",
                (row[dim], row["mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Set labels and title
        ax.set_xlabel(f"Scaling Parameter {i}")
        ax.set_ylabel(f"Mean {target_component}")
        ax.set_title(f"{target_component} vs Scaling Parameter {i}")

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{target_component}_slice_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create a scatter plot showing all points
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all data points
        ax.scatter(df[dim], df[target_component], alpha=0.7)

        # Highlight the best value
        ax.axvline(
            x=best_config[dim],
            color="r",
            linestyle="--",
            label=f"Best value: {best_config[dim]:.2f}",
        )

        # Set labels and title
        ax.set_xlabel(f"Scaling Parameter {i}")
        ax.set_ylabel(target_component)
        ax.set_title(f"All {target_component} Values vs Scaling Parameter {i}")

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{target_component}_scatter_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def find_best_configurations(all_results):
    """Find and print the best configurations based on different criteria."""
    # Define component names
    component_count = len(all_results[0]["best_val_components"])

    print("\n===== Best Configurations Summary =====")

    # Best by ExpL2 (first component)
    best_exp_l2 = min(all_results, key=lambda x: float(x["best_val_components"][0]))
    print("\nBest configuration (by ExpL2 validation loss component):")
    print(f"Scaling: {best_exp_l2['forward_model_scaling']}")
    print(f"ExpL2 validation loss: {float(best_exp_l2['best_val_components'][0]):.6f}")

    print("All validation loss components:")
    for i, comp in enumerate(best_exp_l2["best_val_components"]):
        comp_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
        print(f"  {comp_name}: {float(comp):.6f}")

    # Best by sum if available
    if "sum_best_val" in all_results[0]:
        best_by_sum = min(all_results, key=lambda x: float(x["sum_best_val"]))
        print("\nBest configuration (by sum of validation loss components):")
        print(f"Scaling: {best_by_sum['forward_model_scaling']}")

        print("All validation loss components:")
        for i, comp in enumerate(best_by_sum["best_val_components"]):
            comp_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
            print(f"  {comp_name}: {float(comp):.6f}")

        print(f"Validation loss (unscaled sum): {float(best_by_sum['sum_best_val']):.6f}")
    else:
        # Calculate sum if not in the results
        best_by_calc_sum = min(
            all_results, key=lambda x: sum(float(comp) for comp in x["best_val_components"])
        )

        calc_sum = sum(float(comp) for comp in best_by_calc_sum["best_val_components"])

        print("\nBest configuration (by calculated sum of validation loss components):")
        print(f"Scaling: {best_by_calc_sum['forward_model_scaling']}")

        print("All validation loss components:")
        for i, comp in enumerate(best_by_calc_sum["best_val_components"]):
            comp_name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"Component {i}"
            print(f"  {comp_name}: {float(comp):.6f}")

        print(f"Validation loss (calculated sum): {calc_sum:.6f}")


def main_with_args(args):
    """Main function to run the reanalysis with provided arguments."""
    # Create output directory
    plots_dir = ensure_output_dir(args.output_dir)
    print(f"Plots will be saved to: {plots_dir}")

    # Load results
    all_results = load_results(args.json_file)
    print(f"Loaded {len(all_results)} results")

    # Set target component if not specified
    target_component = args.target_component or COMPONENT_NAMES[0]
    print(f"Using {target_component} as target component for analysis")

    # Generate individual plots if requested
    if args.individual_plots:
        print("Generating individual plots for each combination...")
        for i, result in enumerate(all_results):
            print(f"Plotting combination {i + 1}/{len(all_results)}")
            plot_individual_result(result, plots_dir, i)

    # Create summary plots
    print("Generating summary plots...")
    create_summary_plots(all_results, plots_dir)

    # Find and print best configurations
    find_best_configurations(all_results)

    print(f"\nAll plots have been generated and saved to {plots_dir}")


def main():
    """Parse command-line arguments and run the reanalysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reanalyze grid search results and regenerate plots."
    )
    parser.add_argument("json_file", help="Path to the JSON results file")
    parser.add_argument(
        "--output_dir", default="reanalysis", help="Directory name for output plots"
    )
    parser.add_argument(
        "--target_component",
        default=None,
        help="Target component for analysis (default: first component)",
    )
    parser.add_argument(
        "--individual_plots",
        action="store_true",
        help="Generate individual plots for each combination",
    )
    args = parser.parse_args()

    main_with_args(args)


if __name__ == "__main__":
    import sys

    # If no command-line arguments are provided (or you want to use hardcoded values),
    # use these default values
    if len(sys.argv) == 1:
        # Default values when running the script directly - MODIFY THESE AS NEEDED
        class Args:
            def __init__(self):
                # Set your defaults here
                self.json_file = "/Users/alexi/JAX-ENT/tests/_results/scaling_grid_search_20250317_235647/scaling_grid_search_20250317_235647.json"  # CHANGE THIS
                self.output_dir = "reanalysis"
                self.target_component = None  # Will use first component (ExpL2)
                self.individual_plots = False

        # Use the manually specified args
        main_with_args(Args())
    else:
        # Use command-line arguments
        main()
