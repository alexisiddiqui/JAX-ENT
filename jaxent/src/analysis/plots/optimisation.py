import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_total_losses(opt_history):
    """
    Plot training and validation loss components over optimization steps.
    """
    steps = range(len(opt_history.states))
    train_losses = [state.losses.total_train_loss for state in opt_history.states]
    val_losses = [state.losses.total_val_loss for state in opt_history.states]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", marker="o", markersize=3)
    plt.plot(steps, val_losses, label="Validation Loss", marker="o", markersize=3)

    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.yscale("log")
    plt.tight_layout()

    return plt.gcf()


def plot_loss_components(opt_history):
    """
    Plot training and validation loss components over optimization steps,
    showing both total loss and individual components on separate subplots.
    """
    steps = range(len(opt_history.states))

    # Get individual loss components
    train_components = np.array([state.losses.train_losses for state in opt_history.states])
    val_components = np.array([state.losses.val_losses for state in opt_history.states])
    scaled_train = np.array([state.losses.scaled_train_losses for state in opt_history.states])
    scaled_val = np.array([state.losses.scaled_val_losses for state in opt_history.states])
    total_train = np.array([state.losses.total_train_loss for state in opt_history.states])
    total_val = np.array([state.losses.total_val_loss for state in opt_history.states])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot training losses
    colors = plt.cm.Set2(np.linspace(0, 1, train_components.shape[1]))

    ax1.set_title("Training Loss Components")
    # Plot individual components
    for i, color in enumerate(colors):
        ax1.plot(
            steps,
            train_components[:, i],
            label=f"Component {i + 1}",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax1.plot(
            steps, scaled_train[:, i], label=f"Scaled Component {i + 1}", color=color, alpha=1.0
        )
    # Plot total loss
    ax1.plot(steps, total_train, label="Total Loss", color="black", linewidth=2)
    # ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Loss")

    # Plot validation losses
    ax2.set_title("Validation Loss Components")
    # Plot individual components
    for i, color in enumerate(colors):
        ax2.plot(
            steps,
            val_components[:, i],
            label=f"Component {i + 1}",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax2.plot(steps, scaled_val[:, i], label=f"Scaled Component {i + 1}", color=color, alpha=1.0)
    # Plot total loss
    ax2.plot(steps, total_val, label="Total Loss", color="black", linewidth=2)
    # ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Loss")

    plt.tight_layout()
    return fig


def plot_frame_weights_heatmap(opt_history):
    """
    Create a heatmap showing the evolution of frame weights during optimization.
    """
    # Extract frame weights from each state
    frame_weights = [state.params.frame_weights for state in opt_history.states]
    frame_mask = [state.params.frame_mask for state in opt_history.states]

    # Apply frame mask
    frame_weights = [weights * mask for weights, mask in zip(frame_weights, frame_mask)]

    # Convert to 2D array (steps × frames)
    weights_array = jnp.vstack(frame_weights)

    plt.figure(figsize=(12, 6))
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_under("black")  # Set color for values under the minimum (i.e., zero)

    sns.heatmap(
        weights_array,
        cmap=cmap,
        xticklabels=100,
        yticklabels=50,
        cbar_kws={"label": "Frame Weight"},
        vmin=1e-9,  # Ensure black is shown for 0 by setting a minimum value slightly above zero
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Optimization Step")
    plt.title("Frame Weights Evolution During Optimization")
    plt.tight_layout()

    return plt.gcf()


def plot_split_visualization(train_data, val_data, exp_data, peptide=False):
    """
    Create a visualization of the train/validation split along residue indices.

    Parameters:
    -----------
    train_data : list
        List of training data points.
    val_data : list
        List of validation data points.
    exp_data : list
        List of all experimental data points.
    peptide : bool, default=False
        If True, visualize peptide fragments with overlaps. If False, visualize individual residues.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for saving or further customization.
    """
    fig = plt.figure(figsize=(14, 8))

    if peptide:
        # Extract segment information
        segments = []
        for d in train_data:
            start = d.top.residue_start
            end = d.top.residue_end if d.top.residue_end is not None else start
            frag_idx = d.top.fragment_index if d.top.fragment_index is not None else 0
            segments.append((start, end, frag_idx, "train"))

        for d in val_data:
            start = d.top.residue_start
            end = d.top.residue_end if d.top.residue_end is not None else start
            frag_idx = d.top.fragment_index if d.top.fragment_index is not None else 0
            segments.append((start, end, frag_idx, "val"))

        # Sort by start position and then fragment index
        segments.sort(key=lambda x: (x[0], x[2]))

        # Function to check if two segments overlap
        def overlaps(seg1, seg2):
            start1, end1, _, _ = seg1
            start2, end2, _, _ = seg2
            return start1 <= end2 and start2 <= end1

        # Group overlapping segments together
        overlap_groups = []
        processed = set()

        for i, segment in enumerate(segments):
            if i in processed:
                continue

            # Start a new group with this segment
            group = [i]
            processed.add(i)

            # Find all segments that overlap with any in this group
            j = 0
            while j < len(group):
                seg_idx = group[j]
                for k, other_segment in enumerate(segments):
                    if k not in processed and overlaps(segments[seg_idx], other_segment):
                        group.append(k)
                        processed.add(k)
                j += 1

            # Add the segment indices in this group
            overlap_groups.append(group)

        # Assign y-positions to segments
        y_positions = {}
        y_counter = 0

        for group in overlap_groups:
            if len(group) == 1:
                # Single segment (no overlaps)
                y_positions[group[0]] = y_counter
                y_counter += 1
            else:
                # Multiple overlapping segments, sort by fragment index
                sorted_group = sorted(group, key=lambda i: segments[i][2])
                for i, seg_idx in enumerate(sorted_group):
                    y_positions[seg_idx] = y_counter + i
                y_counter += len(sorted_group)

        # Plot segments
        train_plotted = False
        val_plotted = False

        for i, (start, end, frag_idx, split) in enumerate(segments):
            y_pos = y_positions[i]

            if split == "train":
                color = "blue"
                label = "Training" if not train_plotted else None
                train_plotted = True
            else:
                color = "orange"
                label = "Validation" if not val_plotted else None
                val_plotted = True

            plt.barh(
                y_pos, end - start + 1, left=start, height=0.8, color=color, alpha=0.6, label=label
            )

            # Add fragment index as text
            if frag_idx != 0:
                plt.text(
                    start + (end - start) / 2,
                    y_pos,
                    f"{frag_idx}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

        # Hide y-ticks for cleaner appearance since they're just positional
        plt.yticks([])

    else:
        # For individual residues
        train_residues = [d.top.residue_start for d in train_data]
        val_residues = [d.top.residue_start for d in val_data]
        all_residues = [d.top.residue_start for d in exp_data]

        # Get residue range
        min_res = min(all_residues)
        max_res = max(all_residues)
        residue_range = np.arange(min_res, max_res + 1)

        # Create binary masks
        train_mask = np.isin(residue_range, train_residues)
        val_mask = np.isin(residue_range, val_residues)

        # Create a 2-row heatmap
        heatmap_data = np.zeros((2, len(residue_range)))
        heatmap_data[1, train_mask] = 1  # Training row (top)
        heatmap_data[0, val_mask] = 1  # Validation row (bottom)

        plt.imshow(
            heatmap_data,
            aspect="auto",
            cmap="plasma",
            alpha=0.7,
            extent=(min_res - 0.5, max_res + 0.5, -0.5, 1.5),
        )

        plt.yticks([0, 1], ["Validation", "Training"])

    # Common formatting
    plt.xlabel("Residue Index")
    plt.title("Train/Validation Split by Residue")
    plt.grid(True, alpha=0.3)

    # Handle legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    return fig
