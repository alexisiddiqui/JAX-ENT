import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# globally set axes/tick/legend font‐sizes
mpl.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 12,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "font.size": 24,  # default for all text (fallback)
    }
)
# Define the data
data = {
    "Method": ["HDXer", "HDXer", "jaxENT", "jaxENT"],
    "Experiment": ["RW-only", "RW+BV", "RW-only", "RW+BV"],
    "BPTI": [3856, 32, 144, 144],
    "LXR": [853, 6.8, 135, 135],
    "HOIP": [540, 3, 116, 116],
}

df = pd.DataFrame(data)

# Create a figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Set the bar width
bar_width = 0.35

# Define proteins and positions
proteins = ["BPTI", "LXR", "HOIP"]
x_pos = np.arange(len(proteins))

# Define colors for each experiment
colors = {"RW-only": "chartreuse", "RW+BV": "khaki"}

# Plot HDXer data in the left panel
hdxer_data = df[df["Method"] == "HDXer"]

for i, experiment in enumerate(["RW-only", "RW+BV"]):
    exp_data = hdxer_data[hdxer_data["Experiment"] == experiment]
    if not exp_data.empty:
        values = [exp_data.iloc[0][protein] for protein in proteins]
        ax1.bar(
            x_pos + i * bar_width,
            values,
            width=bar_width,
            label=experiment,
            color=colors[experiment],
        )

# Plot jaxENT data in the right panel
jaxent_data = df[df["Method"] == "jaxENT"]

for i, experiment in enumerate(["RW-only", "RW+BV"]):
    exp_data = jaxent_data[jaxent_data["Experiment"] == experiment]
    if not exp_data.empty:
        values = [exp_data.iloc[0][protein] for protein in proteins]
        ax2.bar(
            x_pos + i * bar_width,
            values,
            width=bar_width,
            label=experiment,
            color=colors[experiment],
        )

# Set log scale on y-axis for both panels
ax1.set_yscale("log")
ax2.set_yscale("log")

# Set labels and title for both panels
ax1.set_xlabel("Protein")
ax1.set_ylabel("Iteration speed (it/s)")
ax1.set_title("HDXer")
ax1.set_xticks(x_pos + bar_width / 2)
ax1.set_xticklabels(proteins)
ax1.legend()

ax2.set_xlabel("Protein")
ax2.set_title("jaxENT")
ax2.set_xticks(x_pos + bar_width / 2)
ax2.set_xticklabels(proteins)
ax2.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("iteration_speed_comparison.png", dpi=300)
plt.show()
