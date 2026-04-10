import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

def make_colorscale(rgb_stops: list[tuple], name: str = "custom") -> mcolors.LinearSegmentedColormap:
    """
    Build a matplotlib colormap from a list of RGB tuples (values 0-255).
    Interpolation is piecewise linear in RGB space, matching PyMOL's approach.
    
    Args:
        rgb_stops: list of (R, G, B) tuples with values in range 0-255
        name: name for the colormap
    
    Returns:
        A matplotlib LinearSegmentedColormap
    """
    # Normalise to 0-1
    colors = [(r / 255, g / 255, b / 255) for r, g, b in rgb_stops]
    
    # Evenly space the stops across [0, 1]
    positions = np.linspace(0, 1, len(colors))
    
    # Build the segmentdata dict that LinearSegmentedColormap expects:
    # each channel is a list of (position, value_left, value_right) tuples
    segmentdata = {"red": [], "green": [], "blue": []}
    for pos, (r, g, b) in zip(positions, colors):
        segmentdata["red"].append((pos, r, r))
        segmentdata["green"].append((pos, g, g))
        segmentdata["blue"].append((pos, b, b))
    
    return mcolors.LinearSegmentedColormap(name, segmentdata)


def plot_colorscale(cmap: mcolors.LinearSegmentedColormap, title: str = "Color scale"):
    """Display the colormap as a horizontal gradient bar."""
    fig, ax = plt.subplots(figsize=(8, 1.5))
    fig.subplots_adjust(bottom=0.4)
    
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label(title)
    plt.tight_layout()
    plt.show()


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # PyMOL-style cyan magenta black
    bwr_stops = [
        (0,   255, 255),   # cyan
        (255, 0,   255),   # magenta
        (0,   0,   0  ),   # black
    ]






    # A multi-stop rainbow-ish ramp
    rainbow_stops = [
        (0,   0,   255),   # blue
        (0,   255, 255),   # cyan
        (0,   255, 0  ),   # green
        (255, 255, 0  ),   # yellow
        (255, 0,   0  ),   # red
    ]

    # ── Build and plot ───────────────────────────────────────────────────────

    bwr_cmap     = make_colorscale(bwr_stops,     name="bwr_pymol")
    rainbow_cmap = make_colorscale(rainbow_stops, name="rainbow_pymol")

    plot_colorscale(bwr_cmap,     title="Cyan → Magenta → Black")
    plot_colorscale(rainbow_cmap, title="Rainbow (5-stop)")

    # ── Use with imshow / scatter etc. ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, cmap, label in zip(axes,
                                [bwr_cmap, rainbow_cmap],
                                ["BWR", "Rainbow"]):
        data = np.random.rand(20, 20)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
    plt.tight_layout()
    plt.show()