import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# ── PyMOL Named Colors ────────────────────────────────────────────────────────

PYMOL_COLORS = {
    "actinium": (0.439215686, 0.670588235, 0.980392157),
    "aluminum": (0.749019608, 0.650980392, 0.650980392),
    "americium": (0.329411765, 0.360784314, 0.949019608),
    "antimony": (0.619607843, 0.388235294, 0.709803922),
    "aquamarine": (0.5, 1.0, 1.0),
    "argon": (0.501960784, 0.819607843, 0.890196078),
    "arsenic": (0.741176471, 0.501960784, 0.890196078),
    "astatine": (0.458823529, 0.309803922, 0.270588235),
    "barium": (0.0, 0.788235294, 0.0),
    "berkelium": (0.541176471, 0.309803922, 0.890196078),
    "beryllium": (0.760784314, 1.0, 0.0),
    "bismuth": (0.619607843, 0.309803922, 0.709803922),
    "black": (0.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "bluewhite": (0.85, 0.85, 1.0),
    "bohrium": (0.878431373, 0.0, 0.219607843),
    "boron": (1.0, 0.709803922, 0.709803922),
    "br0": (0.1, 0.1, 1.0),
    "br1": (0.2, 0.1, 0.9),
    "br2": (0.3, 0.1, 0.8),
    "br3": (0.4, 0.1, 0.7),
    "br4": (0.5, 0.1, 0.6),
    "br5": (0.6, 0.1, 0.5),
    "br6": (0.7, 0.1, 0.4),
    "br7": (0.8, 0.1, 0.3),
    "br8": (0.9, 0.1, 0.2),
    "br9": (1.0, 0.1, 0.1),
    "brightorange": (1.0, 0.7, 0.2),
    "bromine": (0.650980392, 0.160784314, 0.160784314),
    "brown": (0.65, 0.32, 0.17),
    "cadmium": (1.0, 0.850980392, 0.560784314),
    "calcium": (0.239215686, 1.0, 0.0),
    "californium": (0.631372549, 0.211764706, 0.831372549),
    "carbon": (0.2, 1.0, 0.2),
    "cerium": (1.0, 1.0, 0.780392157),
    "cesium": (0.341176471, 0.090196078, 0.560784314),
    "chartreuse": (0.5, 1.0, 0.0),
    "chlorine": (0.121568627, 0.941176471, 0.121568627),
    "chocolate": (0.555, 0.222, 0.111),
    "chromium": (0.541176471, 0.6, 0.780392157),
    "cobalt": (0.941176471, 0.564705882, 0.62745098),
    "copper": (0.784313725, 0.501960784, 0.2),
    "curium": (0.470588235, 0.360784314, 0.890196078),
    "cyan": (0.0, 1.0, 1.0),
    "darksalmon": (0.73, 0.55, 0.52),
    "dash": (1.0, 1.0, 0.0),
    "deepblue": (0.25, 0.25, 0.65),
    "deepolive": (0.6, 0.6, 0.1),
    "deeppurple": (0.6, 0.1, 0.6),
    "deepsalmon": (1.0, 0.42, 0.42),
    "deepteal": (0.1, 0.6, 0.6),
    "density": (0.1, 0.1, 0.6),
    "deuterium": (0.9, 0.9, 0.9),
    "dirtyviolet": (0.7, 0.5, 0.5),
    "dubnium": (0.819607843, 0.0, 0.309803922),
    "dysprosium": (0.121568627, 1.0, 0.780392157),
    "einsteinium": (0.701960784, 0.121568627, 0.831372549),
    "erbium": (0.0, 0.901960784, 0.458823529),
    "europium": (0.380392157, 1.0, 0.780392157),
    "fermium": (0.701960784, 0.121568627, 0.729411765),
    "firebrick": (0.698, 0.13, 0.13),
    "fluorine": (0.701960784, 1.0, 1.0),
    "forest": (0.2, 0.6, 0.2),
    "francium": (0.258823529, 0.0, 0.4),
    "gadolinium": (0.270588235, 1.0, 0.780392157),
    "gallium": (0.760784314, 0.560784314, 0.560784314),
    "germanium": (0.4, 0.560784314, 0.560784314),
    "gold": (1.0, 0.819607843, 0.137254902),
    "gray": (0.5, 0.5, 0.5),
    "green": (0.0, 1.0, 0.0),
    "greencyan": (0.25, 1.0, 0.75),
    "grey": (0.5, 0.5, 0.5),
    "grey10": (0.1, 0.1, 0.1),
    "grey20": (0.2, 0.2, 0.2),
    "grey30": (0.3, 0.3, 0.3),
    "grey40": (0.4, 0.4, 0.4),
    "grey50": (0.5, 0.5, 0.5),
    "grey60": (0.6, 0.6, 0.6),
    "grey70": (0.7, 0.7, 0.7),
    "grey80": (0.8, 0.8, 0.8),
    "grey90": (0.9, 0.9, 0.9),
    "hafnium": (0.301960784, 0.760784314, 1.0),
    "hassium": (0.901960784, 0.0, 0.180392157),
    "helium": (0.850980392, 1.0, 1.0),
    "holmium": (0.0, 1.0, 0.611764706),
    "hotpink": (1.0, 0.0, 0.5),
    "hydrogen": (0.9, 0.9, 0.9),
    "indium": (0.650980392, 0.458823529, 0.450980392),
    "iodine": (0.580392157, 0.0, 0.580392157),
    "iridium": (0.090196078, 0.329411765, 0.529411765),
    "iron": (0.878431373, 0.4, 0.2),
    "krypton": (0.360784314, 0.721568627, 0.819607843),
    "lanthanum": (0.439215686, 0.831372549, 1.0),
    "lawrencium": (0.780392157, 0.0, 0.4),
    "lead": (0.341176471, 0.349019608, 0.380392157),
    "lightblue": (0.75, 0.75, 1.0),
    "lightmagenta": (1.0, 0.2, 0.8),
    "lightorange": (1.0, 0.8, 0.5),
    "lightpink": (1.0, 0.75, 0.87),
    "lightteal": (0.4, 0.7, 0.7),
    "lime": (0.5, 1.0, 0.5),
    "limegreen": (0.0, 1.0, 0.5),
    "limon": (0.75, 1.0, 0.25),
    "lithium": (0.8, 0.501960784, 1.0),
    "lutetium": (0.0, 0.670588235, 0.141176471),
    "magenta": (1.0, 0.0, 1.0),
    "magnesium": (0.541176471, 1.0, 0.0),
    "manganese": (0.611764706, 0.478431373, 0.780392157),
    "marine": (0.0, 0.5, 1.0),
    "meitnerium": (0.921568627, 0.0, 0.149019608),
    "mendelevium": (0.701960784, 0.050980392, 0.650980392),
    "mercury": (0.721568627, 0.721568627, 0.815686275),
    "molybdenum": (0.329411765, 0.709803922, 0.709803922),
    "neodymium": (0.780392157, 1.0, 0.780392157),
    "neon": (0.701960784, 0.890196078, 0.960784314),
    "neptunium": (0.0, 0.501960784, 1.0),
    "nickel": (0.31372549, 0.815686275, 0.31372549),
    "niobium": (0.450980392, 0.760784314, 0.788235294),
    "nitrogen": (0.2, 0.2, 1.0),
    "nobelium": (0.741176471, 0.050980392, 0.529411765),
    "olive": (0.77, 0.7, 0.0),
    "orange": (1.0, 0.5, 0.0),
    "osmium": (0.149019608, 0.4, 0.588235294),
    "oxygen": (1.0, 0.3, 0.3),
    "palecyan": (0.8, 1.0, 1.0),
    "palegreen": (0.65, 0.9, 0.65),
    "paleyellow": (1.0, 1.0, 0.5),
    "palladium": (0.0, 0.411764706, 0.521568627),
    "phosphorus": (1.0, 0.501960784, 0.0),
    "pink": (1.0, 0.65, 0.85),
    "platinum": (0.815686275, 0.815686275, 0.878431373),
    "plutonium": (0.0, 0.419607843, 1.0),
    "polonium": (0.670588235, 0.360784314, 0.0),
    "potassium": (0.560784314, 0.250980392, 0.831372549),
    "praseodymium": (0.850980392, 1.0, 0.780392157),
    "promethium": (0.639215686, 1.0, 0.780392157),
    "protactinium": (0.0, 0.631372549, 1.0),
    "purple": (0.75, 0.0, 0.75),
    "purpleblue": (0.5, 0.0, 1.0),
    "radium": (0.0, 0.490196078, 0.0),
    "radon": (0.258823529, 0.509803922, 0.588235294),
    "raspberry": (0.7, 0.3, 0.4),
    "red": (1.0, 0.0, 0.0),
    "rhenium": (0.149019608, 0.490196078, 0.670588235),
    "rhodium": (0.039215686, 0.490196078, 0.549019608),
    "rubidium": (0.439215686, 0.180392157, 0.690196078),
    "ruby": (0.6, 0.2, 0.2),
    "ruthenium": (0.141176471, 0.560784314, 0.560784314),
    "rutherfordium": (0.8, 0.0, 0.349019608),
    "salmon": (1.0, 0.6, 0.6),
    "samarium": (0.560784314, 1.0, 0.780392157),
    "sand": (0.72, 0.55, 0.3),
    "scandium": (0.901960784, 0.901960784, 0.901960784),
    "seaborgium": (0.850980392, 0.0, 0.270588235),
    "selenium": (1.0, 0.631372549, 0.0),
    "silicon": (0.941176471, 0.784313725, 0.62745098),
    "silver": (0.752941176, 0.752941176, 0.752941176),
    "skyblue": (0.2, 0.5, 0.8),
    "slate": (0.5, 0.5, 1.0),
    "smudge": (0.55, 0.7, 0.4),
    "sodium": (0.670588235, 0.360784314, 0.949019608),
    "splitpea": (0.52, 0.75, 0.0),
    "strontium": (0.0, 1.0, 0.0),
    "sulfur": (0.9, 0.775, 0.25),
    "tantalum": (0.301960784, 0.650980392, 1.0),
    "teal": (0.0, 0.75, 0.75),
    "technetium": (0.231372549, 0.619607843, 0.619607843),
    "tellurium": (0.831372549, 0.478431373, 0.0),
    "terbium": (0.188235294, 1.0, 0.780392157),
    "thallium": (0.650980392, 0.329411765, 0.301960784),
    "thorium": (0.0, 0.729411765, 1.0),
    "thulium": (0.0, 0.831372549, 0.321568627),
    "tin": (0.4, 0.501960784, 0.501960784),
    "titanium": (0.749019608, 0.760784314, 0.780392157),
    "tungsten": (0.129411765, 0.580392157, 0.839215686),
    "tv_blue": (0.3, 0.3, 1.0),
    "tv_green": (0.2, 1.0, 0.2),
    "tv_orange": (1.0, 0.55, 0.15),
    "tv_red": (1.0, 0.2, 0.2),
    "tv_yellow": (1.0, 1.0, 0.2),
    "uranium": (0.0, 0.560784314, 1.0),
    "vanadium": (0.650980392, 0.650980392, 0.670588235),
    "violet": (1.0, 0.5, 1.0),
    "violetpurple": (0.55, 0.25, 0.6),
    "warmpink": (0.85, 0.2, 0.5),
    "wheat": (0.99, 0.82, 0.65),
    "white": (1.0, 1.0, 1.0),
    "xenon": (0.258823529, 0.619607843, 0.690196078),
    "yellow": (1.0, 1.0, 0.0),
    "yelloworange": (1.0, 0.87, 0.37),
    "ytterbium": (0.0, 0.749019608, 0.219607843),
    "yttrium": (0.580392157, 1.0, 1.0),
    "zinc": (0.490196078, 0.501960784, 0.690196078),
    "zirconium": (0.580392157, 0.878431373, 0.878431373),
}

# ── Utility Functions ────────────────────────────────────────────────────────

def get_pymol_color(name: str) -> tuple[float, float, float]:
    """Retrieve PyMOL RGB tuple (0-1.0) by name."""
    return PYMOL_COLORS.get(name.lower(), (1.0, 1.0, 1.0))

def make_colorscale(rgb_stops: list, name: str = "custom") -> mcolors.LinearSegmentedColormap:
    """
    Build a matplotlib colormap from a list of RGB tuples or PyMOL color names.
    Stops can be:
        - (R, G, B) tuples in range 0-255 or 0-1.0
        - String names matching PYMOL_COLORS (e.g. "salmon", "marine")
    
    Args:
        rgb_stops: list of color stops
        name: name for the colormap
    
    Returns:
        A matplotlib LinearSegmentedColormap
    """
    colors = []
    for stop in rgb_stops:
        if isinstance(stop, str):
            # Try PyMOL colors first, then matplotlib
            if stop.lower() in PYMOL_COLORS:
                colors.append(PYMOL_COLORS[stop.lower()])
            else:
                try:
                    colors.append(mcolors.to_rgb(stop))
                except ValueError:
                    print(f"Warning: Color '{stop}' not found. Defaulting to white.")
                    colors.append((1.0, 1.0, 1.0))
        else:
            # Detect 0-255 vs 0-1.0
            if any(c > 1.1 for c in stop): # Use 1.1 to be safe with rounding
                colors.append(tuple(c / 255.0 for c in stop))
            else:
                colors.append(stop)
    
    # Evenly space the stops across [0, 1]
    positions = np.linspace(0, 1, len(colors))
    
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

    # 1. Using named PyMOL colors
    hdx_stops = ["white", "greencyan", "teal", "francium", "black"]
    hdx_cmap = make_colorscale(hdx_stops, name="hdx_pymol")

    # 2. Mixing names and RGB tuples
    mixed_stops = [
        "white", 
        "grey", # grey
        "purpleblue"
    ]
    mixed_cmap = make_colorscale(mixed_stops, name="mixed_pymol")

    # ── Plot ───────────────────────────────────────────────────────

    plot_colorscale(hdx_cmap,   title="White → Cyan → Blue → Purple → Purple → Black")
    plot_colorscale(mixed_cmap, title="Mixed: Firebrick → Grey → Forest")

    # ── Visualise on data ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, cmap, label in zip(axes,
                                [hdx_cmap, mixed_cmap],
                                ["HDX (Named)", "Mixed"]):
        data = np.random.rand(20, 20)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
    plt.tight_layout()
    plt.show()