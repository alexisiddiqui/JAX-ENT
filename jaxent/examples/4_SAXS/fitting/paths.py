"""Centralized paths for CaM pulldown fitting and analysis."""

from pathlib import Path

FITTING_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = FITTING_DIR.parent
DATA_DIR = EXAMPLE_DIR / "data"

# --- Experimental Data ---
EXPERIMENTAL_DATA = {
    "CaM+CDZ": DATA_DIR / "_CaM" / "raw_data" / "SASDNY3" / "experimental_data" / "SASDNY3.dat",
    "CaM-CDZ": DATA_DIR / "_CaM" / "raw_data" / "SASDNX3" / "experimental_data" / "SASDNX3.dat",
}

# --- SAXS Support ---
SAXS_FEATURES_PATH = DATA_DIR / "_filtered_SAXS_features" / "SAXS_curve_input_features.npz"

# --- HDX Support ---
HDX_FEATURES_DIR = DATA_DIR / "_filtered_HDX_features"
HDX_FEATURES_PATH = HDX_FEATURES_DIR / "BV_features.npz"
HDX_TOPOLOGY_PATH = HDX_FEATURES_DIR / "topology_BV_features.json"

# --- Metadata ---
FRAME_ORDERING_PATH = (
    EXAMPLE_DIR
    / "ensemble_generation"
    / "neuralplexer"
    / "collected_structures"
    / "frame_ordering.csv"
)



FRAME_ORDERING_PATH = DATA_DIR / "_filtered_SAXS_features" / "frame_ordering_filtered.csv"