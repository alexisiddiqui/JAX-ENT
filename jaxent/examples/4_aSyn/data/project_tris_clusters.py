import json
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from sklearn.mixture import GaussianMixture
import sys
import copy

# Add the directory to path to import from inertia_moments_clustering
sys.path.append(str(Path(__file__).resolve().parent))
from inertia_moments_clustering import (
    compute_inertia_ratios, compute_ctail_rg,
    compute_reference_ratios, compute_dynamic_reference_ratios,
    plot_shape_space_ctail, plot_free_energy_landscape,
    plot_free_energy_landscape_hexbin, plot_shape_indices_histograms,
    plot_cluster_shape_space, plot_cluster_hexbin, plot_ca_traces,
    plot_ca_traces_macro, plot_moments_histograms,
    plot_ctail_rg_clusters_hist, plot_ctail_rg_clusters_rows,
    plot_macro_hexbin, plot_macro_cluster_bar,
    plot_macro_cluster_pie, plot_ctail_rg_macro_rows,
    REFERENCES, DYNAMIC_REFERENCES, invert_macro_mapping
)

base_dir = Path(__file__).resolve().parent
tris_dir = base_dir / "_cluster_inertia"

print("1. Loading Tris-MD reference data and fitting GMM...")
tris_shape_axes = np.load(tris_dir / "shape_axes.npy")
with open(tris_dir / "macro_cluster_map.json", "r") as f:
    macro_cluster_map = json.load(f)
with open(tris_dir / "coarse_cluster_map.json", "r") as f:
    coarse_cluster_map = json.load(f)

# Load metadata to get the ctail threshold
with open(tris_dir / "cluster_method.json", "r") as f:
    tris_metadata = json.load(f)
resolved_threshold = tris_metadata["ctail_threshold"]
shape_sel_str = tris_metadata["shape_selection"]

# Build reverse mappings
def build_reverse_map(cluster_map):
    rev = {}
    for name, ids in cluster_map.items():
        for cluster_id in ids:
            rev[int(cluster_id)] = name
    return rev

macro_rev_map = build_reverse_map(macro_cluster_map)
coarse_rev_map = build_reverse_map(coarse_cluster_map)

# Pre-calculate references
ref_ratios = compute_reference_ratios(REFERENCES, shape_sel_str)
dynamic_ref_ratios = compute_dynamic_reference_ratios(DYNAMIC_REFERENCES, shape_sel_str)

import joblib

# Load GMM exactly as it was saved for Tris-MD
gmm_path = tris_dir / "gmm_model.pkl"
if not gmm_path.exists():
    raise FileNotFoundError(f"{gmm_path} not found. Please ensure inertia_moments_clustering.py was run recently to generate it.")
gmm = joblib.load(gmm_path)

def process_ensemble(name, out_dir_name, universes_args):
    print(f"\n2. Processing {name}...")
    out_dir = base_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    u = mda.Universe(*universes_args)
    print(f"Loaded {name}: {len(u.trajectory)} frames, {u.atoms.n_atoms} atoms")
    
    x_ratio, y_ratio, I1, I2, I3 = compute_inertia_ratios(u, shape_sel_str)
    ctail_rg = compute_ctail_rg(u, tris_metadata["ctail_selection"])
    
    shape_axes = np.column_stack([x_ratio, y_ratio])
    
    print(f"Predicting clusters for {name} using Tris-MD GMM...")
    cluster_labels = gmm.predict(shape_axes).astype(int)
    
    macro_labels = np.full(cluster_labels.shape, "Unassigned", dtype=object)
    coarse_labels = np.full(cluster_labels.shape, "Unassigned", dtype=object)
    
    for i, label in enumerate(cluster_labels):
        macro_labels[i] = macro_rev_map.get(label, "Unassigned")
        coarse_labels[i] = coarse_rev_map.get(label, "Unassigned")
        
    np.save(out_dir / "shape_axes.npy", shape_axes)
    np.save(out_dir / "ctail_rg.npy", ctail_rg)
    np.save(out_dir / "cluster_labels.npy", cluster_labels)
    np.save(out_dir / "macro_cluster_labels.npy", macro_labels)
    np.save(out_dir / "coarse_cluster_labels.npy", coarse_labels)
    
    # Save the centers and metadata too just in case downstream scripts need them
    centers_xy = gmm.means_
    np.save(out_dir / "cluster_centers.npy", centers_xy)
    
    with open(out_dir / "macro_cluster_map.json", "w") as f:
        json.dump(macro_cluster_map, f, indent=2)
    with open(out_dir / "coarse_cluster_map.json", "w") as f:
        json.dump(coarse_cluster_map, f, indent=2)
        
    metadata = copy.deepcopy(tris_metadata)
    metadata["projected_from"] = "Tris-MD"
    with open(out_dir / "cluster_method.json", "w") as f2:
        json.dump(metadata, f2, indent=2)
        
    print(f"Generating plots for {name}...")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    method = "gmm (Projected from Tris-MD)"
    
    plot_shape_space_ctail(
        x_ratio, y_ratio, ctail_rg, plots_dir,
        ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
    )
    plot_free_energy_landscape(
        x_ratio, y_ratio, plots_dir,
        ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
    )
    plot_free_energy_landscape_hexbin(
        x_ratio, y_ratio, plots_dir,
        ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
    )
    
    w_rod = y_ratio - x_ratio
    w_sphere = x_ratio
    w_disk = 1.0 - y_ratio
    plot_shape_indices_histograms(w_rod, w_sphere, w_disk, plots_dir)
    
    plot_cluster_shape_space(
        x_ratio, y_ratio, cluster_labels, centers_xy, method, plots_dir,
        ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
    )
    plot_cluster_hexbin(
        x_ratio, y_ratio, cluster_labels, method, plots_dir,
        ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
    )
    plot_ca_traces(u, shape_sel_str, cluster_labels, shape_axes, centers_xy, method, plots_dir)
    plot_ca_traces_macro(
        u, shape_sel_str, cluster_labels, coarse_rev_map,
        shape_axes, centers_xy, method, plots_dir
    )
    plot_moments_histograms(I1, I2, I3, plots_dir)
    plot_ctail_rg_clusters_hist(ctail_rg, cluster_labels, plots_dir, ctail_threshold=resolved_threshold)
    plot_ctail_rg_clusters_rows(ctail_rg, cluster_labels, plots_dir, ctail_threshold=resolved_threshold)
    
    plot_macro_hexbin(
        x_ratio, y_ratio, coarse_labels, plots_dir,
        ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
    )
    plot_macro_cluster_bar(
        coarse_labels, ctail_rg, resolved_threshold,
        coarse_cluster_map, plots_dir,
    )
    plot_macro_cluster_pie(
        coarse_labels, ctail_rg, resolved_threshold,
        coarse_cluster_map, plots_dir,
    )
    plot_ctail_rg_macro_rows(
        ctail_rg, coarse_labels, coarse_cluster_map, plots_dir,
        ctail_threshold=resolved_threshold,
    )
            
    print(f"Done with {name}.")

ensembles = [
    (
        "Control-MD", 
        "_cluster_inertia_control", 
        [str(base_dir / "_aSyn/control_MD/md.gro.pdb"), str(base_dir / "_aSyn/control_MD/control_all_combined.xtc")]
    ),
    (
        "AF2-MSAss", 
        "_cluster_inertia_af2", 
        [str(base_dir / "_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb"), str(base_dir / "_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc")]
    ),
    (
        "Shaw-MD", 
        "_cluster_inertia_shaw", 
        [str(base_dir / "_aSyn/a99sb.pdb"), str(base_dir / "_aSyn/a99sb.pdb"), str(base_dir / "_aSyn/c36m.pdb"), str(base_dir / "_aSyn/c22star.pdb")]
    )
]

for name, out_dir_name, args in ensembles:
    process_ensemble(name, out_dir_name, args)

print("\nAll projections complete.")

