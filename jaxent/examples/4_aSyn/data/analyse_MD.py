"""
Analyses the MD ensemble - similar to analyse_aSyn_ensemble.py


Performs per replicate analysis of the MD ensemble over timesteps, for now not doing PCA.

Analyses are organised per starting structure (Rod, Hairpin, Compact/Coil) per panel and replicates are plotted within each panel.

Analyses:
- RadGyr
- n_Tris bound (TR1/TR0 total and TR1/TR0 ratio as separate features)
- N-C contacts (residues: 1-60 and 95-140)


1D line plots for each replicate over timesteps.
2D density plots for each combination of features (e.g. RadGyr vs n_Tris bound - excluding ratio).


"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", message=".*CRYST1.*")
warnings.filterwarnings("ignore", message=".*Failed to guess.*")

# ============================================================================
# Constants
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

STRUCTURES = ["rod", "hairpin", "coil"]
STRUCTURE_LABELS = {"rod": "Rod", "hairpin": "Hairpin", "coil": "Coil"}
N_REPLICATES = 3

CUTOFF_TRIS = 5.0    # Å — heavy-atom distance cutoff defining "bound" Tris
CUTOFF_NC = 8.0      # Å — CA–CA contact cutoff for N–C contacts
N_TERM_RANGE = range(1, 61)       # residues 1–60: N-head
MAIN_CHAIN_RANGE = range(1, 115)  # residues 1–114: Main chain (1:114)
NAC_RANGE = range(61, 96)         # residues 61–95: NAC region
C_TERM_RANGE = range(115, 141)    # residues 115–140: C-tail (acidic tip)
STRIDE = 10          # read every Nth frame (~10 ps effective timestep at 1 ps output)

REP_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
REP_ALPHA = 0.65
LINE_WIDTH = 0.7

STRUCTURE_COLORS = {
    "rod": "#E41A1C",
    "hairpin": "#377EB8",
    "coil": "#4DAF4A",
}

METRIC_LABELS = {
    "radgyr": "Radius of Gyration (Å)",
    "radgyr_n_term": "N-term Radius of Gyration (Å)",
    "radgyr_main_chain": "Main-chain Radius of Gyration (Å)",
    "radgyr_nac": "NAC Radius of Gyration (Å)",
    "radgyr_c_term": "C-term Radius of Gyration (Å)",
    "n_tris_total": "n Tris Bound (TR0 + TR1)",
    "tr1_tr0_ratio": "TR1 / TR0 Ratio",
    "nc_contacts": "N–C Contacts",
}

# Feature pairs for 2D density (ratio excluded per docstring)
DENSITY_PAIRS = [
    ("radgyr", "n_tris_total"),
    ("radgyr", "nc_contacts"),
    ("n_tris_total", "nc_contacts"),
    ("radgyr_n_term", "radgyr"),
    ("radgyr_c_term", "radgyr"),
    ("radgyr_nac", "radgyr"),
    ("radgyr_main_chain", "radgyr"),
    
]


# ============================================================================
# Style
# ============================================================================


def set_publication_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================================
# Computation — single trajectory pass per replicate
# ============================================================================


def load_replicate(
    structure: str,
    rep: int,
    data_dir: Path,
    top_pattern: str,
    traj_pattern: str,
) -> dict:
    """Compute all metrics for one trajectory in a single pass.

    Returns dict with keys: time (ns), radgyr, n_tris_total, tr1_tr0_ratio, nc_contacts
    """
    top = data_dir / top_pattern.format(structure=structure)
    traj = data_dir / traj_pattern.format(structure=structure, rep=rep)

    if not top.exists():
        raise FileNotFoundError(f"Topology not found: {top}")
    if not traj.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj}")

    print(f"  Loading {structure} rep{rep} ...")
    u = mda.Universe(str(top), str(traj))

    # Pre-select atom groups (evaluated once, updated in-loop via positions)
    protein = u.select_atoms("protein")
    tr0_atoms = u.select_atoms("resname TR0")
    tr1_atoms = u.select_atoms("resname TR1")
    n_term_ca = u.select_atoms(f"name CA and resid {' '.join(map(str, N_TERM_RANGE))}")
    main_chain_ca = u.select_atoms(f"name CA and resid {' '.join(map(str, MAIN_CHAIN_RANGE))}")
    nac_ca = u.select_atoms(f"name CA and resid {' '.join(map(str, NAC_RANGE))}")
    c_term_ca = u.select_atoms(f"name CA and resid {' '.join(map(str, C_TERM_RANGE))}")

    has_tr0 = tr0_atoms.n_atoms > 0
    has_tr1 = tr1_atoms.n_atoms > 0
    if not (has_tr0 or has_tr1):
        print(f"    Warning: no TR0 or TR1 atoms found in {structure} rep{rep}")

    # Residue-index arrays for grouping atoms → molecules
    tr0_res_ix = tr0_atoms.resindices if has_tr0 else None
    tr1_res_ix = tr1_atoms.resindices if has_tr1 else None

    if (
        n_term_ca.n_atoms == 0
        or main_chain_ca.n_atoms == 0
        or nac_ca.n_atoms == 0
        or c_term_ca.n_atoms == 0
    ):
        raise ValueError(
            f"CA selection empty for {structure} rep{rep}: "
            f"N-term {n_term_ca.n_atoms}, Main-chain {main_chain_ca.n_atoms}, "
            f"NAC {nac_ca.n_atoms}, C-term {c_term_ca.n_atoms} atoms"
        )

    radgyrs = []
    radgyrs_n_term = []
    radgyrs_main_chain = []
    radgyrs_nac = []
    radgyrs_c_term = []
    tr0_counts = []
    tr1_counts = []
    nc_contact_counts = []

    n_total = len(u.trajectory)
    n_frames = len(range(0, n_total, STRIDE))
    print(f"    {n_total} frames total, processing {n_frames} (stride={STRIDE})")

    for ts in u.trajectory[::STRIDE]:
        # Radius of gyration
        radgyrs.append(protein.radius_of_gyration())
        radgyrs_n_term.append(n_term_ca.radius_of_gyration())
        radgyrs_main_chain.append(main_chain_ca.radius_of_gyration())
        radgyrs_nac.append(nac_ca.radius_of_gyration())
        radgyrs_c_term.append(c_term_ca.radius_of_gyration())

        # Tris binding: count unique molecules with any heavy atom within CUTOFF_TRIS
        prot_pos = protein.positions
        if has_tr0:
            d = cdist(tr0_atoms.positions, prot_pos)
            bound = d.min(axis=1) < CUTOFF_TRIS
            tr0_counts.append(len(np.unique(tr0_res_ix[bound])) if bound.any() else 0)
        else:
            tr0_counts.append(0)

        if has_tr1:
            d = cdist(tr1_atoms.positions, prot_pos)
            bound = d.min(axis=1) < CUTOFF_TRIS
            tr1_counts.append(len(np.unique(tr1_res_ix[bound])) if bound.any() else 0)
        else:
            tr1_counts.append(0)

        # N–C contacts: CA–CA pairs within CUTOFF_NC
        d_nc = cdist(n_term_ca.positions, c_term_ca.positions)
        nc_contact_counts.append(int(np.sum(d_nc < CUTOFF_NC)))

    # Frame indices in the original (pre-stride) trajectory
    frame_idx = np.arange(n_frames, dtype=np.int32) * STRIDE
    # Effective time assuming 10 ps/frame output dt (timestamps are unreliable in
    # concatenated trajectories as they reset to 0 at each section boundary)
    eff_time_ns = frame_idx.astype(np.float32) / 100.0

    tr0 = np.array(tr0_counts, dtype=np.float32)
    tr1 = np.array(tr1_counts, dtype=np.float32)
    total = tr0 + tr1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tr0 > 0, tr1 / tr0, np.nan)

    return {
        "frame": frame_idx,
        "eff_time_ns": eff_time_ns,
        "radgyr": np.array(radgyrs, dtype=np.float32),
        "radgyr_n_term": np.array(radgyrs_n_term, dtype=np.float32),
        "radgyr_main_chain": np.array(radgyrs_main_chain, dtype=np.float32),
        "radgyr_nac": np.array(radgyrs_nac, dtype=np.float32),
        "radgyr_c_term": np.array(radgyrs_c_term, dtype=np.float32),
        "n_tris_total": total,
        "tr1_tr0_ratio": ratio,
        "nc_contacts": np.array(nc_contact_counts, dtype=np.float32),
    }


def load_all(data_dir: Path, top_pattern: str, traj_pattern: str) -> dict:
    """Load all structures × replicates.

    Returns: results[structure][rep] = {metric: np.ndarray}
    """
    results = {}
    for structure in STRUCTURES:
        results[structure] = {}
        for rep in range(1, N_REPLICATES + 1):
            try:
                results[structure][rep] = load_replicate(
                    structure=structure,
                    rep=rep,
                    data_dir=data_dir,
                    top_pattern=top_pattern,
                    traj_pattern=traj_pattern,
                )
            except FileNotFoundError as e:
                print(f"  Warning — skipping: {e}")
    return results


# ============================================================================
# Plotting
# ============================================================================


def save_fig(fig: plt.Figure, name: str, output_dir: Path):
    out_pdf = output_dir / f"{name}.pdf"
    out_png = output_dir / f"{name}.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")


def plot_timeseries(results: dict, metric: str, output_dir: Path):
    """1D line plots: 3 panels (Rod | Hairpin | Coil), one line per replicate.

    X-axis: original frame index (timestamps are unreliable in concatenated
    trajectories). Secondary x-axis shows effective time in ns (1 ps/frame assumed).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

    for col, structure in enumerate(STRUCTURES):
        ax = axes[col]
        struct_data = results.get(structure, {})

        # Plot individual replicates
        rep_y_values = []
        common_frames = None
        for rep_idx, rep in enumerate(range(1, N_REPLICATES + 1)):
            if rep not in struct_data:
                continue
            d = struct_data[rep]
            ax.plot(
                d["frame"],
                d[metric],
                color=REP_COLORS[rep_idx],
                alpha=REP_ALPHA,
                linewidth=LINE_WIDTH,
                label=f"rep{rep}",
                rasterized=True,
            )
            rep_y_values.append(d[metric])
            if common_frames is None:
                common_frames = d["frame"]

        # Plot black dashed mean line across replicates
        if len(rep_y_values) > 0:
            min_len = min(len(y) for y in rep_y_values)
            mean_y = np.mean([y[:min_len] for y in rep_y_values], axis=0)
            ax.plot(
                common_frames[:min_len],
                mean_y,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Mean",
                rasterized=True,
            )

        ax.set_title(STRUCTURE_LABELS[structure])
        ax.set_xlabel("Frame")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS[metric])
        remove_top_right_spines(ax)

        # Secondary x-axis: effective time in ns (frame / 100, assuming 10 ps/frame)
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(lambda f: f / 100.0, lambda t: t * 100.0),
        )
        ax2.set_xlabel("Time (ns)" if col == 1 else "")
        ax2.tick_params(labelsize=8)

    # Single shared legend on the last panel
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    save_fig(fig, f"timeseries_{metric}", output_dir)


def plot_2d_density(results: dict, metric_x: str, metric_y: str, output_dir: Path):
    """2D KDE density: 4 panels (Starting | Rod | Hairpin | Coil)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # Collect pooled data for panel 0
    pooled_x = []
    pooled_y = []
    for structure in STRUCTURES:
        struct_data = results.get(structure, {})
        for rep in range(1, N_REPLICATES + 1):
            if rep in struct_data:
                d = struct_data[rep]
                x, y = d[metric_x], d[metric_y]
                mask = np.isfinite(x) & np.isfinite(y)
                pooled_x.append(x[mask])
                pooled_y.append(y[mask])

    # Panel 0: All starting structures (frame 0 of each replicate) + Pooled Density
    ax0 = axes[0]
    ax0.set_title("Starting Structures")
    ax0.set_xlabel(METRIC_LABELS[metric_x])
    ax0.set_ylabel(METRIC_LABELS[metric_y])
    remove_top_right_spines(ax0)

    if pooled_x:
        px = np.concatenate(pooled_x)
        py = np.concatenate(pooled_y)
        sns.kdeplot(
            x=px,
            y=py,
            ax=ax0,
            fill=True,
            levels=5,
            color="grey",
            alpha=0.3,
            zorder=0,
        )

    for structure in STRUCTURES:
        struct_data = results.get(structure, {})
        start_x, start_y = [], []
        for rep in range(1, N_REPLICATES + 1):
            if rep in struct_data:
                d = struct_data[rep]
                # Frame 0 values
                start_x.append(d[metric_x][0])
                start_y.append(d[metric_y][0])

        if start_x:
            ax0.scatter(
                start_x,
                start_y,
                color=STRUCTURE_COLORS[structure],
                label=STRUCTURE_LABELS[structure],
                edgecolor="black",
                linewidth=1.0,
                s=70,
                alpha=1.0,
                zorder=10,
            )
    ax0.legend(frameon=False, fontsize=8)

    # Panels 1-3: Per-structure pooled density
    for col, structure in enumerate(STRUCTURES):
        ax = axes[col + 1]
        struct_data = results.get(structure, {})

        all_x, all_y = [], []
        for rep in range(1, N_REPLICATES + 1):
            if rep not in struct_data:
                continue
            d = struct_data[rep]
            x, y = d[metric_x], d[metric_y]
            mask = np.isfinite(x) & np.isfinite(y)
            all_x.append(x[mask])
            all_y.append(y[mask])

        ax.set_title(STRUCTURE_LABELS[structure])
        ax.set_xlabel(METRIC_LABELS[metric_x])
        ax.set_ylabel("")  # Y-label already on panel 0
        remove_top_right_spines(ax)

        if not all_x or sum(len(a) for a in all_x) < 10:
            continue

        x_pool = np.concatenate(all_x)
        y_pool = np.concatenate(all_y)

        sns.kdeplot(
            x=x_pool,
            y=y_pool,
            ax=ax,
            fill=True,
            levels=5,
            color=STRUCTURE_COLORS[structure],
            alpha=0.75,
            linewidths=1.0,
        )

    fig.tight_layout()
    save_fig(fig, f"density_{metric_x}_vs_{metric_y}", output_dir)


def plot_comparison_histogram(tris_results: dict, control_results: dict, metric: str, output_dir: Path):
    """Plot comparison histograms (density distributions) for a metric.

    3 panels (Rod | Hairpin | Coil), plotting both ensembles on the same panel.
    Tris MD in black, control MD in grey.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)

    for col, structure in enumerate(STRUCTURES):
        ax = axes[col]
        
        # Gather all data across replicates for tris_MD
        tris_data_list = []
        struct_tris = tris_results.get(structure, {})
        for rep in range(1, N_REPLICATES + 1):
            if rep in struct_tris:
                tris_data_list.append(struct_tris[rep][metric])
        tris_data = np.concatenate(tris_data_list) if len(tris_data_list) > 0 else np.array([])
        
        # Gather all data across replicates for control_MD
        control_data_list = []
        struct_control = control_results.get(structure, {})
        for rep in range(1, N_REPLICATES + 1):
            if rep in struct_control:
                control_data_list.append(struct_control[rep][metric])
        control_data = np.concatenate(control_data_list) if len(control_data_list) > 0 else np.array([])

        # Plot histograms
        if len(tris_data) > 0 or len(control_data) > 0:
            # Determine common bins
            all_vals = []
            if len(tris_data) > 0:
                all_vals.extend(tris_data)
            if len(control_data) > 0:
                all_vals.extend(control_data)
            
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            if min_val == max_val:
                bins = 30
            else:
                bins = np.linspace(min_val, max_val, 31)

            if len(tris_data) > 0:
                ax.hist(
                    tris_data,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    color="black",
                    alpha=0.25,
                    label="Tris MD",
                    rasterized=True,
                )
                ax.hist(
                    tris_data,
                    bins=bins,
                    density=True,
                    histtype="step",
                    color="black",
                    linewidth=1.2,
                    rasterized=True,
                )

            if len(control_data) > 0:
                ax.hist(
                    control_data,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    color="grey",
                    alpha=0.25,
                    label="Control MD",
                    rasterized=True,
                )
                ax.hist(
                    control_data,
                    bins=bins,
                    density=True,
                    histtype="step",
                    color="grey",
                    linewidth=1.2,
                    rasterized=True,
                )

        ax.set_title(STRUCTURE_LABELS[structure])
        ax.set_xlabel(METRIC_LABELS[metric])
        if col == 0:
            ax.set_ylabel("Density")
        remove_top_right_spines(ax)

    # Single shared legend on the last panel
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    save_fig(fig, f"comparison_histogram_{metric}", output_dir)


# ============================================================================
# Main
# ============================================================================


def main():
    set_publication_style()

    sim_configs = [
        {
            "name": "tris_MD",
            "dir": SCRIPT_DIR / "_aSyn" / "tris_MD",
            "top_pattern": "md_mol_center_{structure}.gro",
            "traj_pattern": "tris_{structure}_rep{rep}_combined.xtc",
        },
        {
            "name": "control_MD",
            "dir": SCRIPT_DIR / "_aSyn" / "control_MD",
            "top_pattern": "md.gro.pdb",
            "traj_pattern": "control_{structure}_rep{rep}_combined.xtc",
        },
    ]

    # 1. Load data for both simulation suites up front
    print(f"\n========================================================")
    print("Loading Tris MD Trajectories...")
    print(f"========================================================")
    tris_results = load_all(
        data_dir=sim_configs[0]["dir"],
        top_pattern=sim_configs[0]["top_pattern"],
        traj_pattern=sim_configs[0]["traj_pattern"],
    )

    print(f"\n========================================================")
    print("Loading Control MD Trajectories...")
    print(f"========================================================")
    control_results = load_all(
        data_dir=sim_configs[1]["dir"],
        top_pattern=sim_configs[1]["top_pattern"],
        traj_pattern=sim_configs[1]["traj_pattern"],
    )

    # 2. Plot results for each ensemble individually
    all_results = {
        "tris_MD": tris_results,
        "control_MD": control_results,
    }

    for name, results in all_results.items():
        # Skip if no replicates were successfully loaded
        any_loaded = any(results[struct] for struct in results)
        if not any_loaded:
            print(f"Warning: No trajectories loaded for {name}")
            continue

        print(f"\nGenerating plots for {name} ...")
        output_dir = SCRIPT_DIR / "_figures" / "md_analysis" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        print("  Plotting 1D timeseries ...")
        for metric in METRIC_LABELS:
            plot_timeseries(results, metric, output_dir)

        print("  Plotting 2D density plots ...")
        for mx, my in DENSITY_PAIRS:
            plot_2d_density(results, mx, my, output_dir)

        print(f"Done. Figures saved to: {output_dir}")

    # 3. Plot comparison histograms with both ensembles on the same panel
    print(f"\n========================================================")
    print("Generating Comparison Histograms...")
    print(f"========================================================")
    comparison_dir = SCRIPT_DIR / "_figures" / "md_analysis" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_LABELS:
        plot_comparison_histogram(tris_results, control_results, metric, comparison_dir)

    print(f"\nDone. Comparison histograms saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
