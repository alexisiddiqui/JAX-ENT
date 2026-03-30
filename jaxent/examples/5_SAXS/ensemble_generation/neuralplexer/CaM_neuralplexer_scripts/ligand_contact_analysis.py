"""
Analyze ligand-protein contacts in PDB structures.
Compare reference PDB structures (fetched from RCSB) against a NeuralPLexer
ensemble of predicted conformations.

Usage:
    python ligand_contact_analysis.py \
        --ensemble-dir /path/to/combined_outputs \
        [--pdb-codes 7PU9 7PSZ]
"""

import argparse
import os
import re
import tempfile
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from matplotlib.lines import Line2D
from rdkit import Chem
from scipy.stats import spearmanr

# Suppress MDAnalysis warnings about missing chainIDs/elements
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
# Suppress scipy warnings about constant input in Spearman correlation
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

ligand_contact_threshold = 4.0
clash_threshold = 2.0  # Hard steric overlap for HETATM atoms

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def fetch_pdb(pdb_code):
    """Download PDB file from RCSB PDB server or use local copy if available."""
    pdb_code = pdb_code.upper()
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            print(f"  Downloading {pdb_code} from RCSB PDB...", end=" ", flush=True)
            urllib.request.urlretrieve(url, tmp.name)
            print("\u2713")
            return tmp.name
    except Exception:
        print("\u2717 (trying local files)")
        for local_path in [
            f"/mnt/user-data/uploads/{pdb_code}.pdb",
            f"/mnt/user-data/uploads/{pdb_code.lower()}.pdb",
            f"./{pdb_code}.pdb",
            f"./{pdb_code.lower()}.pdb",
        ]:
            if os.path.exists(local_path):
                print(f"  Found local file: {local_path}")
                return local_path
        print(f"  Error: Could not find {pdb_code}.pdb locally or download from RCSB")
        return None


def parse_rank_plddt(filename):
    """Extract (rank_int, plddt_float) from a filename like complex_rank1_plddt0.6135966.pdb."""
    m = re.search(r'rank(\d+)_plddt(\d+(?:\.\d+)?)', filename)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def get_protein_heavy_atoms(u):
    """Extract coordinates of protein heavy atoms (excluding H)."""
    sel = u.select_atoms("protein and not (name H* or element H)")
    return sel.positions if len(sel) > 0 else None


def get_cdz_ligand_atoms(u):
    """Get CDZ/85H ligand heavy atoms with residue indices.

    Works for both RCSB PDB (resname 85H) and NeuralPLexer output (resname CDZ).
    """
    sel = u.select_atoms("(resname CDZ or resname 85H) and not (name H* or element H)")
    if len(sel) == 0:
        return None, []
    return sel.positions, sel.resids.tolist()


# ---------------------------------------------------------------------------
# Contact calculations
# ---------------------------------------------------------------------------

def calculate_contacts_for_file(pdb_path, label=""):
    """Calculate CDZ/85H ligand-protein contacts for a single PDB file."""
    try:
        u = mda.Universe(pdb_path)
    except Exception as e:
        return {"label": label, "error": str(e)}

    protein_atoms = get_protein_heavy_atoms(u)
    if protein_atoms is None or len(protein_atoms) == 0:
        return {"label": label, "error": "No protein heavy atoms found"}

    ligand_atoms, residue_indices = get_cdz_ligand_atoms(u)
    if ligand_atoms is None or len(ligand_atoms) == 0:
        return {"label": label, "error": "No CDZ/85H ligand atoms found"}

    num_ligand = len(ligand_atoms)
    min_distances = []
    ligand_with_contact = 0

    for lig_atom in ligand_atoms:
        dists = np.linalg.norm(protein_atoms - lig_atom, axis=1)
        d_min = float(np.min(dists))
        min_distances.append(d_min)
        if d_min < ligand_contact_threshold:
            ligand_with_contact += 1

    pct = 100.0 * ligand_with_contact / num_ligand

    return {
        "label": label,
        "total_ligand_atoms": num_ligand,
        "atoms_with_contact": ligand_with_contact,
        "percentage": pct,
        "min_distances": min_distances,
        "residue_indices": residue_indices,
        "mean_min_distance": float(np.mean(min_distances)),
        "median_min_distance": float(np.median(min_distances)),
        "max_min_distance": float(np.max(min_distances)),
        "min_min_distance": float(np.min(min_distances)),
    }


def calculate_contacts_reference(pdb_code):
    """Calculate contacts for a reference PDB code (downloaded from RCSB)."""
    pdb_file = fetch_pdb(pdb_code)
    if pdb_file is None:
        return {"code": pdb_code, "error": "Failed to download PDB file"}
    try:
        result = calculate_contacts_for_file(pdb_file, label=pdb_code)
        result["code"] = pdb_code
        return result
    finally:
        if pdb_file and pdb_file.startswith("/tmp") and os.path.exists(pdb_file):
            os.remove(pdb_file)


def calculate_ca_oxygen_for_file(pdb_path, label="",
                                  contact_threshold=None):
    """Calculate CA (calcium)-oxygen contacts for a single PDB file."""
    CA_PROTEIN_THRESHOLD = 2.5

    if contact_threshold is None:
        contact_threshold = ligand_contact_threshold

    try:
        u = mda.Universe(pdb_path)
    except Exception as e:
        return {"label": label, "error": str(e)}

    # All oxygen atoms
    oxygens = u.select_atoms("element O")
    if len(oxygens) == 0:
        oxygens = u.select_atoms("name O* and not name OH")
    if len(oxygens) == 0:
        return {"label": label, "error": "No oxygen atoms found"}
    oxy_pos = oxygens.positions
    oxy_resids = oxygens.resids
    oxy_is_protein = np.array(
        [a.residue.resname not in ("85H", "CDZ", "CA", "CAL", "HOH", "WAT", "TIP3")
         for a in oxygens],
        dtype=bool,
    )

    # Calcium ions: try element first, fall back to resname
    calcium = u.select_atoms("element CA")
    if len(calcium) == 0:
        calcium = u.select_atoms("resname CA or resname CAL")
    if len(calcium) == 0:
        return {"label": label, "error": "No CA (calcium) atoms found"}

    by_resid = {}
    protein_res_counts = {}

    for atom in calcium:
        ca_resid = int(atom.resid)
        dists = np.linalg.norm(oxy_pos - atom.position, axis=1)
        by_resid.setdefault(ca_resid, []).append(float(np.min(dists)))

        close_mask = (dists < CA_PROTEIN_THRESHOLD) & oxy_is_protein
        for prot_resid in np.unique(oxy_resids[close_mask]):
            protein_res_counts[int(prot_resid)] = (
                protein_res_counts.get(int(prot_resid), 0) + 1
            )

    return {
        "label": label,
        "by_resid": by_resid,
        "protein_res_counts": protein_res_counts,
    }


def calculate_ca_oxygen_reference(pdb_code, contact_threshold=None):
    """CA-oxygen contacts for a reference PDB code."""
    pdb_file = fetch_pdb(pdb_code)
    if pdb_file is None:
        return {"code": pdb_code, "error": "Failed to download PDB file"}
    try:
        result = calculate_ca_oxygen_for_file(pdb_file, label=pdb_code,
                                               contact_threshold=contact_threshold)
        result["code"] = pdb_code
        return result
    finally:
        if pdb_file and pdb_file.startswith("/tmp") and os.path.exists(pdb_file):
            os.remove(pdb_file)


# ---------------------------------------------------------------------------
# Clash calculations (HETATM steric overlaps)
# ---------------------------------------------------------------------------

def calculate_clashes_for_file(pdb_path, excluded_pairs=None, label=""):
    """Count steric clashes for HETATM atoms (ligand and ions).

    Counts atoms (by type) that have ≥1 clash partner (distance < clash_threshold).
    Returns het-het clashes (ligand vs ion, inter-residue) and protein-het clashes.
    """
    try:
        u = mda.Universe(pdb_path)
        # Attempt to guess bonds if not present, to safely exclude covalent bonds from clashes
        try:
            _ = u.bonds
        except Exception:
            import MDAnalysis.topology.guessers as guessers
            u.add_TopologyAttr('bonds', guessers.guess_bonds(u.atoms, u.atoms.positions))
    except Exception as e:
        return {"label": label, "error": str(e)}

    # Select atoms
    protein = u.select_atoms("protein and not (name H* or element H)")
    cdz_atoms = u.select_atoms("resname CDZ and not (name H* or element H)")
    ca_atoms = u.select_atoms("(element CA and not protein) or (resname CA and not protein)")

    # Count atoms with clashes by type
    het_het_ligand = 0    # CDZ atoms with ≥1 clash vs (other CDZ residues OR CA atoms)
    het_het_ion = 0       # CA atoms with ≥1 clash vs (other CA residues OR CDZ atoms)
    prot_het_ligand = 0   # CDZ atoms with ≥1 clash vs protein
    prot_het_ion = 0      # CA atoms with ≥1 clash vs protein
    cdz_self_intra = 0    # CDZ atoms with ≥1 intra-residue clash

    # Het-het: CDZ vs other CDZ residues and vs CA atoms
    if len(cdz_atoms) > 0:
        cdz_resids = sorted(set(cdz_atoms.resids))
        for resid in cdz_resids:
            this_cdz = cdz_atoms.select_atoms(f"resid {resid}")
            other_cdz = cdz_atoms.select_atoms(f"not resid {resid}")
            
            # Ensure we iterate by enumerated index 'i' which matches the SDF heavy atom order
            for i, atom in enumerate(this_cdz):
                has_clash = False
                has_self_intra = False
                
                # Check intra-residue clashes
                if excluded_pairs is not None:
                    valid_intra_pos = [other.position for j, other in enumerate(this_cdz) if (i, j) not in excluded_pairs]
                    if valid_intra_pos:
                        dists = np.linalg.norm(np.array(valid_intra_pos) - atom.position, axis=1)
                        has_self_intra = np.any(dists < clash_threshold)
                else:
                    # Naive fallback
                    valid_intra_pos = [other.position for j, other in enumerate(this_cdz) if i != j]
                    if valid_intra_pos:
                        dists = np.linalg.norm(np.array(valid_intra_pos) - atom.position, axis=1)
                        has_self_intra = np.any((dists < clash_threshold) & (dists > 1.6))

                # Check vs other CDZ residues (inter-residue clashes)
                if len(other_cdz) > 0:
                    dists = np.linalg.norm(other_cdz.positions - atom.position, axis=1)
                    has_clash = has_clash or np.any(dists < clash_threshold)
                # Check vs CA atoms
                if len(ca_atoms) > 0:
                    dists = np.linalg.norm(ca_atoms.positions - atom.position, axis=1)
                    has_clash = has_clash or np.any(dists < clash_threshold)
                if has_clash:
                    het_het_ligand += 1
                if has_self_intra:
                    cdz_self_intra += 1

    # Het-het: CA vs other CA residues and vs CDZ atoms
    if len(ca_atoms) > 0:
        ca_resids = sorted(set(ca_atoms.resids))
        for resid in ca_resids:
            this_ca = ca_atoms.select_atoms(f"resid {resid}")
            other_ca = ca_atoms.select_atoms(f"not resid {resid}")
            for atom in this_ca:
                has_clash = False
                # Check vs other CA residues (ion-ion clashes)
                if len(other_ca) > 0:
                    dists = np.linalg.norm(other_ca.positions - atom.position, axis=1)
                    has_clash = has_clash or np.any(dists < clash_threshold)
                # Check vs CDZ atoms
                if len(cdz_atoms) > 0:
                    dists = np.linalg.norm(cdz_atoms.positions - atom.position, axis=1)
                    has_clash = has_clash or np.any(dists < clash_threshold)
                if has_clash:
                    het_het_ion += 1

    # Protein-het: protein vs CDZ
    if len(protein) > 0 and len(cdz_atoms) > 0:
        for atom in cdz_atoms:
            dists = np.linalg.norm(protein.positions - atom.position, axis=1)
            if np.any(dists < clash_threshold):
                prot_het_ligand += 1

    # Protein-het: protein vs CA
    if len(protein) > 0 and len(ca_atoms) > 0:
        for atom in ca_atoms:
            dists = np.linalg.norm(protein.positions - atom.position, axis=1)
            if np.any(dists < clash_threshold):
                prot_het_ion += 1

    return {
        "label": label,
        "has_cdz": len(cdz_atoms) > 0,
        "het_het_ligand": het_het_ligand,
        "het_het_ion": het_het_ion,
        "prot_het_ligand": prot_het_ligand,
        "prot_het_ion": prot_het_ion,
        "cdz_self_intra": cdz_self_intra,
    }


def calculate_ensemble_clashes(ensemble_dir, excluded_pairs=None):
    """Count steric clashes for every frame in an ensemble."""
    frames = load_ensemble(ensemble_dir)
    results = []
    for i, fr in enumerate(frames):
        r = calculate_clashes_for_file(fr["path"], excluded_pairs=excluded_pairs, label=f"rank{fr['rank']}")
        r["rank"] = fr["rank"]
        r["plddt"] = fr["plddt"]
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def load_ensemble(ensemble_dir):
    """Return sorted list of dicts with rank, plddt, path for each frame."""
    ensemble_dir = Path(ensemble_dir)
    frames = []
    for f in sorted(ensemble_dir.glob("complex_rank*.pdb")):
        parsed = parse_rank_plddt(f.name)
        if parsed:
            rank, plddt = parsed
            frames.append({"rank": rank, "plddt": plddt, "path": str(f)})
    frames.sort(key=lambda x: x["rank"])
    return frames


def calculate_ensemble_contacts(ensemble_dir):
    """CDZ ligand-protein contacts for every frame in an ensemble."""
    frames = load_ensemble(ensemble_dir)
    results = []
    for i, fr in enumerate(frames):
        print(f"  [{i+1}/{len(frames)}] rank {fr['rank']}  pLDDT={fr['plddt']:.4f}")
        r = calculate_contacts_for_file(fr["path"], label=f"rank{fr['rank']}")
        r["rank"] = fr["rank"]
        r["plddt"] = fr["plddt"]
        results.append(r)
    return results


def calculate_ensemble_ca(ensemble_dir):
    """CA-oxygen contacts for every frame in an ensemble."""
    frames = load_ensemble(ensemble_dir)
    results = []
    for i, fr in enumerate(frames):
        r = calculate_ca_oxygen_for_file(fr["path"], label=f"rank{fr['rank']}")
        r["rank"] = fr["rank"]
        r["plddt"] = fr["plddt"]
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Plotting – Distance histogram
# ---------------------------------------------------------------------------

def create_distance_histogram(ref_results, ref_codes,
                              ens_results=None, ensemble_label="Ensemble", output_dir=None):
    """Faceted histograms: one panel per reference PDB + one ensemble panel."""

    valid_refs = [(c, r) for c, r in zip(ref_codes, ref_results) if "error" not in r]
    has_ensemble = ens_results and any("error" not in r for r in ens_results)
    n_panels = len(valid_refs) + (1 if has_ensemble else 0)

    if n_panels == 0:
        print("No valid results to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    # --- Reference panels ---
    for ax_idx, (code, results) in enumerate(valid_refs):
        distances = np.array(results["min_distances"])
        res_indices = results["residue_indices"]
        unique_idx = sorted(set(res_indices))

        for ci, ridx in enumerate(unique_idx):
            mask = np.array(res_indices) == ridx
            axes[ax_idx].hist(
                distances[mask], bins=12, alpha=0.7, density=True,
                label=f"85H #{ridx}", color=colors[ci % len(colors)],
                edgecolor="black", linewidth=0.8,
            )

        axes[ax_idx].axvline(x=ligand_contact_threshold, color="grey",
                             linestyle=":", linewidth=2,
                             label=f"{ligand_contact_threshold} \u00c5 threshold")
        axes[ax_idx].set_xlabel("Min Distance to Protein (\u00c5)", fontsize=11,
                                fontweight="bold")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        axes[ax_idx].set_title(code.upper(), fontsize=13, fontweight="bold")
        axes[ax_idx].grid(axis="y", alpha=0.3)
        axes[ax_idx].legend(fontsize=9, loc="upper right", framealpha=0.95)

    # --- Ensemble panel (pooled over all frames) ---
    if has_ensemble:
        ax_ens = axes[-1]
        pooled = []
        for r in ens_results:
            if "error" not in r:
                pooled.extend(r["min_distances"])
        pooled = np.array(pooled)
        ax_ens.hist(pooled, bins=30, alpha=0.7, color="#4c72b0", density=True,
                    edgecolor="black", linewidth=0.5, label="All CDZ atoms")
        ax_ens.axvline(x=ligand_contact_threshold, color="grey",
                       linestyle=":", linewidth=2,
                       label=f"{ligand_contact_threshold} \u00c5 threshold")
        ax_ens.set_xlabel("Min Distance to Protein (\u00c5)", fontsize=11,
                          fontweight="bold")
        ax_ens.set_title(ensemble_label, fontsize=13, fontweight="bold")
        ax_ens.grid(axis="y", alpha=0.3)
        ax_ens.legend(fontsize=9, loc="upper right", framealpha=0.95)

    fig.suptitle("Distribution of Minimum Ligand-Protein Distances (CDZ / 85H)",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(str(output_dir), f"distance_histogram.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\u2713 Histogram saved to: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plotting – Cumulative contact distribution
# ---------------------------------------------------------------------------

def create_cumulative_contact_plot(ref_results, ref_codes,
                                   ens_results=None, ensemble_label="Ensemble", output_dir=None):
    """Two-panel cumulative plot.

    Left : references hued by PDB:residue_index (original behaviour).
    Right: ensemble frames, perceptually-uniform colour by frame,
           alpha inversely proportional to pLDDT, + black average.
    """

    valid_refs = [(c, r) for c, r in zip(ref_codes, ref_results) if "error" not in r]
    has_ensemble = ens_results and any("error" not in r for r in ens_results)
    n_panels = (1 if valid_refs else 0) + (1 if has_ensemble else 0)

    if n_panels == 0:
        print("No valid results for cumulative plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # --- Left panel: references ---
    if valid_refs:
        ax = axes[ax_idx]; ax_idx += 1
        cmap_ref = plt.get_cmap("tab10")
        series = []
        for code, results in valid_refs:
            distances = np.array(results["min_distances"])
            res_indices = np.array(results["residue_indices"])
            for ridx in sorted(set(res_indices.tolist())):
                d = np.sort(distances[res_indices == ridx])
                series.append((f"{code.upper()}:{ridx}", d))

        for i, (label, d) in enumerate(series):
            x = np.concatenate([[0.0], d, [d[-1]]])
            y = np.concatenate([[0.0], np.arange(1, len(d)+1)/len(d)*100, [100.0]])
            ax.step(x, y, where="post", label=label,
                    color=cmap_ref(i % 10), linewidth=2)

        ax.axvline(x=ligand_contact_threshold, color="grey", linestyle=":",
                   linewidth=1.8, label=f"{ligand_contact_threshold} \u00c5 threshold")
        ax.set_xlabel("Cutoff Distance (\u00c5)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative % of Ligand Atoms", fontsize=12, fontweight="bold")
        ax.set_title("References", fontsize=13, fontweight="bold")
        ax.set_xlim(left=0.0); ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    # --- Right panel: ensemble ---
    if has_ensemble:
        ax = axes[ax_idx]
        valid_ens = [r for r in ens_results if "error" not in r]
        plddts = np.array([r["plddt"] for r in valid_ens])
        plddt_min, plddt_max = plddts.min(), plddts.max()
        plddt_range = plddt_max - plddt_min if plddt_max > plddt_min else 1.0

        cmap_ens = plt.get_cmap("viridis")
        n_frames = len(valid_ens)

        # Collect all ECDFs for averaging
        # Use a common x-grid for the average
        all_dists_flat = []
        ecdf_interp = []  # list of (x_grid_values, y_values) for averaging

        x_grid = np.linspace(0, 6.0, 500)

        for fi, r in enumerate(valid_ens):
            d = np.sort(r["min_distances"])
            all_dists_flat.extend(r["min_distances"])
            # ECDF
            x = np.concatenate([[0.0], d, [d[-1] + 1e-6]])
            y = np.concatenate([[0.0], np.arange(1, len(d)+1)/len(d)*100, [100.0]])

            # Interpolate onto common grid
            y_interp = np.interp(x_grid, x, y)
            ecdf_interp.append(y_interp)

            # Alpha: lower pLDDT -> higher alpha
            alpha = 0.15 + 0.75 * (1.0 - (r["plddt"] - plddt_min) / plddt_range)
            color = cmap_ens(fi / max(n_frames - 1, 1))

            ax.step(x, y, where="post", color=color, linewidth=0.8, alpha=alpha)

        # Average line
        avg_y = np.mean(ecdf_interp, axis=0)
        ax.plot(x_grid, avg_y, color="black", linewidth=2.5, label="Ensemble average")

        ax.axvline(x=ligand_contact_threshold, color="grey", linestyle=":",
                   linewidth=1.8, label=f"{ligand_contact_threshold} \u00c5 threshold")

        # Colorbar for frames
        sm = plt.cm.ScalarMappable(cmap=cmap_ens,
                                    norm=plt.Normalize(1, n_frames))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
        cbar.set_label("Frame (rank)", fontsize=10)

        ax.set_xlabel("Cutoff Distance (\u00c5)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative % of Ligand Atoms", fontsize=12, fontweight="bold")
        ax.set_title(ensemble_label, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 6.0); ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(str(output_dir), f"cumulative_contact_distribution.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\u2713 Cumulative plot saved to: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plotting – CA-oxygen cumulative
# ---------------------------------------------------------------------------

def create_ca_oxygen_cumulative_plot(ref_ca_results, ref_codes,
                                     ens_ca_results=None,
                                     ensemble_label="Ensemble", output_dir=None):
    """CA-oxygen cumulative distribution.

    Left : references (hue = resid, dash = PDB).
    Right: ensemble average.
    """
    valid_refs = [(c, r) for c, r in zip(ref_codes, ref_ca_results) if "error" not in r]
    valid_ens = [r for r in (ens_ca_results or []) if "error" not in r]
    has_refs = bool(valid_refs)
    has_ens = bool(valid_ens)
    n_panels = int(has_refs) + int(has_ens)
    if n_panels == 0:
        print("No valid CA-oxygen results to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # --- Left panel: references ---
    if has_refs:
        ax = axes[ax_idx]; ax_idx += 1
        all_resids = sorted({r for _, res in valid_refs for r in res["by_resid"]})
        cmap = plt.get_cmap("tab10")
        resid_color = {r: cmap(i % 10) for i, r in enumerate(all_resids)}
        linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
        pdb_ls = {code: linestyles[i % len(linestyles)]
                  for i, (code, _) in enumerate(valid_refs)}

        legend_pdb = []; legend_resid = []
        added_pdb = set(); added_resid = set()

        for code, res in valid_refs:
            ls = pdb_ls[code]
            for resid in sorted(res["by_resid"]):
                d = np.sort(res["by_resid"][resid])
                color = resid_color[resid]
                x = np.concatenate([[0.0], d, [d[-1]]])
                y = np.concatenate([[0], np.arange(1, len(d)+1), [len(d)]])
                ax.step(x, y, where="post", color=color, linestyle=ls, linewidth=2)

                if code not in added_pdb:
                    legend_pdb.append(Line2D([0],[0], color="gray", linestyle=ls,
                                             linewidth=2, label=code.upper()))
                    added_pdb.add(code)
                if resid not in added_resid:
                    legend_resid.append(Line2D([0],[0], color=color, linestyle="-",
                                                linewidth=3, label=f"resid {resid}"))
                    added_resid.add(resid)

        thresh_h = Line2D([0],[0], color="grey", linestyle=":", linewidth=1.8,
                          label=f"{ligand_contact_threshold} \u00c5 threshold")
        ax.axvline(x=ligand_contact_threshold, color="grey", linestyle=":", linewidth=1.8)
        ax.legend(handles=legend_resid + legend_pdb + [thresh_h],
                  fontsize=9, loc="lower right", framealpha=0.95)

        ax.set_xlabel("Distance to Nearest Oxygen (\u00c5)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Frequency (# O contacts)", fontsize=12, fontweight="bold")
        ax.set_title("References", fontsize=13, fontweight="bold")
        ax.set_xlim(left=0.0); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

    # --- Right panel: ensemble average ---
    if has_ens:
        ax = axes[ax_idx]
        # Pool all CA min distances across frames, compute average ECDF
        all_resids_ens = sorted({r for res in valid_ens for r in res["by_resid"]})
        cmap_e = plt.get_cmap("tab10")
        resid_color_e = {r: cmap_e(i % 10) for i, r in enumerate(all_resids_ens)}
        x_grid = np.linspace(0, 15, 300)

        for resid in all_resids_ens:
            ecdfs = []
            for res in valid_ens:
                dists = res["by_resid"].get(resid, [])
                if not dists:
                    continue
                d = np.sort(dists)
                x = np.concatenate([[0.0], d, [d[-1] + 1e-6]])
                y = np.concatenate([[0], np.arange(1, len(d)+1), [len(d)]])
                y_interp = np.interp(x_grid, x, y)
                ecdfs.append(y_interp)
            if ecdfs:
                avg_y = np.mean(ecdfs, axis=0)
                ax.plot(x_grid, avg_y, color=resid_color_e[resid], linewidth=2,
                        label=f"CA resid {resid} (avg)")

        ax.axvline(x=ligand_contact_threshold, color="grey", linestyle=":",
                   linewidth=1.8, label=f"{ligand_contact_threshold} \u00c5 threshold")
        ax.set_xlabel("Distance to Nearest Oxygen (\u00c5)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Frequency (# O contacts)", fontsize=12, fontweight="bold")
        ax.set_title(f"{ensemble_label} (average)", fontsize=13, fontweight="bold")
        ax.set_xlim(left=0.0); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(str(output_dir), f"ca_oxygen_cumulative.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\u2713 CA-oxygen cumulative plot saved to: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plotting – CA residue contact bar
# ---------------------------------------------------------------------------

def create_ca_residue_contact_bar_plot(ref_ca_results, ref_codes,
                                       ens_ca_results=None,
                                       ensemble_label="Ensemble",
                                       ca_protein_threshold=2.5, output_dir=None):
    """Grouped bar chart: CA-protein oxygen contacts per protein residue.

    Left : references (one bar colour per PDB).
    Right: ensemble average.
    """
    valid_refs = [
        (c, r) for c, r in zip(ref_codes, ref_ca_results)
        if "error" not in r and r.get("protein_res_counts")
    ]
    valid_ens = [r for r in (ens_ca_results or [])
                 if "error" not in r and r.get("protein_res_counts")]
    has_refs = bool(valid_refs)
    has_ens = bool(valid_ens)
    n_panels = int(has_refs) + int(has_ens)
    if n_panels == 0:
        print("No CA-protein oxygen contacts found for bar plot.")
        return

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(max(10, 12) * n_panels * 0.6, 5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # --- Left panel: references ---
    if has_refs:
        ax = axes[ax_idx]; ax_idx += 1
        all_resids = sorted({r for _, res in valid_refs for r in res["protein_res_counts"]})
        n_resids = len(all_resids)
        n_pdbs = len(valid_refs)
        resid_idx = {r: i for i, r in enumerate(all_resids)}
        bar_w = 0.8 / n_pdbs
        offsets = np.linspace(-(n_pdbs-1)/2, (n_pdbs-1)/2, n_pdbs) * bar_w
        cmap = plt.get_cmap("tab10")

        for pi, (code, res) in enumerate(valid_refs):
            counts = res["protein_res_counts"]
            x_pos = [resid_idx[r] + offsets[pi] for r in all_resids]
            y_vals = [counts.get(r, 0) for r in all_resids]
            ax.bar(x_pos, y_vals, width=bar_w*0.9, color=cmap(pi % 10),
                   label=code.upper(), edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(n_resids))
        ax.set_xticklabels([str(r) for r in all_resids], rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Protein Residue ID", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"# CA Contacts (< {ca_protein_threshold} \u00c5)",
                      fontsize=12, fontweight="bold")
        ax.set_title("References", fontsize=13, fontweight="bold")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9, framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)

    # --- Right panel: ensemble average ---
    if has_ens:
        ax = axes[ax_idx]
        # Average protein_res_counts across ensemble frames
        all_resids_e = sorted({r for res in valid_ens for r in res["protein_res_counts"]})
        n_resids_e = len(all_resids_e)
        avg_counts = {}
        for r in all_resids_e:
            vals = [res["protein_res_counts"].get(r, 0) for res in valid_ens]
            avg_counts[r] = np.mean(vals)

        x_pos = list(range(n_resids_e))
        y_vals = [avg_counts[r] for r in all_resids_e]
        ax.bar(x_pos, y_vals, color="#4c72b0", edgecolor="white", linewidth=0.5,
               label=f"{ensemble_label} avg (n={len(valid_ens)})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(r) for r in all_resids_e], rotation=45,
                           ha="right", fontsize=9)
        ax.set_xlabel("Protein Residue ID", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"# CA Contacts (< {ca_protein_threshold} \u00c5)",
                      fontsize=12, fontweight="bold")
        ax.set_title(f"{ensemble_label} (average)", fontsize=13, fontweight="bold")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9, framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"CA (Calcium) \u2013 Protein Oxygen Contacts per Residue (< {ca_protein_threshold} \u00c5)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(str(output_dir), f"ca_residue_contacts.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\u2713 CA-residue bar plot saved to: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plotting – pLDDT vs clashes scatter
# ---------------------------------------------------------------------------

def create_plddt_vs_clashes_plot(clash_results, ensemble_label="Ensemble", output_dir=None):
    """Two-panel scatter: pLDDT vs clashes for ligand and ions.

    Left panel: het-het clashes.
    Right panel: protein-het clashes.
    Each point = one ensemble frame, coloured by hetatm type (ligand/ion).
    Spearman correlation ρ reported on each panel.
    CDZ series are suppressed if no CDZ ligands are present in the ensemble.
    """
    valid = [r for r in clash_results if "error" not in r]
    if not valid:
        print("No valid clash results to plot.")
        return

    # Check if CDZ is present in any frame
    has_cdz = any(r.get("has_cdz", True) for r in valid)

    # Extract data
    plddts = np.array([r["plddt"] for r in valid])
    het_het_lig = np.array([r["het_het_ligand"] for r in valid])
    het_het_ion = np.array([r["het_het_ion"] for r in valid])
    prot_het_lig = np.array([r["prot_het_ligand"] for r in valid])
    prot_het_ion = np.array([r["prot_het_ion"] for r in valid])
    cdz_self_intra = np.array([r["cdz_self_intra"] for r in valid])

    # Change from 1, 2 to 1, 3 and expand the width to 20
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    colors = {"ligand": "#1f77b4", "ion": "#ff7f0e"}
    marker_size = 100

    # --- Left panel: het-het clashes ---
    ax = axes[0]
    if has_cdz:
        ax.scatter(plddts, het_het_lig, s=marker_size, alpha=0.7, color=colors["ligand"],
                   label="Ligand (CDZ)", edgecolor="black", linewidth=0.5)
    ax.scatter(plddts, het_het_ion, s=marker_size, alpha=0.7, color=colors["ion"],
               label="Ion (CA)", edgecolor="black", linewidth=0.5)

    # Spearman correlations
    if has_cdz and len(het_het_lig) > 2:
        rho_lig, p_lig = spearmanr(plddts, het_het_lig)
    else:
        rho_lig, p_lig = np.nan, np.nan
    if len(het_het_ion) > 2:
        rho_ion, p_ion = spearmanr(plddts, het_het_ion)
    else:
        rho_ion, p_ion = np.nan, np.nan

    corr_text = (
        (f"Ligand: ρ = {rho_lig:.3f} (p={p_lig:.3f})\n" if has_cdz else "") +
        f"Ion: ρ = {rho_ion:.3f} (p={p_ion:.3f})"
    )
    ax.text(0.98, 0.97, corr_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("pLDDT", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"# Atoms with Clash (< {clash_threshold} Å)", fontsize=12, fontweight="bold")
    ax.set_title("Het-Het Clashes", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)

    # --- Right panel: protein-het clashes ---
    ax = axes[1]
    if has_cdz:
        ax.scatter(plddts, prot_het_lig, s=marker_size, alpha=0.7, color=colors["ligand"],
                   label="Ligand (CDZ)", edgecolor="black", linewidth=0.5)
    ax.scatter(plddts, prot_het_ion, s=marker_size, alpha=0.7, color=colors["ion"],
               label="Ion (CA)", edgecolor="black", linewidth=0.5)

    # Spearman correlations
    if has_cdz and len(prot_het_lig) > 2:
        rho_lig, p_lig = spearmanr(plddts, prot_het_lig)
    else:
        rho_lig, p_lig = np.nan, np.nan
    if len(prot_het_ion) > 2:
        rho_ion, p_ion = spearmanr(plddts, prot_het_ion)
    else:
        rho_ion, p_ion = np.nan, np.nan

    corr_text = (
        (f"Ligand: ρ = {rho_lig:.3f} (p={p_lig:.3f})\n" if has_cdz else "") +
        f"Ion: ρ = {rho_ion:.3f} (p={p_ion:.3f})"
    )
    ax.text(0.98, 0.97, corr_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("pLDDT", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"# Atoms with Clash (< {clash_threshold} Å)", fontsize=12, fontweight="bold")
    ax.set_title("Protein-Het Clashes", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)

    # --- Third panel: Self clashes (Intra-residue CDZ only) ---
    ax = axes[2]
    if has_cdz:
        ax.scatter(plddts, cdz_self_intra, s=marker_size, alpha=0.7, color=colors["ligand"],
                   label="Ligand (Intra-residue)", edgecolor="black", linewidth=0.5)

        # Spearman correlation
        if len(cdz_self_intra) > 2:
            rho_self, p_self = spearmanr(plddts, cdz_self_intra)
        else:
            rho_self, p_self = np.nan, np.nan

        corr_text = f"Ligand: ρ = {rho_self:.3f} (p={p_self:.3f})"
        ax.text(0.98, 0.97, corr_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("pLDDT", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"# Atoms with Clash (< {clash_threshold} Å)", fontsize=12, fontweight="bold")
    ax.set_title("Self Clashes (CDZ intra-residue)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)

    fig.suptitle(f"pLDDT vs Steric Clashes ({ensemble_label})",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(str(output_dir), f"plddt_vs_clashes.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\u2713 pLDDT vs clashes plot saved to: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ligand-protein contacts: references vs NeuralPLexer ensemble."
    )
    parser.add_argument("--ensemble-dir", required=True,
                        help="Path to combined_outputs/ directory from combine_outputs.py.")
    parser.add_argument("--pdb-codes", nargs="+", default=["7PU9", "7PSZ"],
                        help="Reference PDB codes to fetch from RCSB (default: 7PU9, 7PSZ).")
    parser.add_argument("--ensemble-label", default=None,
                        help="Label for the ensemble panel (default: derived from dir name).")
    parser.add_argument("--ligand-sdf", default="NeuralPLexer/APO-CaM/85H_ideal.sdf",
                        help="Path to the reference SDF file for the ligand to define bonds and exclude them from clashes.")
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir).resolve()
    if not ensemble_dir.is_dir():
        print(f"[ERROR] --ensemble-dir does not exist: {ensemble_dir}")
        return

    ens_label = args.ensemble_label or ensemble_dir.parent.name

    excluded_pairs = set()
    if args.ligand_sdf and os.path.exists(args.ligand_sdf):
        try:
            suppl = Chem.SDMolSupplier(args.ligand_sdf, sanitize=False, removeHs=False)
            mol = next(suppl)
            if mol is None:
                raise ValueError("RDKit could not parse SDF")
            
            # Identify heavy atoms and their original RDKit indices
            heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            heavy_rdkit_indices = [a.GetIdx() for a in heavy_atoms]
            # Map original RDKit index -> local heavy-atom index (0 to N-1)
            rdkit_to_local = {idx: i for i, idx in enumerate(heavy_rdkit_indices)}
            
            for i, atom_i in enumerate(heavy_atoms):
                excluded_pairs.add((i, i))
                # 1-2 bonds
                for neighbor in atom_i.GetNeighbors():
                    if neighbor.GetAtomicNum() > 1:
                        j = rdkit_to_local[neighbor.GetIdx()]
                        excluded_pairs.add((i, j))
                        # 1-3 bonds
                        for next_neighbor in neighbor.GetNeighbors():
                            if next_neighbor.GetAtomicNum() > 1:
                                k = rdkit_to_local[next_neighbor.GetIdx()]
                                excluded_pairs.add((i, k))
            print(f"Loaded bond topology from {args.ligand_sdf} ({len(heavy_atoms)} heavy atoms)")
        except Exception as e:
            print(f"[WARNING] Failed to parse SDF bonds with RDKit: {e}")
    else:
        print(f"[WARNING] SDF file not found at {args.ligand_sdf}. Intra-residue clashes will use a naive distance heuristic.")

    # =====================================================================
    # 1. Reference analysis
    # =====================================================================
    print("=" * 80)
    print("LIGAND-PROTEIN CONTACT ANALYSIS (CDZ / 85H)")
    print(f"Contact threshold: < {ligand_contact_threshold} \u00c5")
    print("=" * 80)

    ref_results = []
    for code in args.pdb_codes:
        print(f"\n--- {code.upper()} (reference) ---")
        r = calculate_contacts_reference(code)
        ref_results.append(r)
        if "error" in r:
            print(f"  Error: {r['error']}")
        else:
            print(f"  Ligand atoms : {r['total_ligand_atoms']}")
            print(f"  Contacts < {ligand_contact_threshold} \u00c5: "
                  f"{r['atoms_with_contact']} ({r['percentage']:.1f}%)")
            print(f"  Mean min dist: {r['mean_min_distance']:.3f} \u00c5")

    # =====================================================================
    # 2. Ensemble analysis
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"ENSEMBLE: {ens_label}")
    print("=" * 80)
    ens_results = calculate_ensemble_contacts(ensemble_dir)
    valid_ens = [r for r in ens_results if "error" not in r]
    print(f"\n  Valid frames with CDZ contacts: {len(valid_ens)} / {len(ens_results)}")
    if valid_ens:
        all_pcts = [r["percentage"] for r in valid_ens]
        print(f"  Contact % range: {min(all_pcts):.1f}% \u2013 {max(all_pcts):.1f}%  "
              f"(mean {np.mean(all_pcts):.1f}%)")

    # =====================================================================
    # 3. Plots – CDZ ligand contacts
    # =====================================================================
    print("\n" + "=" * 80)
    print("GENERATING CDZ CONTACT PLOTS")
    print("=" * 80)
    create_distance_histogram(ref_results, args.pdb_codes,
                              ens_results, ensemble_label=ens_label, output_dir=ensemble_dir)
    create_cumulative_contact_plot(ref_results, args.pdb_codes,
                                   ens_results, ensemble_label=ens_label, output_dir=ensemble_dir)

    # =====================================================================
    # 4. CA-oxygen analysis
    # =====================================================================
    print("\n" + "=" * 80)
    print("CA (CALCIUM) \u2013 OXYGEN CONTACT ANALYSIS")
    print("=" * 80)

    ref_ca = []
    for code in args.pdb_codes:
        print(f"\n--- {code.upper()} (reference) ---")
        r = calculate_ca_oxygen_reference(code)
        ref_ca.append(r)
        if "error" in r:
            print(f"  Error: {r['error']}")
        else:
            for resid, dists in r["by_resid"].items():
                n = len(dists)
                n_below = int(np.sum(np.array(dists) < ligand_contact_threshold))
                print(f"  CA resid {resid}: {n} O contacts, "
                      f"{n_below} < {ligand_contact_threshold} \u00c5 ({100*n_below/n:.1f}%)")

    print(f"\n--- Ensemble: {ens_label} ---")
    ens_ca = calculate_ensemble_ca(ensemble_dir)
    valid_ens_ca = [r for r in ens_ca if "error" not in r]
    print(f"  Valid frames with CA data: {len(valid_ens_ca)} / {len(ens_ca)}")

    create_ca_oxygen_cumulative_plot(ref_ca, args.pdb_codes,
                                     ens_ca, ensemble_label=ens_label, output_dir=ensemble_dir)
    create_ca_residue_contact_bar_plot(ref_ca, args.pdb_codes,
                                       ens_ca, ensemble_label=ens_label, output_dir=ensemble_dir)

    # =====================================================================
    # 5. pLDDT vs clashes analysis
    # =====================================================================
    print("\n" + "=" * 80)
    print("pLDDT vs STERIC CLASHES ANALYSIS")
    print(f"Clash threshold: < {clash_threshold} \u00c5")
    print("=" * 80)

    ens_clashes = calculate_ensemble_clashes(ensemble_dir, excluded_pairs)
    valid_ens_clashes = [r for r in ens_clashes if "error" not in r]
    print(f"\n  Valid frames with clash data: {len(valid_ens_clashes)} / {len(ens_clashes)}")

    create_plddt_vs_clashes_plot(ens_clashes, ensemble_label=ens_label, output_dir=ensemble_dir)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
