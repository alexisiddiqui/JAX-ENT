#!/usr/bin/env python3
"""Quick mock-ups of the manuscript aSyn figures (Fig 5, Fig 6, SI 10-16) from script-generated components.

Composites only: every panel is a PNG written by a script (render_asyn_pilot.py panels of a pilot campaign,
extract_PF.py, inertia_moments_clustering.py, compare_ensembles_intro_schema.py, analyse_MD.py) placed on a page
with a letter; nothing is drawn by hand.  Where the manuscript panel has no script-generated counterpart yet the
page says so (grey box) instead of substituting a look-alike.  Panels that stand in for a different quantity than
the manuscript's are labelled with what they actually show.

Usage:
  python jaxent/examples/4_aSyn/analysis/mockup_asyn_figures.py <campaign_dir> [--variant floor] [--out DIR]
Writes <campaign_dir>/_mockup_figures/{fig5,fig6,si10,...}.png and aSyn_figures_mockup.pdf (one page per figure).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
EX = HERE.parent
DATA = EX/"data"
sys.path.insert(0, str(HERE))
import render_asyn_pilot as R  # noqa: E402  (hexbin conventions)

CELLS = {"Tris": "TrisMD__Tris_only", "Control": "ControlMD__Tris_only",
         "Extracellular": "TrisMD__Extracellular", "Intracellular": "TrisMD__Intracellular"}
ENSEMBLES = {"Tris-MD": DATA/"_cluster_inertia", "Control-MD": DATA/"_cluster_inertia_control",
             "AF2-MSAss": DATA/"_cluster_inertia_af2", "Shaw-MD": DATA/"_cluster_inertia_shaw"}


# ----------------------------------------------------------------------------------------------- helpers
def _img(ax, path, letter=None, note=None, crop=None):
    """Place a PNG on an axis; missing file -> grey placeholder with the note."""
    ax.set_axis_off()
    path = Path(path) if path else None
    if path is not None and path.exists():
        im = plt.imread(path)
        if crop:  # fractional (top, bottom, left, right) crop
            h, w = im.shape[:2]
            t, b, l, r = crop
            im = im[int(t*h):int((1-b)*h), int(l*w):int((1-r)*w)]
        ax.imshow(im, interpolation="lanczos")
    else:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#e8e8e8"))
        ax.text(0.5, 0.5, f"not script-generated yet:\n{note or (path.name if path else '')}", ha="center",
                va="center", fontsize=8, color="#555", transform=ax.transAxes, wrap=True)
    if letter:
        ax.text(-0.04, 1.0, f"{letter}.", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")
    if note and (path is not None and path.exists()):
        ax.set_title(note, fontsize=7, color="#444", loc="left", pad=3)


def _page(pdf, fig, out, name, caption):
    fig.text(0.02, 0.012, caption, fontsize=7.5, wrap=True, va="bottom")
    fig.savefig(out/f"{name}.png", dpi=170, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved {out/f'{name}.png'}")


def ensemble_density_panel(path):
    """SI 11 counterpart: unweighted free-energy hexbins of the four ensembles' shape axes (same conventions as
    render_asyn_pilot: gridsize 20, F = -ln sum w - min, YlGnBu_r 0-8 kT)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    ref = R.reference_ratios()
    for ax, (name, d) in zip(axes.ravel(), ENSEMBLES.items()):
        sa = d/"shape_axes.npy"
        if not sa.exists():
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{name}: no shape_axes.npy", ha="center", transform=ax.transAxes)
            continue
        xy = np.load(sa)
        w = np.full(len(xy), 1/len(xy))
        F = R._free_energy(xy[:, 0], xy[:, 1], w)
        hb = ax.hexbin(xy[:, 0], xy[:, 1], C=w, reduce_C_function=np.sum, gridsize=R.GRIDSIZE, mincnt=1,
                       cmap="YlGnBu_r", edgecolors="none")
        hb.set_array(F)
        hb.set_norm(mcolors.Normalize(0, 8))
        R._shape_axes(ax, ref, legend=False)
        fig.colorbar(hb, ax=ax, shrink=0.8).set_label(r"$F/k_BT$ (uniform weights)")
        ax.set_title(f"{name}  (n = {len(xy):,})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------- figures
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--variant", default="floor")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    C = a.campaign_dir.resolve()
    out = (a.out or C/"_mockup_figures").resolve()
    out.mkdir(parents=True, exist_ok=True)
    P = {k: C/v/a.variant/"plots" for k, v in CELLS.items()}
    tag = f"{C.name} [{a.variant}]"
    pdf = PdfPages(out/"aSyn_figures_mockup.pdf")

    # Figure 5 ------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 13.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 1.1, 1.0], hspace=0.25, wspace=0.25)
    _img(fig.add_subplot(gs[0, :]), DATA/"_figures"/"compare_ensembles"/"coverage_overlay_99pct.png", "A",
         "99th-percentile coverage, 4 ensembles (compare_ensembles_intro_schema.py; May run, Control-MD pre-repair)")
    _img(fig.add_subplot(gs[1, 0:2]), P["Tris"]/"d_landscape_selected.png", "B", "rule-D weighted landscape (mean of folds)")
    _img(fig.add_subplot(gs[1, 2:4]), P["Tris"]/"d_landscape_dF.png", "C", "F_sel - F_prior")
    _img(fig.add_subplot(gs[1, 4:6]), P["Tris"]/"g_structure_ruleD.png", "D",
         "stand-in: rule-D ln PF on 2KKW (manuscript: top-20 weighted structures + Tris)", crop=(0.08, 0.08, 0.02, 0.02))
    _img(fig.add_subplot(gs[2, 0:2]), P["Tris"]/"g_structure_ruleD_minus_prior.png", "E",
         "rule-D minus prior ln PF on 2KKW (manuscript: error vs experiment)", crop=(0.08, 0.08, 0.02, 0.02))
    _img(fig.add_subplot(gs[2, 2:4]), P["Tris"]/"d_landscape_foldsigma.png", "F", "SD of F_sel over folds")
    _img(fig.add_subplot(gs[2, 4:6]), P["Tris"]/"g_structure_ruleD_foldSD.png", "G", "fold SD of ln PF on 2KKW",
         crop=(0.08, 0.08, 0.02, 0.02))
    for col, nm in ((0, "ruleD_minus_prior"), (4, "ruleD_foldSD")):
        cax = fig.add_axes([0.14+col*0.123, 0.085, 0.11, 0.03])
        _img(cax, P["Tris"]/f"g_colourbar_{nm}.png")
    fig.suptitle(f"Figure 5 mock-up: Tris-MD x Tris  ({tag})", fontsize=11)
    _page(pdf, fig, out, "fig5", "Fig 5 mock-up. A: existing script output (not re-run). B, C, F: render_asyn_pilot "
          "shape-space panels (rule-D selection, 4 leave-one-block-out folds). D, E, G: headless PyMOL projections "
          "of per-residue ln PF (prior-relative; the manuscript's experiment-relative error panel needs residue "
          "targets on the same scale). Not reproduced: structure insets, top-20 structure render, Tris molecules.")

    # Figure 6 ------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(1, 4, wspace=0.2)
    for i, (L, cell, nm, note) in enumerate((("A", "Tris", "d_landscape_foldsigma", "Tris+Tris-MD  fold SD of F"),
                                             ("B", "Control", "d_landscape_foldsigma", "Tris+Control-MD  fold SD of F"),
                                             ("C", "Tris", "d_landscape_dF", "Tris+Tris-MD  F_sel - F_prior"),
                                             ("D", "Control", "d_landscape_dF", "Tris+Control-MD  F_sel - F_prior"))):
        _img(fig.add_subplot(gs[0, i]), P[cell]/f"{nm}.png", L, note)
    fig.suptitle(f"Figure 6 mock-up: 1.0 us row only  ({tag})", fontsize=11)
    _page(pdf, fig, out, "fig6", "Fig 6 mock-up. Only the full-length (1.0 us) ensembles have been fitted in this "
          "campaign; the 0.25 / 0.5 us truncation rows need the same driver run on features_0.25us / features_0.5us "
          "(control clustering for those truncations is regenerated and ready). Uncertainty here is SD over the four "
          "peptide-block folds, not over optimisation replicates.")

    # SI 10 ---------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    _img(fig.add_subplot(111), DATA/"_aSyn"/"aSyn_metrics_comparison.png", None,
         "extract_PF.py (2026-11-07: per-ms rates converted to s^-1; ln PF now O(1))")
    fig.suptitle("SI Fig 10 mock-up: observed rates, intrinsic rates, residue ln PF per condition", fontsize=11)
    _page(pdf, fig, out, "si10", "SI 10. Regenerated by extract_PF.py with the corrected units; Lysosomal still uses "
          "the workbook rates as read (its time scale is unresolved).")

    # SI 11 ---------------------------------------------------------------------------------------------
    dens = ensemble_density_panel(out/"_si11_ensemble_density.png")
    fig = plt.figure(figsize=(9.5, 9))
    _img(fig.add_subplot(111), dens, None, "unweighted shape-space free energy, four ensembles (shape_axes.npy of each clustering dir)")
    fig.suptitle("SI Fig 11 mock-up: unweighted conformational density of candidate ensembles", fontsize=11)
    _page(pdf, fig, out, "si11", "SI 11. Hexbin conventions of render_asyn_pilot (gridsize 20, F = -ln density, 0-8 kT); "
          "Control-MD uses the regenerated 108,054-frame clustering.")

    # SI 12 (not available) ------------------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 5))
    _img(fig.add_subplot(111), None, None, "SI 12 weighted bound-Tris per residue: needs tris_residue_contacts.npz "
         "(restore from tris_MD/_pre_topup/features/ or recompute with compute_tris_residue_contacts.py) and a "
         "weighted-contact panel in the renderer")
    fig.suptitle("SI Fig 12: not script-generated yet", fontsize=11)
    _page(pdf, fig, out, "si12", "SI 12 placeholder.")

    # SI 13 ---------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(4, 1, hspace=0.3)
    for i, (L, cell) in enumerate((("A", "Tris"), ("B", "Extracellular"), ("C", "Intracellular"), ("D", "Control"))):
        _img(fig.add_subplot(gs[i, 0]), P[cell]/"f_residue_lnpf.png", L,
             f"{CELLS[cell]}: prior vs rule-D ln PF (fold range), peptide ln(mean k_int / k_fit) bars")
    fig.suptitle(f"SI Fig 13 mock-up: residue ln PF per condition  ({tag})", fontsize=11)
    _page(pdf, fig, out, "si13", "SI 13 mock-up. Rows: Tris, Extracellular, Intracellular (Tris-MD) and Tris (Control-MD). "
          "No Lysosomal fit (time scale unresolved). Experimental values are peptide-level (raw uptake fits), not the "
          "residue targets of the manuscript.")

    # SI 14 ---------------------------------------------------------------------------------------------
    IP = DATA/"_cluster_inertia"/"plots"
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.2)
    _img(fig.add_subplot(gs[:, 0]), IP/"ctail_rg_macro_rows.png", "A-C", "C-tail Rg per macro-cluster (inertia_moments_clustering.py)")
    _img(fig.add_subplot(gs[0, 1]), IP/"macro_hexbin_energy.png", "D", "macro-cluster shape space")
    _img(fig.add_subplot(gs[1, 1]), P["Tris"]/"e_macro_fractions.png", "E/F", "macro fractions, prior vs rule D (fold SD)")
    fig.suptitle("SI Fig 14 mock-up: C-tail Rg by cluster and macro-cluster shape space", fontsize=11)
    _page(pdf, fig, out, "si14", "SI 14 mock-up. A-D from the Tris-MD inertia clustering outputs (unweighted); E/F replaces the "
          "manuscript's weighted bar with the pilot's prior-vs-selected macro fractions.")

    # SI 15 ---------------------------------------------------------------------------------------------
    conds = ("Tris", "Extracellular", "Intracellular")
    fig = plt.figure(figsize=(12, 11.5))
    gs = fig.add_gridspec(3, 3, wspace=0.12, hspace=0.18)
    for r, (L, nm, lab) in enumerate((("A", "d_landscape_selected", "weighted landscape"),
                                     ("B", "d_landscape_dF", "F_sel - F_prior"),
                                     ("C", "d_landscape_foldsigma", "fold SD"))):
        for c, cond in enumerate(conds):
            _img(fig.add_subplot(gs[r, c]), P[cond]/f"{nm}.png", L if c == 0 else None, f"{cond}: {lab}")
    fig.suptitle(f"SI Fig 15 mock-up: landscapes, weight change, uncertainty across conditions  ({tag})", fontsize=11)
    _page(pdf, fig, out, "si15", "SI 15 mock-up. Columns Tris / Extracellular / Intracellular (Tris-MD ensemble); "
          "no Lysosomal column.")

    # SI 16 ---------------------------------------------------------------------------------------------
    MD = DATA/"_figures"/"md_analysis"/"tris_MD"
    ts = sorted(MD.glob("timeseries_*.png"))
    fig = plt.figure(figsize=(11, 3.6*max(1, min(3, len(ts)))))
    if ts:
        gs = fig.add_gridspec(min(3, len(ts)), 1, hspace=0.25)
        for i, p in enumerate(ts[:3]):
            _img(fig.add_subplot(gs[i, 0]), p, "ABC"[i], p.stem)
    else:
        _img(fig.add_subplot(111), None, None, "analyse_MD.py timeseries_*.png not found in data/_figures/md_analysis/tris_MD")
    fig.suptitle("SI Fig 16 mock-up: Tris-MD trajectory analysis (analyse_MD.py, May run)", fontsize=11)
    _page(pdf, fig, out, "si16", "SI 16. Existing analyse_MD.py outputs (not re-run).")

    # Extra: pilot diagnostics page (new, not in the manuscript) ---------------------------------------
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 0.8], hspace=0.2, wspace=0.08)
    _img(fig.add_subplot(gs[0, :]), P["Tris"]/"a_uptake_grid.png", "A", "peptide uptake: data, prior, rule-D per fold")
    _img(fig.add_subplot(gs[1, 0]), P["Tris"]/"b_zscore_heatmap.png", "B", "z-scores (held-out rows boxed)")
    _img(fig.add_subplot(gs[1, 1]), P["Tris"]/"d_landscape_prior.png", "C", "prior landscape")
    _img(fig.add_subplot(gs[2, :]), P["Tris"]/"c_selection_vs_S.png", "D", "ESS / chi2 / flags / F vs S")
    fig.suptitle(f"Extra: pilot diagnostics, Tris-MD x Tris  ({tag})", fontsize=11)
    _page(pdf, fig, out, "extra_diagnostics", "Diagnostic panels that have no manuscript counterpart.")

    pdf.close()
    print(f"Saved {out/'aSyn_figures_mockup.pdf'}")


if __name__ == "__main__":
    main()
