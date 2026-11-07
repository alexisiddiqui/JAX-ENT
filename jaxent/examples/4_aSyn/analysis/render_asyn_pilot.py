#!/usr/bin/env python3
"""Separate, composable PNG panels for the aSyn uptake pilot (fitting/run_asyn_uptake_sidecars.py).

Per cell and noise variant, reads <cell>/<variant>/{pilot_selected.npz, candidates.csv, fold_table.csv} and
<cell>/{prep_arrays.npz, prep_peptides.csv}; writes <cell>/<variant>/plots/:
  a_uptake_grid.png            30 peptides: data +- sigma, prior, rule-D selection per fold (bold = held-out fold);
                               tail peptides (start >= 114, not fitted) shaded
  b_zscore_heatmap.png         z = (obs - pred)/sigma, peptide x timepoint: prior, each fold, cross-validated;
                               held-out rows boxed, Sidak-flagged held-out cells dotted
  c_selection_vs_S.png         per fold vs maxent_scaling S: ESS, chi2/N train, held-out chi2/N, flags, F
  d_landscape_prior.png        inertia shape-space free energy, uniform weights (YlGnBu_r, 0-8 kT)     [Tris-MD only]
  d_landscape_selected.png     same, mean of the rule-D fold weights                                      [Tris-MD only]
  d_landscape_dF.png           F_sel - F_prior (PuOr, +-8 kT)                                             [Tris-MD only]
  d_landscape_foldsigma.png    SD over folds of F_sel (Reds)                                              [Tris-MD only]
  e_macro_fractions.png        Rod/Wavy/Compact/Unassigned weighted fractions, prior vs selected (fold SD)  [Tris-MD only]
  f_residue_lnpf.png           rate-averaged per-residue ln PF (incl. delta), prior vs selected (fold range),
                               experimental peptide ln PF = ln(mean k_int / k_fit) as bars
  g_structure_<metric>.png     headless PyMOL (jaxent/src/analysis/pymol/Project_logPFs.py through
                               common/pymol_headless.py, view/style from analysis/pymol_viz_configs/
                               project_logPF_Tris_SD.yaml, 2kkw.pdb): prior ln PF, rule-D mean ln PF,
                               rule-D minus prior, fold SD; each with a matching g_colourbar_<metric>.png.
                               Inputs written to pymol/ (npz with pred_ln_pf, prior .dat).  Skipped with a
                               message when no PyMOL binary is found (JAXENT_PYMOL / PYMOL_ANA env).
Hexbin landscape conventions follow analysis/plot_feature_distributions_aSyn_conditions.py
(gridsize 20, F = -ln sum w - min, triangle Rod/Sphere/Disk) and data/inertia_moments_clustering.py.
Shape-space panels use data/_cluster_inertia (Tris-MD) or data/_cluster_inertia_control (Control-MD,
projected onto the Tris GMM; regenerated 2026-11-07 for the repaired 108,054-frame ensemble).

Usage: python jaxent/examples/4_aSyn/analysis/render_asyn_pilot.py <campaign_dir>/<cell> [--variant floor]
       [--no-pymol]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
EX = HERE.parent
REPO = EX.parents[2]
INERTIA = {"TrisMD": EX/"data"/"_cluster_inertia", "ControlMD": EX/"data"/"_cluster_inertia_control"}
PYMOL_SCRIPT = REPO/"jaxent"/"src"/"analysis"/"pymol"/"Project_logPFs.py"
PYMOL_STYLE = HERE/"pymol_viz_configs"/"project_logPF_Tris_SD.yaml"  # view matrix + putty/ray settings
PYMOL_TOPOLOGY = EX/"data"/"_aSyn"/"tris_MD"/"md_mol_center_coil.pdb"  # full-sequence residue map (134 non-Pro)
PYMOL_COLOURS = {"green": "#00c000", "white": "#ffffff", "purple": "#bf00bf", "blue": "#0000ff", "red": "#ff0000",
                 "orange": "#ff8000", "yellow": "#ffff00", "magenta": "#ff00ff", "cyan": "#00ffff", "grey": "#808080"}
STRUCTURE_PANELS = (  # name, metric, target, reference, spectrum, (vmin, vmax), label
    ("prior", "protection_factor", "prior", None, "white_blue", (0.0, 6.0), "prior ln PF (uniform weights, incl. delta)"),
    ("ruleD", "protection_factor", "ruleD", None, "white_blue", (0.0, 6.0), "rule-D ln PF (mean over folds)"),
    ("ruleD_minus_prior", "difference_signed", "ruleD", "prior", "orange_white_blue", (-2.0, 2.0),
     "rule-D minus prior ln PF"),
    ("ruleD_foldSD", "uncertainty_sd", "ruleD", None, "white_red", (0.0, 1.0), "rule-D ln PF SD over folds"),
)
FOLD_COLOURS = {"A_1-38": "#1b9e77", "B_39-69": "#d95f02", "C_55-94": "#7570b3", "D_90-113": "#e7298a"}
MACRO = ["Rod", "Wavy", "Compact", "Unassigned"]
MACRO_COLOURS = {"Rod": "grey", "Wavy": "blue", "Compact": "orange", "Unassigned": "lightgrey"}
GRIDSIZE = 20


def _load(cell_dir: Path, variant: str):
    vd = cell_dir/variant
    S = dict(np.load(vd/"pilot_selected.npz", allow_pickle=True))
    prep = dict(np.load(cell_dir/"prep_arrays.npz", allow_pickle=True))
    cand = pd.read_csv(vd/"candidates.csv")
    sel = pd.read_csv(vd/"fold_table.csv")
    out = vd/"plots"
    out.mkdir(parents=True, exist_ok=True)
    return S, prep, cand, sel, out


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _heldout_fold_of(blocks, folds):
    return {i: b for i, b in enumerate(blocks) if b in folds}


# --------------------------------------------------------------------------------------------- (a)
def plot_uptake_grid(S, prep, title, path):
    t = prep["times"]
    y, sig, blocks, labels = S["y"], S["sigma"], S["blocks"], S["labels"]
    folds = list(S["folds"])
    P = S["pred_ruleD"]
    n = len(labels)
    ncol = 6
    nrow = int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6*ncol, 2.2*nrow), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        tail = blocks[i] == "tail"
        if tail:
            ax.set_facecolor("#f0f0f0")
        for k, f in enumerate(folds):
            held = blocks[i] == f
            ax.plot(t, P[k, i], color=FOLD_COLOURS.get(f, "C0"), lw=2.2 if held else 0.7,
                    alpha=1 if held else 0.6, zorder=3 if held else 2)
        ax.plot(t, S["pred_prior"][i], color="k", ls="--", lw=1.0, zorder=2)
        ax.errorbar(t, y[i], yerr=sig[i], fmt="o", ms=3, color="k", capsize=1.5, lw=0.8, zorder=4)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"{labels[i]}" + (" (tail)" if tail else f" [{blocks[i][0]}]"), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("D (fraction)", fontsize=8)
    handles = [plt.Line2D([], [], color="k", ls="--", label="prior (uniform)"),
               plt.Line2D([], [], color="k", marker="o", ls="", label="data +- sigma")]
    handles += [plt.Line2D([], [], color=FOLD_COLOURS[f], lw=2, label=f"rule D, fold {f} (bold = its held-out block)")
                for f in folds]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


# --------------------------------------------------------------------------------------------- (b)
def plot_z_heatmap(S, prep, title, path):
    y, sig, blocks, labels = S["y"], S["sigma"], S["blocks"], list(S["labels"])
    folds = list(S["folds"])
    t = prep["times"]
    zp = (y-S["pred_prior"])/sig
    zf = [(y-S["pred_ruleD"][k])/sig for k in range(len(folds))]
    zcv = zp.copy()*np.nan
    for i, b in enumerate(blocks):
        if b in folds:
            zcv[i] = zf[folds.index(b)][i]
        else:
            zcv[i] = np.mean([z[i] for z in zf], 0)
    panels = [("prior", zp, None)] + [(f"rule D\nfold {f}", zf[k], f) for k, f in enumerate(folds)] + \
             [("cross-validated\n(held-out fold per row)", zcv, "cv")]
    vmax = max(np.nanmax(np.abs(z)) for _, z, _ in panels)
    normz = mcolors.SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3*len(panels), 8), sharey=True)
    for ax, (name, z, f) in zip(axes, panels):
        im = ax.imshow(z, aspect="auto", cmap="RdBu_r", norm=normz)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([f"{v:g}" for v in t], rotation=90, fontsize=7)
        ax.set_title(name, fontsize=8)
        rows = [i for i, b in enumerate(blocks) if (b == f) or (f == "cv" and b != "tail")]
        if f is not None and rows:
            ntot = len(rows)*len(t) if f != "cv" else None
            for blk in (folds if f == "cv" else [f]):
                rr = [i for i, b in enumerate(blocks) if b == blk]
                ax.add_patch(Rectangle((-0.5, min(rr)-0.5), len(t), len(rr), fill=False, lw=1.5, ec="k"))
                tn = norm.ppf(0.5+0.5*(0.95)**(1/(len(rr)*len(t))))
                for i in rr:
                    for j in range(len(t)):
                        if abs(z[i, j]) > tn:
                            ax.plot(j, i, "k.", ms=3)
        tail_rows = [i for i, b in enumerate(blocks) if b == "tail"]
        if tail_rows:
            ax.axhline(min(tail_rows)-0.5, color="grey", lw=1.2, ls="--")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels([l+(" *" if b == "tail" else "") for l, b in zip(labels, blocks)], fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("z = (obs - pred) / sigma  (symlog)")
    fig.suptitle(title+"\nboxes: held-out block; dots: Sidak-flagged |z| (alpha 0.05 per block); * tail (not fitted)",
                 fontsize=10)
    _save(fig, path)


# --------------------------------------------------------------------------------------------- (c)
def plot_selection(cand, sel, title, path):
    pool = cand[cand.in_pool]
    Ss = sorted(pool.strength[pool.strength > 0].unique())
    x0 = Ss[0]/3 if Ss else 1.0
    xs = lambda s: np.where(s > 0, s, x0)
    metrics = [("ESS", "ESS (fraction)", True), ("chi2_train", "train chi2/N", True), ("ho_chi2", "held-out chi2/N", True),
               ("ho_n_flags", "held-out flags", False), ("F", "F = 1/2 sum z^2 + KL", True)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 3.4))
    for fold, g in pool.groupby("fold", sort=False):
        c = FOLD_COLOURS.get(fold, "C0")
        fin = g[(g.kind == "final") | (g.kind == "prior")].sort_values("strength")
        for ax, (m, lab, logy) in zip(axes, metrics):
            ax.plot(xs(fin.strength), fin[m], "-", color=c, lw=1.2, label=fold)
            oth = g[g.kind == "convergence"]
            ax.plot(xs(oth.strength), oth[m], ".", color=c, ms=3, alpha=0.6)
            d = sel[(sel.fold == fold) & (sel.rule == "D")].iloc[0]
            cc = sel[(sel.fold == fold) & (sel.rule == "C")].iloc[0]
            ax.plot(xs(np.array([d.strength])), [d[m]], "*", color=c, ms=13, mec="k", zorder=5)
            ax.plot(xs(np.array([cc.strength])), [cc[m]], "o", mfc="none", mec=c, ms=10, mew=1.5, zorder=5)
            if logy:
                ax.set_yscale("log")
    for ax, (m, lab, _) in zip(axes, metrics):
        ax.set_xscale("log")
        ax.set_xlabel("maxent_scaling S (KL weight 1/S); left point = prior")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle(title+"   (lines: final state per S; dots: convergence checkpoints; star: rule D; ring: rule C)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------------------------- (d)
def _hex_sums(x, y, w):
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=GRIDSIZE, mincnt=1)
    vals = hb.get_array().copy()
    plt.close(fig)
    return vals


def _free_energy(x, y, w):
    with np.errstate(divide="ignore"):
        F = -np.log(_hex_sums(x, y, w))
    return F-np.nanmin(F)


def _shape_axes(ax, ref_ratios=None, legend=False):
    ax.plot([0, 1, 0.5, 0], [1, 1, 0.5, 1], "k--", lw=1.5, zorder=5)
    kw = dict(fontsize=10, zorder=6, path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
    ax.text(0, 1.04, "Rod", ha="center", **kw)
    ax.text(1, 1.04, "Sphere", ha="center", **kw)
    ax.text(0.5, 0.46, "Disk", ha="center", **kw)
    ax.plot([0.5, 1], [0.5, 1], color="gray", lw=1, ls=":", alpha=0.7, zorder=4)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(0.42, 1.10)
    ax.set_xlabel(r"$I_1/I_3$")
    ax.set_ylabel(r"$I_2/I_3$")
    if ref_ratios:
        try:
            from plot_feature_distributions_aSyn_conditions import _overlay_references
            _overlay_references(ax, ref_ratios, is_legend=legend)
            if legend:
                ax.legend(fontsize=7, loc="lower right")
        except Exception as exc:  # pragma: no cover - optional overlay
            print(f"reference overlay skipped: {exc}")


def _landscape(x, y, values, cmap, normc, label, title, path, ref_ratios, w_for_bins):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    hb = ax.hexbin(x, y, C=w_for_bins, reduce_C_function=np.sum, gridsize=GRIDSIZE, mincnt=1, cmap=cmap,
                   edgecolors="none")
    hb.set_array(values)
    hb.set_norm(normc)
    _shape_axes(ax, ref_ratios, legend=True)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label(label)
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def reference_ratios():
    try:
        import sys
        sys.path.insert(0, str(HERE))
        from plot_feature_distributions_aSyn_conditions import REFERENCES, compute_reference_inertia_ratios
        return compute_reference_inertia_ratios(REFERENCES, "resid 1-114 and name CA")
    except Exception as exc:  # pragma: no cover
        print(f"reference ratios unavailable: {exc}")
        return None


def plot_landscapes(S, title, out, shape_axes, ref_ratios):
    W = S["weights_ruleD"]
    x, y = shape_axes[:, 0], shape_axes[:, 1]
    F_ = W.shape[1]
    uni = np.full(F_, 1/F_)
    Fp = _free_energy(x, y, uni)
    Fs = _free_energy(x, y, W.mean(0))
    Ff = np.stack([_free_energy(x, y, w) for w in W])
    level = mcolors.BoundaryNorm(list(range(9)), ncolors=256, extend="max")
    _landscape(x, y, Fp, "YlGnBu_r", level, r"$\Delta F / k_BT$", f"{title}\nprior (uniform)",
               out/"d_landscape_prior.png", ref_ratios, uni)
    _landscape(x, y, Fs, "YlGnBu_r", mcolors.BoundaryNorm(list(range(9)), ncolors=256, extend="max"),
               r"$\Delta F / k_BT$", f"{title}\nrule D (mean of {len(W)} fold weights)", out/"d_landscape_selected.png",
               ref_ratios, uni)
    _landscape(x, y, Fs-Fp, "PuOr", mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8),
               r"$\Delta\Delta F / k_BT$ (selected - prior)", f"{title}\nF_sel - F_prior", out/"d_landscape_dF.png",
               ref_ratios, uni)
    sd = Ff.std(0)
    _landscape(x, y, sd, "Reds", mcolors.Normalize(vmin=0, vmax=max(2.0, float(np.nanpercentile(sd, 99)))),
               r"$\sigma_{folds}(\Delta F) / k_BT$", f"{title}\nSD of F over the {len(W)} fold selections",
               out/"d_landscape_foldsigma.png", ref_ratios, uni)
    return dict(dF_min=float(np.nanmin(Fs-Fp)), dF_max=float(np.nanmax(Fs-Fp)), foldsigma_max=float(np.nanmax(sd)))


# --------------------------------------------------------------------------------------------- (e)
def plot_macro(S, title, path, macro):
    W = S["weights_ruleD"]
    F_ = W.shape[1]
    prior = np.array([np.mean(macro == m) for m in MACRO])
    fr = np.array([[w[macro == m].sum() for m in MACRO] for w in W])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xx = np.arange(len(MACRO))
    ax.bar(xx-0.2, prior, 0.38, color=[MACRO_COLOURS[m] for m in MACRO], alpha=0.45, edgecolor="k", label="prior")
    ax.bar(xx+0.2, fr.mean(0), 0.38, yerr=fr.std(0), capsize=4, color=[MACRO_COLOURS[m] for m in MACRO],
           edgecolor="k", hatch="//", label="rule D (mean +- fold SD)")
    for i in range(len(MACRO)):
        ax.text(xx[i]+0.2, fr.mean(0)[i]+fr.std(0)[i]+0.01, f"{fr.mean(0)[i]:.2f}", ha="center", fontsize=7)
        ax.text(xx[i]-0.2, prior[i]+0.01, f"{prior[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xx)
    ax.set_xticklabels(MACRO)
    ax.set_ylabel("weighted fraction")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)
    return dict(zip(MACRO, prior.round(3).tolist())), {m: (float(fr.mean(0)[i]), float(fr.std(0)[i])) for i, m in enumerate(MACRO)}


# --------------------------------------------------------------------------------------------- (f)
def plot_lnpf(S, prep, title, path):
    res = prep["res"]
    L = S["lnpf_ruleD"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(res, S["lnpf_prior"], color="k", lw=1, ls="--", label="prior (uniform), incl. delta")
    ax.fill_between(res, L.min(0), L.max(0), color="C0", alpha=0.25, lw=0)
    ax.plot(res, L.mean(0), color="C0", lw=1.3, label="rule D (mean; band = fold range)")
    for s, e, v, b in zip(prep["starts"], prep["ends"], prep["lnpf_exp"], prep["blocks"]):
        ax.plot([s+0.5, e+0.5], [v, v], color="grey" if b == "tail" else FOLD_COLOURS.get(b, "C3"), lw=2.2, alpha=0.8)
    ax.plot([], [], color="C3", lw=2.2, label="exp. peptide ln(mean k_int / k_fit) (colour = block; grey = tail)")
    ax.axvspan(113.5, 140.5, color="#f0f0f0", zorder=0)
    ax.set_xlabel("residue")
    ax.set_ylabel("ln PF (rate-averaged)")
    ax.set_xlim(1, 141)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(title+f"   (delta = {float(S['delta']):.2f})", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


def _colourbar(spectrum, vmin, vmax, label, path):
    """Matplotlib colour bar matching a PyMOL ``a_b[_c]`` spectrum name."""
    cmap = mcolors.LinearSegmentedColormap.from_list(spectrum, [PYMOL_COLOURS[c] for c in spectrum.split("_")])
    fig, ax = plt.subplots(figsize=(3.2, 0.55))
    fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap), cax=ax, orientation="horizontal")
    ax.set_title(label, fontsize=7)
    ax.tick_params(labelsize=7)
    _save(fig, path)


def plot_structures(S, prep, cell_dir: Path, out: Path) -> dict:
    """Headless PyMOL putty/spectrum renders of per-residue ln PF (see module docstring)."""
    from jaxent.examples.common.pymol_headless import find_pymol, run_pymol_script
    pd_ = out/"pymol"
    pd_.mkdir(exist_ok=True)
    res = np.asarray(prep["res"], int)
    np.savez(pd_/"prior.npz", pred_ln_pf=np.asarray(S["lnpf_prior"], float)[None, :], residues=res)
    np.savez(pd_/"ruleD.npz", pred_ln_pf=np.asarray(S["lnpf_ruleD"], float), residues=res, folds=S["folds"])
    np.savetxt(pd_/"prior.dat", np.c_[res, np.asarray(S["lnpf_prior"], float)], fmt="%d %.6f",
               header="residue_number log_pf  (prior, uniform weights, incl. delta)")
    done = {}
    if find_pymol() is None:
        print(f"{cell_dir.name}: structure panels skipped (no PyMOL binary; set JAXENT_PYMOL)")
        return done
    for name, metric, target, reference, spectrum, (lo, hi), label in STRUCTURE_PANELS:
        args = ["--config", PYMOL_STYLE, "--working_dir", EX, "--target_topology", PYMOL_TOPOLOGY,
                "--target_data", pd_/f"{target}.npz", "--pf_metric", metric, "--spectrum_colours", spectrum,
                f"--spectrum_range={lo},{hi}", "--save_png", "--output_prefix", pd_/name]
        if reference is not None:
            args += ["--reference_data", pd_/f"{reference}.dat"]
        try:
            run_pymol_script(PYMOL_SCRIPT, args, log=pd_/"pymol.log")
        except Exception as exc:  # noqa: BLE001 - report and continue with the other panels
            print(f"{cell_dir.name}: PyMOL panel {name} failed: {exc}")
            continue
        png = pd_/f"{name}_logPFs_{metric}.png"
        if png.exists():
            png.replace(out/f"g_structure_{name}.png")
            _colourbar(spectrum, lo, hi, label, out/f"g_colourbar_{name}.png")
            done[name] = str(out/f"g_structure_{name}.png")
            print(f"Saved {out/f'g_structure_{name}.png'}")
        else:
            print(f"{cell_dir.name}: PyMOL panel {name} produced no PNG (see {pd_/'pymol.log'})")
    return done


def render_cell(cell_dir: Path, variant: str = "floor", pymol: bool = True) -> dict:
    cell_dir = Path(cell_dir)
    S, prep, cand, sel, out = _load(cell_dir, variant)
    title = f"{cell_dir.name} [{variant}]"
    plot_uptake_grid(S, prep, title, out/"a_uptake_grid.png")
    plot_z_heatmap(S, prep, title, out/"b_zscore_heatmap.png")
    plot_selection(cand, sel, title, out/"c_selection_vs_S.png")
    plot_lnpf(S, prep, title, out/"f_residue_lnpf.png")
    summary = {}
    inertia = INERTIA[cell_dir.name.split("__")[0]]
    shape = np.load(inertia/"shape_axes.npy")
    if len(shape) == S["weights_ruleD"].shape[1]:
        summary["landscape"] = plot_landscapes(S, title, out, shape, reference_ratios())
        macro = np.load(inertia/"macro_cluster_labels.npy", allow_pickle=True)
        summary["macro_prior"], summary["macro_selected"] = plot_macro(S, title, out/"e_macro_fractions.png", macro)
    else:
        print(f"{cell_dir.name}: shape-space panels skipped (clustering frames {len(shape)} in {inertia.name} vs "
              f"{S['weights_ruleD'].shape[1]} ensemble frames)")
    if pymol:
        summary["structures"] = plot_structures(S, prep, cell_dir, out)
    pd.Series({k: str(v) for k, v in summary.items()}).to_json(out/"render_summary.json", indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cell_dir", type=Path)
    ap.add_argument("--variant", default="floor")
    ap.add_argument("--no-pymol", action="store_true", help="skip the headless PyMOL structure panels")
    a = ap.parse_args()
    render_cell(a.cell_dir, a.variant, pymol=not a.no_pymol)


if __name__ == "__main__":
    main()
