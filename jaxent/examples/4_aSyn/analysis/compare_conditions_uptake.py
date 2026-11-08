#!/usr/bin/env python3
"""Compare the reweighted ensembles across conditions against a 'no fit' baseline (uniform weights, same
calibration), using the pilot_selected.npz written by run_asyn_uptake_sidecars.py --phase analyze.

For every cell (ensemble__condition) and noise variant:
  * per-region residuals (fraction uptake, predicted - observed) for the prior and the rule-D fits;
  * held-out chi2/N, flags and ESS from fold_table.csv;
  * regional mean ln PF of the prior and of the fold-averaged rule-D ensemble.
Across conditions (pairs with Tris_only as reference):
  * observed d_uptake = y_cond - y_Tris per peptide and time point, predicted d_uptake for the prior (identical
    ensemble: differences come from k_int only) and for the fits (fold-averaged), their correlation and the
    regional means; the paper's regions: N-term motifs 10-33 / 46-60, NAC 61-95, C 90-113, tail 114-140.
Writes <out>/condition_comparison.md and condition_comparison.csv.

Usage: python compare_conditions_uptake.py <pilot output dir> [--variant floor] [--ensemble TrisMD]
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

REGIONS = {"N 2-38": (2, 38), "N 39-60": (39, 60), "NAC 61-95": (61, 95), "C 90-113": (90, 113), "tail 114-140": (114, 140)}
PAPER_RES = {"nucleation 115-135": range(115, 136), "elongation N 10-33": range(10, 34), "elongation N 46-60": range(46, 61),
             "elongation NAC 61-95": range(61, 96), "elongation C 101-113": range(101, 114)}


def region_of(start, end):
    for name, (a, b) in REGIONS.items():
        if start >= a and end <= b:
            return name
    return "C 90-113" if start >= 90 and end < 114 else ("tail 114-140" if start >= 114 else "N 2-38" if end <= 38 else "mixed")


def load_cell(out, cell, variant):
    d = out/cell/variant
    z = np.load(d/"pilot_selected.npz", allow_pickle=True)
    ft = pd.read_csv(d/"fold_table.csv")
    info = json.loads((out/cell/"prep_info.json").read_text())
    arr = np.load(out/cell/"prep_arrays.npz", allow_pickle=True)
    return dict(z=z, ft=ft, info=info, arr=arr, labels=[str(x) for x in z["labels"]], starts=arr["starts"], ends=arr["ends"],
                res=arr["res"], y=z["y"], sigma=z["sigma"], pred_prior=z["pred_prior"], pred_fit=z["pred_ruleD"].mean(0),
                pred_fit_folds=z["pred_ruleD"], lnpf_prior=z["lnpf_prior"], lnpf_fit=z["lnpf_ruleD"].mean(0),
                lnpf_fit_folds=z["lnpf_ruleD"], ess=z["weights_ruleD"].shape[1]*0+np.array([1/np.sum(w**2)/len(w) for w in z["weights_ruleD"]]))


def regional(values, starts, ends, agg=np.mean):
    df = pd.DataFrame({"region": [region_of(s, e) for s, e in zip(starts, ends)], "v": values})
    return df.groupby("region", sort=False).v.agg(agg).reindex(list(REGIONS)).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--variant", default="floor")
    ap.add_argument("--ensemble", default="TrisMD")
    ap.add_argument("--reference", default="Tris_only")
    a = ap.parse_args()
    cells = sorted(p.name for p in a.out.iterdir() if p.name.startswith(a.ensemble + "__") and (p/a.variant/"pilot_selected.npz").exists())
    conds = [c.split("__")[1] for c in cells]
    C = {c.split("__")[1]: load_cell(a.out, c, a.variant) for c in cells}
    lines = [f"# Condition comparison: {a.ensemble}, variant `{a.variant}` (rule-D fold average vs no fit)", ""]
    rows = []
    # ---- per condition: no fit vs fit
    lines += ["## Per condition: residuals (predicted - observed fraction), region means over peptides x time points", "",
              "| condition | delta (source) | model | " + " | ".join(REGIONS) + " | held-out chi2/N | flags/cells | ESS % |", "|---|---|---|" + "---:|"*len(REGIONS) + "---:|---:|---:|"]
    for cond, c in C.items():
        info = c["info"]
        for model, pred in (("no fit", c["pred_prior"]), ("rule-D fit", c["pred_fit"])):
            resid = (pred - c["y"]).mean(1)
            reg = regional(resid, c["starts"], c["ends"])
            rule = "prior" if model == "no fit" else "D"
            f = c["ft"][c["ft"].rule == rule]
            ess = 100*float(np.mean(c["ess"])) if rule == "D" else 100.0
            lines.append(f"| {cond} | {info['delta']:.2f} ({info.get('delta_source', 'own')}) | {model} | " + " | ".join(f"{v:+.3f}" for v in reg.values)
                         + f" | {f.ho_chi2.mean():.1f} | {int(f.ho_n_flags.sum())}/{int(f.ho_n_cells.sum())} | {ess:.1f} |")
            for r, v in reg.items():
                rows.append(dict(kind="residual", condition=cond, model=model, region=r, value=v))
    # ---- ln PF by region: prior vs fit
    lines += ["", "## Ensemble ln PF (calibrated), region means over residues", "",
              "| condition | model | " + " | ".join(REGIONS) + " | " + " | ".join(PAPER_RES) + " |", "|---|---|" + "---:|"*(len(REGIONS)+len(PAPER_RES))]
    for cond, c in C.items():
        res = c["res"]
        for model, ln in (("no fit", c["lnpf_prior"]), ("rule-D fit", c["lnpf_fit"])):
            vals = [np.mean(ln[(res >= lo) & (res <= hi)]) for lo, hi in REGIONS.values()]
            pv = [np.mean(ln[np.isin(res, list(rr))]) for rr in PAPER_RES.values()]
            lines.append(f"| {cond} | {model} | " + " | ".join(f"{v:.2f}" for v in vals + pv) + " |")
            for r, v in zip(list(REGIONS) + list(PAPER_RES), vals + pv):
                rows.append(dict(kind="lnPF", condition=cond, model=model, region=r, value=float(v)))
    # ---- cross-condition differences vs reference
    ref = C[a.reference]
    lines += ["", f"## Condition differences vs {a.reference}: d_uptake = cond - {a.reference} (fraction), per peptide x time point", "",
              "| condition | model | corr(pred d, obs d) | RMS obs d | RMS (pred d - obs d) | " + " | ".join(f"obs/pred {r}" for r in REGIONS) + " |",
              "|---|---|---:|---:|---:|" + "---|"*len(REGIONS)]
    for cond, c in C.items():
        if cond == a.reference:
            continue
        if c["labels"] != ref["labels"]:
            raise ValueError(f"peptide sets differ between {cond} and {a.reference}")
        d_obs = c["y"] - ref["y"]
        for model, (pc, pr) in (("no fit", (c["pred_prior"], ref["pred_prior"])), ("rule-D fit", (c["pred_fit"], ref["pred_fit"]))):
            d_pred = pc - pr
            corr = np.corrcoef(d_pred.ravel(), d_obs.ravel())[0, 1]
            ro = regional(d_obs.mean(1), c["starts"], c["ends"]); rp = regional(d_pred.mean(1), c["starts"], c["ends"])
            lines.append(f"| {cond} | {model} | {corr:+.2f} | {np.sqrt(np.mean(d_obs**2)):.3f} | {np.sqrt(np.mean((d_pred-d_obs)**2)):.3f} | "
                         + " | ".join(f"{o:+.3f} / {p:+.3f}" for o, p in zip(ro.values, rp.values)) + " |")
            rows.append(dict(kind="d_uptake_corr", condition=cond, model=model, region="all", value=float(corr)))
            for r in REGIONS:
                rows.append(dict(kind="d_uptake_obs", condition=cond, model=model, region=r, value=float(ro[r])))
                rows.append(dict(kind="d_uptake_pred", condition=cond, model=model, region=r, value=float(rp[r])))
        # ln PF difference profile by paper region
        dln_fit = c["lnpf_fit"] - ref["lnpf_fit"]; dln_prior = c["lnpf_prior"] - ref["lnpf_prior"]
        res = c["res"]
        lines.append(f"| {cond} | d lnPF fit (prior) by paper region | " + "; ".join(
            f"{name}: {np.mean(dln_fit[np.isin(res, list(rr))]):+.2f} ({np.mean(dln_prior[np.isin(res, list(rr))]):+.2f})" for name, rr in PAPER_RES.items()) + " | | | " + " | "*len(REGIONS) + "|")
    lines += ["", "Observed order in the paper (after bradykinin normalisation): protection Tris > Extracellular > Intracellular everywhere; "
              "C-terminal 115-135 exposure sets nucleation, N-terminal 10-33 / 46-60, NAC and 101-113 exposure set elongation.", ""]
    (a.out/f"condition_comparison_{a.variant}.md").write_text("\n".join(lines))
    pd.DataFrame(rows).to_csv(a.out/f"condition_comparison_{a.variant}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
