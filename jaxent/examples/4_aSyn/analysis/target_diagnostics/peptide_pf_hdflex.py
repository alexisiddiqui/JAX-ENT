"""Recompute Tris protection factors the way HDfleX defines them: fit the observed whole-peptide uptake AND the
theoretical intrinsic peptide curve (sum over residues of 1-exp(-k_int t)) with the same stretched exponential
N(1-exp(-(k t)^beta)), PF_pep = k_int,fit / k_obs,fit. Compare with our residue-level ln(k_int/k_obs) averaged
over the peptide, and with the BV prior of Tris-MD and Shaw on the same residues."""
import csv, json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import jaxent.src.interfaces.topology as pt
pd.set_option("display.width", 250)
B = Path(__file__).resolve().parents[2] / "data"
CONDS = {"Tris only": "Tris_only", "Extracellular": "Extracellular", "Intracellular": "Intracellular"}
grids = json.load(open(B / "_aSyn/uptake/S11_time_grids.json")); norm = grids["norm_factor"]
def tbl(p):
    t = pd.read_csv(p, header=1); t.columns = ["resid"] + list(t.columns[1:]); t["resid"] = t.resid.astype(int)
    return t.set_index("resid").apply(pd.to_numeric, errors="coerce")
K, PF = tbl(B / "_aSyn/aSyn_k_obs.csv"), tbl(B / "_aSyn/aSyn_PF.csv")
seq = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
def feats(d):
    f = np.load(d / "features.npz", allow_pickle=True); res = np.array([int(t.residues[0]) for t in pt.PTSerialiser.load_list_from_json(d / "topology.json")])
    return dict(zip(res, 0.35 * np.asarray(f["heavy_contacts"]).mean(1) + 2.0 * np.asarray(f["acceptor_contacts"]).mean(1)))
PRI = {"Tris-MD": feats(B / "_aSyn/tris_MD/features"), "Shaw-MD": feats(B / "_aSyn/a99sb_features")}
rows = list(csv.reader(open(sys.argv[1] if len(sys.argv) > 1 else "_si_sheets/Figure_S11.csv")))
sections, cur = {}, None
for r in rows:
    if len(r) == 1 and r[0].strip(): cur = r[0].strip(); sections[cur] = {"hdr": None, "rows": []}; continue
    if cur is None: continue
    if sections[cur]["hdr"] is None: sections[cur]["hdr"] = r; continue
    if r and r[0]: sections[cur]["rows"].append(r)
se = lambda t, k, b, N: N * (1 - np.exp(-(k * t) ** b))
def fit_se(t, u, N):
    (k, b), _ = curve_fit(lambda t, k, b: se(t, k, b, N), t, u, p0=[1.0, 0.7], bounds=([1e-6, 0.05], [1e4, 1.0]), maxfev=20000); return k, b
for cond, key in CONDS.items():
    hdr = sections[cond]["hdr"]; grid = np.array([g for g in grids["time_grid_s"][key] if g is not None]); out = []
    for r in sections[cond]["rows"]:
        g = lambda n: float(r[hdr.index(n)]) if r[hdr.index(n)] not in ("",) else np.nan
        up = np.array([g(f"Uptake_1_{i}") for i in range(1, 10)]); up = up[np.isfinite(up)][: len(grid)]
        q, kexp, start, end = g("q_1"), g("k_exp_1"), int(r[1]), int(r[2])
        if not np.isfinite(q) or len(up) < len(grid): continue
        resid = [i for i in range(start + 1, end + 1) if seq[i - 1] != "P"]
        kint = np.array([PF.loc[i, cond] * K.loc[i, cond] * 1000 for i in resid if i in PF.index and np.isfinite(PF.loc[i, cond]) and K.loc[i, cond] > 0])
        if len(kint) < max(2, 0.6 * len(resid)): continue
        intr = np.array([np.sum(1 - np.exp(-kint * t)) for t in grid]) * (q / len(kint))  # theoretical intrinsic curve, scaled to q
        k_int_fit, b_int = fit_se(grid, intr, q); k_obs_fit, b_obs = fit_se(grid, up, q)
        k_obs_rep = kexp * 1000 / norm[key]  # reported, back to raw s^-1
        lnpf_res = float(np.mean(np.log(PF.loc[[i for i in resid if i in PF.index], cond].dropna())))
        pri = {n: float(np.mean([p[i] for i in resid if i in p])) for n, p in PRI.items()}
        out.append(dict(pep=f"{start}-{end}", q=int(q), beta_obs=round(b_obs, 2), k_obs_fit=round(k_obs_fit, 3), k_obs_rep=round(k_obs_rep, 3), ratio=round(k_obs_fit / k_obs_rep, 2),
                        beta_int=round(b_int, 2), k_int_fit=round(k_int_fit, 2), mean_kint=round(kint.mean(), 2), lnPF_hdflex=round(np.log(k_int_fit / k_obs_rep), 2),
                        lnPF_residue_mean=round(lnpf_res, 2), ln_meankint_over_kobs=round(np.log(kint.mean() / k_obs_rep), 2), **{f"prior {n}": round(v, 2) for n, v in pri.items()}))
    df = pd.DataFrame(out)
    print(f"\n=== {cond}: {len(df)} peptides, grid (s) {grid.tolist()}; stretched-exp refit vs reported k (after undoing the bradykinin factor {norm[key]:.3f}): ratio median {df.ratio.median():.3f} IQR {df.ratio.quantile(.25):.3f}-{df.ratio.quantile(.75):.3f}")
    print(f"  beta_obs median {df.beta_obs.median():.2f}; beta_int median {df.beta_int.median():.2f}")
    print(f"  ln PF per peptide, medians: HDfleX-style (k_int,fit/k_obs) {df.lnPF_hdflex.median():+.2f} | residue-mean ln(k_int_i/k_obs) {df.lnPF_residue_mean.median():+.2f} | ln(mean k_int / k_obs) {df.ln_meankint_over_kobs.median():+.2f}"
          f" | BV prior Tris-MD {df['prior Tris-MD'].median():.2f}, Shaw-MD {df['prior Shaw-MD'].median():.2f}")
    print(df.to_string(index=False))
