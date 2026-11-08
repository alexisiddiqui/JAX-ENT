"""Which time points constrain the reported rates? Parse Figure S11 (peptide + ETD fragment uptake time series)
per condition, take the whole-peptide series (index 1), compare uptake fractions at each time point with the
plateau, and refit a single exponential on candidate time grids against the reported k_exp."""
import csv, numpy as np, pandas as pd
from scipy.optimize import curve_fit
pd.set_option("display.width", 250)
import sys
rows = list(csv.reader(open(sys.argv[1] if len(sys.argv) > 1 else "_si_sheets/Figure_S11.csv")))
sections = {}; cur = None
for r in rows:
    if len(r) == 1 and r[0].strip():
        cur = r[0].strip(); sections[cur] = {"hdr": None, "rows": []}; continue
    if cur is None: continue
    if sections[cur]["hdr"] is None: sections[cur]["hdr"] = r; continue
    if r and r[0]: sections[cur]["rows"].append(r)
# intact-protein time grids from Figure S5 right panel (ms)
GRID = {"Tris only": [0, 50, 100, 250, 500, 5000, 60000, 300000], "Extracellular": [0, 50, 100, 250, 500, 5000, 60000, 300000],
        "Intracellular": [0, 50, 100, 250, 500, 5000, 60000, 300000], "Lysosomal": [0, 10000, 20000, 50000, 100000, 998000, 11972000, 59858000]}
def get(hdr, r, name):
    try: v = r[hdr.index(name)]
    except (ValueError, IndexError): return np.nan
    return float(v) if v not in ("", None) else np.nan
for cond, s in sections.items():
    hdr = s["hdr"]; out = []
    tcols = [h for h in hdr if h.startswith("Uptake_1_")]
    for r in s["rows"]:
        up = np.array([get(hdr, r, h) for h in tcols]); q = get(hdr, r, "q_1"); kexp = get(hdr, r, "k_exp_1")
        up = up[np.isfinite(up)]
        if len(up) < 3 or not np.isfinite(q): continue
        plateau = up.max(); frac = up / plateau
        out.append(dict(pep=f"{r[1]}-{r[2]}", n_t=len(up), q=q, plateau=round(plateau, 1), f_t1=round(frac[1], 2), f_t2=round(frac[2], 2), f_t3=round(frac[3], 2), f_t4=round(frac[4], 2) if len(frac) > 4 else np.nan,
                        k_exp=kexp, uptake=up))
    df = pd.DataFrame(out)
    print(f"\n=== {cond}: {len(df)} peptides; time points per peptide {df.n_t.value_counts().to_dict()}; grid assumed (ms) {GRID.get(cond)}")
    print(f"  fraction of plateau at time points 1..4: medians {df.f_t1.median():.2f} {df.f_t2.median():.2f} {df.f_t3.median():.2f} {df.f_t4.median():.2f}; "
          f"peptides >=90% exchanged at t1: {(df.f_t1>=0.9).sum()}/{len(df)}; >=50% at t1: {(df.f_t1>=0.5).sum()}/{len(df)}")
    # refit single exponential with the grid (ms) on the first len(up) points
    g = np.array(GRID.get(cond), float); ks = []
    for r in df.itertuples():
        t = g[:len(r.uptake)] if len(r.uptake) <= len(g) else np.r_[g, [g[-1]*2]*(len(r.uptake)-len(g))]
        try:
            (k, D), _ = curve_fit(lambda t, k, D: D*(1-np.exp(-k*t)), t, r.uptake, p0=[1/ max(t[1], 1), r.uptake.max()], maxfev=20000); ks.append(k)
        except Exception: ks.append(np.nan)
    df["k_refit_per_ms"] = ks; df["ratio_kexp/krefit"] = df.k_exp / df.k_refit_per_ms
    print(f"  reported k_exp median {df.k_exp.median():.3e} (range {df.k_exp.min():.2e}-{df.k_exp.max():.2e}); refit k (per ms on assumed grid) median {df.k_refit_per_ms.median():.3e}; ratio median {df['ratio_kexp/krefit'].median():.2f} IQR {df['ratio_kexp/krefit'].quantile(.25):.2f}-{df['ratio_kexp/krefit'].quantile(.75):.2f}")
    print(df.drop(columns=["uptake"]).head(6).to_string(index=False))
    if cond == "Lysosomal":
        print("  Lysosomal first peptide uptake series:", np.round(df.uptake.iloc[0], 2).tolist())
