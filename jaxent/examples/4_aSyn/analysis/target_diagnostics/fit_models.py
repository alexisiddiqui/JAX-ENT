"""Offset/scale/OLS/median models of the corrected ln PF targets on Tris-MD and Shaw (a99SB-disp) features,
noise weighting from the reported k_obs SD, region summaries and leave-one-segment-out influence."""
import numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import jaxent.src.interfaces.topology as pt
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
B = Path(__file__).resolve().parents[2]
CONDS = ["Tris only", "Extracellular", "Intracellular", "Lysosomal"]

def load_feats(d):
    f = np.load(d / "features.npz", allow_pickle=True)
    res = np.array([int(t.residues[0]) for t in pt.PTSerialiser.load_list_from_json(d / "topology.json")])
    return dict(res=res, Nc=np.asarray(f["heavy_contacts"]).mean(1), Nh=np.asarray(f["acceptor_contacts"]).mean(1),
                nframes=f["heavy_contacts"].shape[1])
ENS = {"Tris-MD": load_feats(B / "data/_aSyn/tris_MD/features"), "Shaw-MD": load_feats(B / "data/_aSyn/a99sb_features")}
for n, e in ENS.items():
    print(f"{n}: {len(e['res'])} residues ({e['res'].min()}-{e['res'].max()}), {e['nframes']} frames, "
          f"<Nc> median {np.median(e['Nc']):.2f}, <Nh> median {np.median(e['Nh']):.3f}, prior lnPF median {np.median(0.35*e['Nc']+2*e['Nh']):.2f}")

def tbl(p):
    t = pd.read_csv(p, header=1); t.columns = ["resid"] + list(t.columns[1:]); t["resid"] = t.resid.astype(int)
    for c in CONDS: t[c] = pd.to_numeric(t[c], errors="coerce")
    return t.set_index("resid")
K, SD = tbl(B / "data/_aSyn/aSyn_k_obs.csv"), tbl(B / "data/_aSyn/aSyn_k_obs_SD.csv")
PF, PF_OLD = tbl(B / "data/_aSyn/aSyn_PF.csv"), tbl(B / "data/_aSyn/_pre_ms_units_20261105/aSyn_PF.csv")

# segments = runs of identical k_obs (Tris column) -> the fragment-level resolution of the data
seg = pd.Series(index=K.index, dtype=float); sid = 0; prev = None
for r in K.index:
    v = K.loc[r, "Tris only"]
    if not v > 0: prev = None; continue
    if prev is None or not np.isclose(v, prev): sid += 1
    seg[r] = sid; prev = v
segs = seg.dropna().astype(int)
lens = segs.groupby(segs).apply(lambda s: f"{s.index.min()}-{s.index.max()}")
print(f"\n{segs.nunique()} segments of identical k_obs (Tris), lengths: {np.bincount(segs.value_counts().values)[1:].tolist()} (count of segments with 1,2,3,... residues)")
print("segments:", ", ".join(lens.tolist()))

REGIONS = {"N 1-60": (1, 60), "NAC 61-95": (61, 95), "C 96-119": (96, 119), "tail 120-140": (120, 140)}
print("\n=== region means: ln PF target (current units), ln k_obs (s^-1), ln k_int (s^-1); ensemble <Nc>, <Nh>, prior ln PF")
rows = []
for rn, (a, b) in REGIONS.items():
    rr = [r for r in K.index if a <= r <= b and K.loc[r, "Tris only"] > 0]
    d = dict(region=rn, n=len(rr))
    for c in CONDS:
        y = np.log(PF.loc[rr, c]); ko = np.log(K.loc[rr, c] * 1000); d[f"lnPF {c[:5]}"] = round(y.mean(), 2); d[f"lnkobs {c[:5]}"] = round(ko.mean(), 2); d[f"lnkint {c[:5]}"] = round((y + ko).mean(), 2)
    for n, e in ENS.items():
        m = np.isin(e["res"], rr); d[f"Nc {n}"] = round(e["Nc"][m].mean(), 2); d[f"Nh {n}"] = round(e["Nh"][m].mean(), 3); d[f"prior {n}"] = round((0.35*e["Nc"]+2*e["Nh"])[m].mean(), 2)
    rows.append(d)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== where the median sits: median ln PF target per condition vs prior median")
for c in CONDS:
    y = np.log(PF[c].dropna()); yo = np.log(PF_OLD[c].dropna())
    print(f"{c:14s} current median {np.median(y):+.2f} (IQR {np.percentile(y,25):+.2f}..{np.percentile(y,75):+.2f}, min {y.min():+.2f} max {y.max():+.2f}) | pre-units median {np.median(yo):+.2f} | "
          + " | ".join(f"{n} prior median {np.median(0.35*e['Nc']+2*e['Nh']):.2f} -> shift {np.median(y)-np.median(0.35*e['Nc']+2*e['Nh']):+.2f}" for n, e in ENS.items()))

def fits(Nc, Nh, y, w=None):
    """Return dict of model -> (params, mse). w = per-residue weights (1/sigma^2) for the WLS row only."""
    p = 0.35*Nc + 2*Nh; one = np.ones_like(y); out = {}
    mse = lambda pred: float(np.mean((pred - y)**2))
    out["prior(0.35,2.0)"] = ((), mse(p))
    b0 = float(np.median(y - p)); out["median shift b0"] = ((b0,), mse(p + b0))
    b0 = float(np.mean(y - p)); out["mean shift b0"] = ((b0,), mse(p + b0))
    s = float(p @ y / (p @ p)); out["scale s"] = ((s,), mse(s*p))
    X = np.c_[one, p]; b = np.linalg.lstsq(X, y, rcond=None)[0]; out["b0 + s*prior"] = (tuple(b), mse(X @ b))
    X = np.c_[one, Nc, Nh]; b = np.linalg.lstsq(X, y, rcond=None)[0]; out["OLS b0,bc,bh"] = (tuple(b), mse(X @ b))
    Xn = np.c_[Nc, Nh, one, -one]; bn, _ = nnls(Xn, y); out["NNLS bc,bh>=0 + b0"] = ((bn[2]-bn[3], bn[0], bn[1]), mse(Xn @ bn))
    if w is not None:
        sw = np.sqrt(w); b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]; out["WLS b0,bc,bh (1/sd^2)"] = (tuple(b), mse(X @ b))
    return out

print("\n=== models per ensemble x condition (uniform weights, all residues with data). params rounded; mse in ln^2")
for n, e in ENS.items():
    ic = {r: i for i, r in enumerate(e["res"])}
    for c in CONDS:
        rr = [r for r in PF.index if np.isfinite(PF.loc[r, c]) and PF.loc[r, c] > 0 and r in ic]
        y = np.log(PF.loc[rr, c].to_numpy()); Nc = np.array([e["Nc"][ic[r]] for r in rr]); Nh = np.array([e["Nh"][ic[r]] for r in rr])
        sig = (SD.loc[rr, c] / K.loc[rr, c]).to_numpy(); sig = np.where(sig > 0, sig, np.nan); sig = np.where(np.isfinite(sig), sig, np.nanmedian(sig))
        f = fits(Nc, Nh, y, w=1/sig**2); p = 0.35*Nc + 2*Nh
        print(f"\n[{n}] {c} n={len(rr)}  corr(y,Nc) {np.corrcoef(y,Nc)[0,1]:+.2f}  corr(y,Nh) {np.corrcoef(y,Nh)[0,1]:+.2f}  corr(y,prior) {np.corrcoef(y,p)[0,1]:+.2f}  var(y) {y.var():.2f}")
        for k, (par, m) in f.items():
            print(f"   {k:24s} mse {m:6.2f}   params {tuple(round(float(x), 3) for x in par)}")

print("\n=== influence (Tris-MD features): OLS b0,bc,bh dropping one segment at a time; segments with largest |delta bh| and |delta bc|")
e = ENS["Tris-MD"]; ic = {r: i for i, r in enumerate(e["res"])}
for c in CONDS:
    rr = np.array([r for r in segs.index if np.isfinite(PF.loc[r, c]) and PF.loc[r, c] > 0 and r in ic])
    y = np.log(PF.loc[rr, c].to_numpy()); Nc = np.array([e["Nc"][ic[r]] for r in rr]); Nh = np.array([e["Nh"][ic[r]] for r in rr]); s_ = segs.loc[rr].to_numpy()
    X = np.c_[np.ones_like(y), Nc, Nh]; full = np.linalg.lstsq(X, y, rcond=None)[0]
    inf = []
    for sg in np.unique(s_):
        m = s_ != sg; b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]; inf.append((sg, lens[sg], b[1]-full[1], b[2]-full[2], y[~m].mean()))
    inf = pd.DataFrame(inf, columns=["seg", "residues", "d_bc", "d_bh", "mean_y"])
    top = inf.reindex(inf.d_bh.abs().sort_values(ascending=False).index).head(4)
    print(f"{c:14s} full OLS b0 {full[0]:+.2f} bc {full[1]:+.3f} bh {full[2]:+.3f} | most influential on bh: " + "; ".join(f"{r.residues} (dbh {r.d_bh:+.2f}, dbc {r.d_bc:+.3f}, y={r.mean_y:+.1f})" for r in top.itertuples()))
    for lab, m in (("residues <120", rr < 120), ("residues >=120", rr >= 120)):
        b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]; print(f"      diagnostic OLS on {lab:14s} n={m.sum():3d}: b0 {b[0]:+.2f} bc {b[1]:+.3f} bh {b[2]:+.3f}  corr(y,Nc) {np.corrcoef(y[m],Nc[m])[0,1]:+.2f}")

print("\n=== noise: per-residue sigma(ln PF) = SD/k_obs; residues with sigma > 0.3")
for c in CONDS:
    rr = [r for r in K.index if K.loc[r, c] > 0]; sig = (SD.loc[rr, c] / K.loc[rr, c])
    big = sig[sig > 0.3]
    print(f"{c:14s} sigma median {sig.median():.3f}, 90th pct {sig.quantile(.9):.3f}; n(sigma=0) {(sig==0).sum()}; sigma>0.3 at residues {big.round(2).to_dict()}")
