"""Can ANY frame reweighting reach the targets? Under log_pf averaging ln PF_i = sum_f w_f (0.35 Nc_if + 2 Nh_if),
so the best weights are a non-negative least-squares problem over frames (with a free offset b0 and prior-scale s=1).
Solved on a random frame subsample for each ensemble; also the fraction of single frames positively correlated."""
import numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import jaxent.src.interfaces.topology as pt
B = Path(__file__).resolve().parents[2]; rng = np.random.default_rng(0)
CONDS = ["Tris only", "Extracellular", "Intracellular", "Lysosomal"]
PF = pd.read_csv(B / "data/_aSyn/aSyn_PF.csv", header=1); PF.columns = ["resid"] + list(PF.columns[1:]); PF = PF.set_index(PF.resid.astype(int))
for name, d, nsub in (("Tris-MD", B / "data/_aSyn/tris_MD/features", 8000), ("Shaw-MD", B / "data/_aSyn/a99sb_features", 2998)):
    f = np.load(d / "features.npz", allow_pickle=True)
    res = np.array([int(t.residues[0]) for t in pt.PTSerialiser.load_list_from_json(d / "topology.json")])
    L = 0.35 * np.asarray(f["heavy_contacts"]) + 2.0 * np.asarray(f["acceptor_contacts"])  # (res, frames) per-frame ln PF
    idx = np.sort(rng.choice(L.shape[1], size=min(nsub, L.shape[1]), replace=False)); Ls = L[:, idx]
    for c in CONDS:
        y = np.log(pd.to_numeric(PF[c], errors="coerce")); m = np.isfinite(y.reindex(res)).to_numpy()
        yy = y.reindex(res).to_numpy()[m]; A = Ls[m]
        # single-frame correlations with the target profile
        cf = np.array([np.corrcoef(A[:, j], yy)[0, 1] for j in range(A.shape[1])])
        # NNLS over frames with free offset (columns +1/-1), weights unnormalised (absorbs scale) -> upper bound on what reweighting+scale+offset can do
        X = np.c_[A, np.ones(len(yy)), -np.ones(len(yy))]; w, rn = nnls(X, yy, maxiter=5000)
        pred = X @ w; wf = w[:-2]; ess = (wf.sum()**2 / (wf**2).sum()) if wf.sum() > 0 else 0
        # NNLS with weights constrained to sum to 1 (approx: add a heavily weighted row sum(w)=1) and offset free
        lam = 50.0; Xc = np.r_[X, lam * np.r_[np.ones(A.shape[1]), 0, 0][None, :]]; yc = np.r_[yy, lam]; wc, _ = nnls(Xc, yc, maxiter=5000)
        predc = X @ wc; wfc = wc[:-2]
        print(f"[{name} {A.shape[1]} frames] {c:14s} var(y) {yy.var():.2f} | single frames: corr median {np.median(cf):+.2f}, frac corr>0 {np.mean(cf>0):.2f}, best {cf.max():+.2f} | "
              f"NNLS free-scale+offset: mse {np.mean((pred-yy)**2):.2f} corr {np.corrcoef(pred,yy)[0,1]:+.2f} sum(w) {wf.sum():.2f} ESS {ess:.0f} b0 {w[-2]-w[-1]:+.2f} | "
              f"NNLS sum(w)=1 +offset: mse {np.mean((predc-yy)**2):.2f} corr {np.corrcoef(predc,yy)[0,1]:+.2f} ESS {(wfc.sum()**2/(wfc**2).sum()):.0f} b0 {wc[-2]-wc[-1]:+.2f}")
