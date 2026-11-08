"""Tris-MD vs Shaw global density, time trend within replicates, and what default BV (0.35, 2.0) predicts on
later frames / Shaw-density-reweighted frames, against the corrected residue ln PF targets."""
import numpy as np, pandas as pd, sys, time
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import jaxent.src.interfaces.topology as pt
B = Path(__file__).resolve().parents[2] / "data"; rng = np.random.default_rng(0)
CONDS = ["Tris only", "Extracellular", "Intracellular", "Lysosomal"]
PF = pd.read_csv(B / "_aSyn/aSyn_PF.csv", header=1); PF.columns = ["resid"] + list(PF.columns[1:]); PF = PF.set_index(PF.resid.astype(int))
f = np.load(B / "_aSyn/tris_MD/features/features.npz", allow_pickle=True)
res = np.array([int(t.residues[0]) for t in pt.PTSerialiser.load_list_from_json(B / "_aSyn/tris_MD/features/topology.json")])
Nc, Nh = np.asarray(f["heavy_contacts"]), np.asarray(f["acceptor_contacts"]); L = 0.35 * Nc + 2.0 * Nh  # (res, frames)
nF = L.shape[1]; nrep = 9; per = nF // nrep; assert per * nrep == nF, (nF, per)
rep = np.repeat(np.arange(nrep), per); tfrac = np.tile(np.arange(per) / per, nrep)  # fraction of 120 ns within replicate
names = [f"{s}{r}" for s in ("rod", "coil", "hairpin") for r in (1, 2, 3)]
Y = {c: np.log(pd.to_numeric(PF[c], errors="coerce")).reindex(res).to_numpy() for c in CONDS}
m = np.isfinite(Y["Tris only"])
def score(w, c="Tris only"):
    w = w / w.sum(); pred = L[m] @ w; y = Y[c][m]
    return dict(level=round(float(pred.mean()), 2), mse=round(float(np.mean((pred - y) ** 2)), 2), corr=round(float(np.corrcoef(pred, y)[0, 1]), 2),
                mse_shift=round(float(np.var(pred - y)), 2), ess=round(float(1 / np.sum(w ** 2)), 0))
print(f"frames {nF} = {nrep} replicates x {per}; uniform ensemble vs Tris targets: {score(np.ones(nF))}")

print("\n=== per replicate (uniform within replicate) and per time-tenth within all replicates: prior level, <Nc>, MSE/corr vs Tris targets")
for i, n in enumerate(names):
    w = (rep == i).astype(float); print(f"  {n:9s} <Nc> {Nc[:, rep == i].mean():5.2f}  {score(w)}")
for q in range(10):
    sel = (tfrac >= q / 10) & (tfrac < (q + 1) / 10); w = sel.astype(float)
    print(f"  time {q*12:3d}-{(q+1)*12:3d} ns  <Nc> {Nc[:, sel].mean():5.2f}  <Nh> {Nh[:, sel].mean():.3f}  {score(w)}")

print("\n=== global density: Tris-MD vs Shaw (inertia shape axes + C-tail Rg from _cluster_inertia*)")
ST = np.c_[np.load(B / "_cluster_inertia/shape_axes.npy"), np.load(B / "_cluster_inertia/ctail_rg.npy")]
SS = np.c_[np.load(B / "_cluster_inertia_shaw/shape_axes.npy"), np.load(B / "_cluster_inertia_shaw/ctail_rg.npy")]
assert len(ST) == nF
for j, lab in enumerate(("shape axis 1", "shape axis 2", "C-tail Rg (A)")):
    print(f"  {lab:14s} Tris-MD 5/50/95%: {np.percentile(ST[:, j], [5, 50, 95]).round(2).tolist()}   Shaw: {np.percentile(SS[:, j], [5, 50, 95]).round(2).tolist()}")
# density-ratio weights w ~ p_shaw / p_tris on the 3 shape coordinates (KDE on standardised coords)
mu, sd = ST.mean(0), ST.std(0); kT = gaussian_kde(((ST - mu) / sd)[rng.choice(nF, 15000, replace=False)].T); kS = gaussian_kde(((SS - mu) / sd).T)
t0 = time.time(); Z = ((ST - mu) / sd).T; ratio = np.exp(kS.logpdf(Z) - kT.logpdf(Z)); print(f"  KDE density ratio computed in {time.time()-t0:.0f}s; ratio 5/50/95%: {np.percentile(ratio,[5,50,95]).round(2).tolist()}")
for clip in (np.inf, 20, 5):
    w = np.minimum(ratio, clip); print(f"  Shaw-density reweight (clip {clip}): <Nc> {np.average(Nc.mean(0), weights=w):5.2f}  {score(w)}")
for frac in (0.5, 0.2, 0.1, 0.05):
    k = int(frac * nF); sel = np.argsort(-ratio)[:k]; w = np.zeros(nF); w[sel] = 1
    print(f"  crop to top {int(frac*100):2d}% Shaw-like frames (n={k}): <Nc> {Nc[:, sel].mean():5.2f}  time-frac mean {tfrac[sel].mean():.2f}  {score(w)}")
low = np.argsort(Nc.mean(0))[: nF // 10]; w = np.zeros(nF); w[low] = 1
print(f"  crop to 10% least-contact frames: <Nc> {Nc[:, low].mean():5.2f}  {score(w)}")

print("\n=== reachability (NNLS over frames, default BV, no offset: sum-to-one via penalty row) on frame subsets, Tris targets")
def reach(cols, label, lam=50.0):
    A = L[m][:, cols]; y = Y["Tris only"][m]
    X = np.r_[A, lam * np.ones((1, A.shape[1]))]; yy = np.r_[y, lam]; t0 = time.time(); w, _ = nnls(X, yy, maxiter=20000)
    pred = A @ (w / w.sum()); print(f"  {label:38s} n={len(cols):6d}  mse {np.mean((pred-y)**2):.2f} corr {np.corrcoef(pred,y)[0,1]:+.2f} level {pred.mean():.2f} ESS {w.sum()**2/np.sum(w**2):.0f}  ({time.time()-t0:.0f}s)")
reach(rng.choice(nF, 8000, replace=False), "random 8000")
reach(np.where(tfrac >= 2 / 3)[0][::4], "later third of each replicate (1 in 4)")
reach(np.argsort(-ratio)[:8000], "top 8000 Shaw-like frames")
reach(np.arange(nF)[::2], "every 2nd frame (54k)")
