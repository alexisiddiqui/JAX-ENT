"""selgci/ shared code (read-only analysis; run from repo root with .venv/bin/python).
MoPrP: scaled BV (b') = additive lnPF shift delta with lnPF floored at 0 (pop/p1c; delta -3.108 MSAss / -3.174 Filtered),
rate averaging, _featurise_a101_k109 features, all 14 peptides x 15 timepoints.
p0: MaxEnt reference w ∝ exp(lam . O) on rule-feature (family A) openness, segment set S11, targets p_open = K/(1+K),
identical rows merged -- recomputed exactly as dg/d3_t2_reweight.py 'A|S11|exact' but kept in log space (so frames the
infeasible MSAss system drives to 0 keep a finite, very negative log p0).
ISO: self-consistent ISO_BI / ISO_TRI features, residue-level frame_uptake target + Gaussian noise.
Solver: min_w chi2_train(w) + KL(w||p0)/S, w = softmax(eta), L-BFGS on the logits (numpy/scipy, no framework)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RG = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(RG, 'dg'))
import d0_common as D          # noqa: E402  (also puts gci2/c0_common on path as D.C)
import d0b_solvers as V        # noqa: E402
C = D.C
OUT = HERE + '/'
SIGMA_FLOOR = 0.0586
ALPHA = 0.05
POP_MODELS = json.load(open(os.path.join(RG, 'pop', 'p1_models.json')))
MO_STATES = ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded', 'PUF2-like']


def t_n(n, alpha=ALPHA):
    """Sidak per-cell |z| threshold for n cells at family-wise alpha."""
    return float(norm.ppf(0.5 + 0.5 * (1 - alpha) ** (1.0 / n)))


def kl(w, logw, logp0):
    m = w > 0
    return float(np.sum(w[m] * (logw[m] - logp0[m])))


def ess(w):
    return float(1.0 / np.sum(w ** 2) / len(w))


# ------------------------------------------------------------------ MoPrP ----
def moprp_p0(ens):
    """recompute dg T2 'A|S11|exact' reference in log space; returns logp0 (500,), info."""
    T = D.seg_table().set_index('seg'); IND = dict(np.load(os.path.join(RG, 'dg', 'd2_indicators.npz')))
    O, t, names = [], [], []
    for s in D.S11:
        o = IND[f'{ens}|{s}|A']
        if np.isnan(o).all(): continue
        O.append(o); t.append(T.loc[s, 'p_open']); names.append(s)
    seen = {}
    for o, ti, n in zip(O, t, names):
        key = np.asarray(o).tobytes()
        seen.setdefault(key, [o, [], []]); seen[key][1].append(ti); seen[key][2].append(n)
    Oe, te, ne, dropped = [], [], [], []
    for o, ts, ns in seen.values():
        g = float(np.exp(np.mean(np.log(ts))))
        if not (o.min() < g < o.max()): dropped.append('+'.join(ns)); continue
        Oe.append(o); te.append(g); ne.append('+'.join(ns))
    Oe, te = np.array(Oe), np.array(te)
    w, lam, info = V.maxent_exact(Oe, te)
    z = -np.log(Oe.shape[1]) + lam @ Oe
    logp0 = z - logsumexp(z)
    ref = np.load(os.path.join(RG, 'dg', 'd3_weights.npz'))[f'{ens}|A|S11|exact|inf']
    info.update(constraints=ne, dropped=dropped, lam=lam.tolist(), max_dev_vs_d3=float(np.abs(np.exp(logp0) - ref).max()),
                achieved=(Oe @ np.exp(logp0)).tolist(), targets=te.tolist())
    return logp0, info


class MoPrP:
    """b' rate-averaged forward model; sigma = per-peptide pooled RMS of 1/weights (optionally floored)."""

    def __init__(self, ens, floor=True):
        res, h, a, k = D.load(ens)
        self.ens = ens; self.M = D.pmap(res)
        delta = POP_MODELS[f'{ens}|shift_clamp0|matched']['delta']; self.delta = delta
        L = np.maximum(D.BC0 * h + D.BH0 * a + delta, 0.0)
        self.Kr = k[:, None] * np.exp(-L)
        self.T = D.TIMES; self.y = D.DEXP.copy()
        sp = np.sqrt(np.mean((1.0 / D.WTS) ** 2, 1))
        self.sig_pep = np.maximum(sp, SIGMA_FLOOR) if floor else sp
        self.sig = self.sig_pep[:, None] * np.ones((1, len(self.T)))
        self.labels = D.assign(ens); self.nF = self.Kr.shape[1]
        self.unit_lab = D.LAB; self.NT = len(self.T)

    def pred(self, w):
        kbar = self.Kr @ w
        return self.M @ (1 - np.exp(-kbar[:, None] * self.T[None]))

    def chi2_grad(self, w, rows):
        """mean over cells of the given peptide rows of z^2, and gradient wrt w."""
        kbar = self.Kr @ w
        U = 1 - np.exp(-kbar[:, None] * self.T[None])
        P = self.M[rows] @ U; r = self.y[rows] - P; iv = 1 / self.sig[rows] ** 2; N = r.size
        val = np.sum(iv * r ** 2) / N
        dP = -2 * iv * r / N
        E = self.T[None] * np.exp(-kbar[:, None] * self.T[None])
        g = ((self.M[rows].T @ dP) * E).sum(1) @ self.Kr
        return val, g

    def pops(self, w):
        return np.array([w[self.labels == i].sum() for i in range(6)])

    def recovery(self, w):
        return D.recovery(self.labels, w)


# -------------------------------------------------------------------- ISO ----
ISO_B = 'jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_self_consistent_iso/'
ISO_GT = {"intermediate": 0.0, "open": 0.4, "closed": 0.6}
ISO_SM = {-1: "intermediate", 0: "open", 1: "closed"}


class ISO:
    """self-consistent ISO features, BV(0.35,2.0) frame_uptake, residue-level target (+ noise sigma), identity map."""

    def __init__(self, which='bi', sigma=0.01, seed=0):
        z = np.load(ISO_B + f'fit_features/features_iso_{which}.npz')
        res = np.array([t['residues'][0] for t in json.load(open(ISO_B + f'fit_features/topology_iso_{which}.json'))['topologies']])
        segs = np.loadtxt(ISO_B + 'target_frame_uptake/residue/target_segs.txt', dtype=int)
        tgt = np.loadtxt(ISO_B + 'target_frame_uptake/residue/target_dfrac.dat')
        tms = np.array(open(ISO_B + 'target_frame_uptake/residue/target_dfrac.dat').readline().split()[1:6], float)
        M = C.pep_map_norm(res, segs, 1)
        assert np.allclose(M, np.eye(len(res)))
        L = 0.35 * np.asarray(z['heavy_contacts'], float) + 2.0 * np.asarray(z['acceptor_contacts'], float)
        self.F = C.frame_pep_uptake(L, np.asarray(z['k_ints'], float), M, tms)      # (F, R, T)
        self.Ff = self.F.reshape(self.F.shape[0], -1)
        rng = np.random.default_rng(1000 + seed + int(round(sigma * 1e4)))
        self.y0 = tgt; self.y = tgt + sigma * rng.standard_normal(tgt.shape)
        self.sig_pep = np.full(len(res), sigma); self.sig = np.full(tgt.shape, sigma)
        lab = pd.read_csv('jaxent/examples/1_IsoValidation_OMass/data/_clustering_results/'
                          f'cluster_assignments_ISO_{which.upper()}.csv')
        self.labels = lab[lab.ensemble == f'ISO_{which.upper()}'].cluster_assignment.values
        assert len(self.labels) == self.F.shape[0]
        self.nF = self.F.shape[0]; self.res = res; self.NT = len(tms)
        self.unit_lab = [str(r) for r in res]

    def pred(self, w):
        return np.einsum('frt,f->rt', self.F, w)

    def chi2_grad(self, w, rows):
        Fr = self.F[:, rows]                                    # (F, n, T)
        P = np.einsum('frt,f->rt', Fr, w); r = self.y[rows] - P; iv = 1 / self.sig[rows] ** 2; N = r.size
        val = np.sum(iv * r ** 2) / N
        dP = -2 * iv * r / N
        return val, np.einsum('frt,rt->f', Fr, dP)

    def pops(self, w):
        return np.array([w[self.labels == i].sum() for i in (0, 1, -1)])   # open, closed, unassigned

    def recovery(self, w):
        from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
        return float(calculate_recovery_percentage(self.labels, w / w.sum(), ISO_GT, ISO_SM))


# ------------------------------------------------------------------ solver ----
def fit(model, rows, logp0, S, eta0=None, gtol=1e-9):
    """min chi2_train(w) + KL(w||p0)/S over w = softmax(eta). Returns w, logw, info."""
    if S == 0:
        return np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0)

    def f(eta):
        lw = eta - logsumexp(eta); w = np.exp(lw)
        c, g = model.chi2_grad(w, rows)
        val = c + np.sum(w * (lw - logp0)) / S
        g = g + (lw - logp0 + 1) / S
        return val, w * (g - w @ g)
    x0 = logp0.copy() if eta0 is None else eta0
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=50000, maxfun=100000, ftol=1e-16, gtol=gtol, maxcor=30))
    lw = r.x - logsumexp(r.x); w = np.exp(lw)
    # stationarity: lw - logp0 + S*dchi2/dw must be constant over the support
    _, g = model.chi2_grad(w, rows)
    s = lw - logp0 + S * g; m = w > 1e-8
    stat = float(np.sum(w[m] * np.abs(s[m] - np.sum(w[m] * s[m]) / w[m].sum())) / w[m].sum())
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat, fun=float(r.fun))


# ------------------------------------------------------------------ splits ----
def moprp_splits():
    """three 2-fold contiguous-block splits of the 14 peptides (index sets)."""
    lab = D.LAB
    blocks = {'cut45|60': ([i for i, l in enumerate(lab) if int(l.split('-')[1]) <= 45], 'N 5-45', 'C 60-101'),
              'cut31|32': ([i for i, l in enumerate(lab) if int(l.split('-')[1]) <= 31], 'N 5-31', 'C 32-101'),
              'cut75|75': ([i for i, l in enumerate(lab) if int(l.split('-')[1]) <= 75], 'N 5-75', 'C 75-101')}
    out = []
    for name, (a, la, lb) in blocks.items():
        b = [i for i in range(len(lab)) if i not in a]
        out.append((name, la, b, a))        # fold label = held-out block; train on the other block
        out.append((name, lb, a, b))
    return out   # (split, heldout_label, train_rows, val_rows)


def iso_splits(nres):
    out = []
    for frac in (0.5, 1 / 3, 2 / 3):
        c = int(round(frac * nres)); a = list(range(c)); b = list(range(c, nres))
        name = f'cut{c}'
        out.append((name, f'N 0-{c-1}', b, a)); out.append((name, f'C {c}-{nres-1}', a, b))
    return out


# ------------------------------------------------------------------ scores ----
def heldout(model, w, rows, nboot=0, rng=None):
    P = model.pred(w)
    z = (model.y[rows] - P[rows]) / model.sig[rows]          # (nv, T)
    n = z.size; tn = t_n(n); fl = np.abs(z) > tn
    out = dict(chi2_cell=float(np.mean(z ** 2)), half_sum_z2=float(0.5 * np.sum(z ** 2)), n_cells=n, t_n=tn,
               n_flags=int(fl.sum()), flagged_units=';'.join(f'{model.unit_lab[rows[i]]}({int(fl[i].sum())})' for i in range(len(rows)) if fl[i].any()),
               n_units_flagged=int(fl.any(1).sum()), maxabsz=float(np.abs(z).max()), mean_res=float((z * model.sig[rows]).mean()))
    return out, np.mean(z ** 2, 1)        # per-unit mean z^2 (for bootstrap)


def fit_ef(model, rows, logp0, S, Phi, mu0=None, gtol=1e-10):
    """Same objective as fit(), but in the exponential-family parametrisation that contains its stationary point:
    w(mu) = softmax(logp0 + mu . Phi), Phi (d, F) = the per-frame basis of dchi2/dw (rate mode: per-residue rates of the
    training residues, standardised). L-BFGS on mu (d << F). Returns w, logw, info (stat = stationarity residual)."""
    if S == 0:
        return np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0), np.zeros(Phi.shape[0])

    def f(mu):
        eta = logp0 + mu @ Phi; lw = eta - logsumexp(eta); w = np.exp(lw)
        c, g = model.chi2_grad(w, rows)
        val = c + np.sum(w * (lw - logp0)) / S
        G = g + (lw - logp0 + 1) / S
        return val, Phi @ (w * (G - w @ G))
    x0 = np.zeros(Phi.shape[0]) if mu0 is None else mu0
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=50000, maxfun=100000, ftol=1e-16, gtol=gtol, maxcor=50))
    eta = logp0 + r.x @ Phi; lw = eta - logsumexp(eta); w = np.exp(lw)
    _, g = model.chi2_grad(w, rows)
    s = lw - logp0 + S * g; m = w > 1e-8
    stat = float(np.sum(w[m] * np.abs(s[m] - np.sum(w[m] * s[m]) / w[m].sum())) / w[m].sum())
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat, fun=float(r.fun)), r.x


def moprp_basis(model, rows, logp0):
    """standardised per-residue rates of residues covered by the training peptides."""
    rr = np.where(model.M[rows].sum(0) > 0)[0]
    X = model.Kr[rr]; p = np.exp(logp0)
    mu = X @ p; sd = np.sqrt(np.maximum((X ** 2) @ p - mu ** 2, 0)); sd[sd == 0] = 1
    return (X - mu[:, None]) / sd[:, None]


def fit_dual_linear(model, rows, logp0, S, lam0=None, gtol=1e-10):
    """Exact convex dual for a LINEAR forward model (ISO frame_uptake): primal min_w (1/N) sum iv (y - Fw)^2 + KL(w||p0)/S.
    Dual (concave in lam, one lam per training cell): D(lam) = -(1/S) log sum_f p0_f exp(-S (F^T lam)_f) - lam.y - (N/4) sum lam^2/iv,
    w = p0 exp(-S F^T lam)/Z. Solved with L-BFGS on -D. Same objective/optimum as fit(); stationarity residual reported."""
    if S == 0:
        return np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0), None
    Fr = model.F[:, rows].reshape(model.nF, -1); y = model.y[rows].ravel(); iv = 1 / model.sig[rows].ravel() ** 2; N = y.size

    def negD(lam):
        eta = logp0 - S * (Fr @ lam); lz = logsumexp(eta); w = np.exp(eta - lz)
        D = -lz / S - lam @ y - 0.25 * N * np.sum(lam ** 2 / iv)
        g = Fr.T @ w - y - 0.5 * N * lam / iv
        return -D, -g
    r = minimize(negD, np.zeros(N) if lam0 is None else lam0, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=50000, maxfun=100000, ftol=1e-16, gtol=gtol, maxcor=30))
    eta = logp0 - S * (Fr @ r.x); lw = eta - logsumexp(eta); w = np.exp(lw)
    _, g = model.chi2_grad(w, rows)
    s = lw - logp0 + S * g; m = w > 1e-8
    stat = float(np.sum(w[m] * np.abs(s[m] - np.sum(w[m] * s[m]) / w[m].sum())) / w[m].sum())
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat, fun=float(r.fun)), r.x
