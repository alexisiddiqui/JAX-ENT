"""lap/ shared code (read-only analysis; run from repo root with .venv/bin/python, OMP_NUM_THREADS=4).
Reuses selgci/q0_common.py (data, b' rate model, splits, solvers, held-out scores) and dg/ (indicators, MaxEnt dual, soft fit).
Adds: frame graphs (stored sidecar k=20 L2-logPF union graph for MSAss; rebuilt for Filtered), Laplacians, smoothed openness
o_tau = (I + tau L)^-1 o, p0 re-solve on smoothed O, LP feasibility check, and the chi2 + KL/S + gamma w'Lw solver."""
import os, sys
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(os.path.abspath(__file__))
RG = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(RG, 'selgci'))
import q0_common as Q          # noqa: E402
D, V = Q.D, Q.V
OUT = HERE + '/'
SIDE = 'jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_policy_sidecars_fresh_20261023/geometry/logpf_knn20.npz'
TAUS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
STATES5 = ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded', 'PUF2-like']
REGION_SEGS = {'b1': ['127-132'], 'a1': ['133-148', '149-153'], 'b2': ['154-167', '155-162'],
               'core': ['182-196', '197-204', '197-206', '205-212', '217-223']}


# ------------------------------------------------------------------ graphs ----
def stored_graph():
    z = np.load(SIDE); n = int(z['n_nodes'])
    A = np.zeros((n, n)); A[z['edge_sources'], z['edge_targets']] = 1.0; A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    return A


def logpf_frames(ens):
    """per-frame BV lnPF (0.35 Nc + 2.0 Nh; unfloored, unshifted: L2 distances are shift-invariant), (F, R)."""
    res, h, a, k = D.load(ens)
    return (D.BC0 * h + D.BH0 * a).T


def rebuilt_graph(ens, k=20, weighted=True):
    """k=20 L2-logPF neighbours (self excluded), symmetric union; Gaussian weights exp(-d^2/2h^2), h = median positive
    pairwise distance (median-scaled bandwidth 1)."""
    X = logpf_frames(ens); Dm = squareform(pdist(X)); n = len(Dm)
    M = Dm.copy(); np.fill_diagonal(M, 0.0)
    idx = np.argsort(M, axis=1, kind='stable')[:, :k + 1]
    A = np.zeros((n, n))
    for i in range(n):
        for j in idx[i]:
            if j != i: A[i, j] = 1.0
    A = np.maximum(A, A.T)
    h = float(np.median(Dm[np.triu_indices(n, 1)][Dm[np.triu_indices(n, 1)] > 0]))
    if weighted: A = A * np.exp(-Dm ** 2 / (2 * h ** 2))
    return A, h


def graph(ens, which='primary'):
    """primary: MSAss = stored sidecar graph (unweighted, as stored); Filtered = rebuilt Gaussian-weighted.
    'rebuilt': rebuilt Gaussian-weighted for either ensemble (MSAss check)."""
    if which == 'primary' and ens == 'AF2_MSAss':
        return stored_graph()
    return rebuilt_graph(ens)[0]


def laplacians(A):
    d = A.sum(1); L = np.diag(d) - A
    dm = 1 / np.sqrt(np.maximum(d, 1e-300)); Ls = np.eye(len(A)) - dm[:, None] * A * dm[None]
    return L, Ls


def smooth(O, L, tau):
    """rows of O (n_seg, F) smoothed: o_tau = (I + tau L)^-1 o."""
    if tau == 0: return O.copy()
    return np.linalg.solve(np.eye(L.shape[0]) + tau * L, O.T).T


# ------------------------------------------------------------- openness O ----
def raw_O(ens, fam='A'):
    """per-segment openness rows (S11 order) for family A (rule features) or B70 (native-contact Q_seg < 0.7); returns
    O (n, F), targets p_open, names (segments; A has 133-143 unfeatured and absent from S11 anyway)."""
    IND = dict(np.load(os.path.join(RG, 'dg', 'd2_indicators.npz')))
    T = D.seg_table().set_index('seg')
    O, t, names = [], [], []
    for s in D.S11:
        o = IND[f'{ens}|{s}|{fam}']
        if np.isnan(o).all(): continue
        O.append(o); t.append(T.loc[s, 'p_open']); names.append(s)
    return np.array(O), np.array(t), names


def merge_rows(O, t, names, Oraw=None):
    """q0/d3 merge: rows identical in the RAW (binary) O are merged with a geometric-mean target (smoothing is linear, so
    merged raw rows stay identical); rows whose target lies outside (min, max) of the row are dropped."""
    Oraw = O if Oraw is None else Oraw
    seen = {}
    for i, (o, ti, n) in enumerate(zip(Oraw, t, names)):
        key = np.asarray(o).tobytes()
        seen.setdefault(key, [i, [], []]); seen[key][1].append(ti); seen[key][2].append(n)
    Oe, te, ne, dropped = [], [], [], []
    for i, ts, ns in seen.values():
        g = float(np.exp(np.mean(np.log(ts)))); o = O[i]
        if not (o.min() < g < o.max()): dropped.append('+'.join(ns)); continue
        Oe.append(o); te.append(g); ne.append('+'.join(ns))
    return np.array(Oe), np.array(te), ne, dropped


def lp_feasible(O, t):
    """max eps s.t. O w = t, sum w = 1, w_i >= eps. Returns eps* (>0: interior-feasible, MaxEnt exists; ~0: only on the
    boundary; None: infeasible) and the min-L1 residual of the closest feasible-simplex point."""
    n_c, F = O.shape
    c = np.zeros(F + 1); c[-1] = -1
    Aeq = np.zeros((n_c + 1, F + 1)); Aeq[:n_c, :F] = O; Aeq[n_c, :F] = 1
    beq = np.r_[t, 1.0]
    Aub = np.zeros((F, F + 1)); Aub[:, :F] = -np.eye(F); Aub[:, -1] = 1
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(F), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * F + [(None, 1.0 / F)], method='highs')
    eps = float(r.x[-1]) if r.status == 0 else None
    # min sum |O w - t| over the simplex (relative residual per constraint)
    c2 = np.r_[np.zeros(F), np.ones(2 * n_c) / 1.0]
    A2 = np.zeros((n_c + 1, F + 2 * n_c)); A2[:n_c, :F] = O; A2[:n_c, F:F + n_c] = np.eye(n_c); A2[:n_c, F + n_c:] = -np.eye(n_c)
    A2[n_c, :F] = 1
    r2 = linprog(c2 / np.r_[np.ones(F), t, t], A_eq=A2, b_eq=np.r_[t, 1.0], bounds=[(0, None)] * (F + 2 * n_c), method='highs')
    return eps, float(r2.fun) if r2.status == 0 else np.nan


def solve_p0(O, t, names, Oraw=None):
    """exact MaxEnt (dg V.maxent_exact on merged rows, uniform base) in log space; also the LP feasibility of the merged
    system and the d3 log-least-squares soft fit (S=100) as the fallback. Returns dict."""
    Oe, te, ne, dropped = merge_rows(O, t, names, Oraw)
    F = O.shape[1]
    if len(Oe) == 0:          # every row dropped (target outside the smoothed row's range): p0 = uniform
        Oe = np.zeros((0, F)); te = np.zeros(0); lam = np.zeros(0); logp = np.full(F, -np.log(F)); eps = 1.0 / F; l1 = 0.0
    else:
        eps, l1 = lp_feasible(Oe, te)
        w, lam, info = V.maxent_exact(Oe, te)
        z = -np.log(F) + lam @ Oe; logp = z - logsumexp(z)
    ach = Oe @ np.exp(logp)
    ws, _ = V.soft_fit(O, t, 100)
    return dict(logp0=logp, lam=lam, Oe=Oe, te=te, names=ne, dropped=dropped, eps=eps, l1rel=l1,
                achieved=ach, max_rel_err=float(np.max(np.abs(ach / te - 1))) if len(te) else 0.0,
                lam_max=float(np.abs(lam).max()) if len(lam) else 0.0,
                w_soft=ws, feasible=(eps is not None and eps > 1e-12))


# ------------------------------------------------------------ OMC solver ----
def fit_omc(model, rows, logp0, S, L, gamma, eta0=None, gtol=1e-9):
    """min chi2_train(w) + KL(w||p0)/S + gamma w'Lw over w = softmax(eta) (500 logits, L-BFGS; the Lw term takes the optimum
    outside the q0.fit_ef exponential family, so the full-logit q0.fit form is used)."""
    if S == 0:
        return np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0)

    def f(eta):
        lw = eta - logsumexp(eta); w = np.exp(lw)
        c, g = model.chi2_grad(w, rows)
        Lw = L @ w
        val = c + np.sum(w * (lw - logp0)) / S + gamma * w @ Lw
        g = g + (lw - logp0 + 1) / S + 2 * gamma * Lw
        return val, w * (g - w @ g)
    x0 = logp0.copy() if eta0 is None else eta0
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=50000, maxfun=100000, ftol=1e-16, gtol=gtol, maxcor=30))
    lw = r.x - logsumexp(r.x); w = np.exp(lw)
    _, g = model.chi2_grad(w, rows)
    s = lw - logp0 + S * (g + 2 * gamma * (L @ w)); m = w > 1e-8
    stat = float(np.sum(w[m] * np.abs(s[m] - np.sum(w[m] * s[m]) / w[m].sum())) / w[m].sum())
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat, fun=float(r.fun), x=r.x)


# ------------------------------------------------------------ selection ----
ESS_MIN = 0.10


def select(g, units, keyfmt, rng):
    """verbatim selgci/q3_select.select (rules A, B, C, C1SE, D, refs), except that the candidate set is
    {ESS >= 10%} U {S = 0}: with a smoothed p0 the reference itself can have ESS < 10% (selgci assumed it never does)."""
    g = g.sort_values('S').reset_index(drop=True)
    c = g[(g.ESS >= ESS_MIN) | (g.S == 0)]
    out = {}
    mf = c.ho_n_flags.min(); fs = c[c.ho_n_flags == mf]
    out['A_flags'] = fs.S.idxmin()
    out['B_F'] = c.F.idxmin()
    i0 = c.ho_chi2_cell.idxmin(); out['C_minchi2'] = i0
    U = np.stack([units[keyfmt(S)] for S in c.S])
    nU = U.shape[1]; idx = rng.integers(0, nU, (1000, nU))
    boot = U[:, idx].mean(2)
    se = boot[list(c.index).index(i0)].std()
    ok = c[c.ho_chi2_cell <= c.ho_chi2_cell.min() + se]
    out['C_1SE'] = ok.S.idxmin()
    out['D_flags_then_F'] = fs.F.idxmin()
    # D restricted to ESS >= 50% candidates (the selgci acceptance check applied as a constraint; p0 if none)
    c50 = c[(c.ESS >= 0.5) | (c.S == 0)]; mf50 = c50.ho_n_flags.min(); fs50 = c50[c50.ho_n_flags == mf50]
    out['D_ess50'] = fs50.F.idxmin()
    out['ref_p0'] = g.S.idxmin()
    out['ref_oracle'] = c.rec.idxmax()
    rows = []
    for rule, i in out.items():
        r = g.loc[i].to_dict(); r.update(rule=rule, se_boot=se, min_flags=int(mf), n_minflag_pts=len(fs),
                                         F_rel=g.loc[i, 'F'] - g.loc[g.S.idxmin(), 'F'], S_index=int(i), at_grid_edge=bool(i == c.index.max()))
        rows.append(r)
    return rows
