"""stk/ shared code (read-only analysis; run from repo root with .venv/bin/python, OMP_NUM_THREADS=4).
Reuses selgci/q0_common.py (data, b' rate model, p0, EF solver, ISO dual solver, splits) and lap/l0_common.fit_omc.
Adds: ONE self-tuning kernel per ensemble in L2-lnPF space, used in two roles
  (a) lazy-walk covariance of the mean prediction C_walk = (1/N^2) F~^T G F~ (gci2/h2 symmetric eigen-form of 2(I-P)^-1 - I,
      stationary mode excluded), lambda_geo = tr(C_walk)/tr(C_ens)  [= inflation/N, gci2 convention], N_eff = N/inflation;
  (b) symmetric normalised Laplacian L_st = I - D^-1/2 K~ D^-1/2 for the OMC term.
Two-sided Sigma = D_exp + lam * C_shape, C_shape = C_walk / lambda_geo (trace = tr C_ens), so lam = lambda_geo gives
D_exp + C_walk exactly and lam = 1/N gives the independent-frame magnitude tr(C_ens)/N.
Scoring: diag z_i = r_i / sqrt(Sigma_ii) (flags, Sidak t_n), F_diag = 1/2 sum z^2 + KL, F_full = 1/2 r^T Sigma^-1 r + KL.
Rules A / B / D (+ D|ESS>=50, refs p0 / oracle) exactly as selgci q3 / lap l0 (candidates ESS >= 10% U {S=0})."""
import os, sys
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
RG = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(RG, 'selgci'))
sys.path.insert(0, os.path.join(RG, 'lap'))
import q0_common as Q          # noqa: E402
import l0_common as L0         # noqa: E402
D = Q.D
OUT = HERE + '/'
SGRID = np.r_[0.0, np.logspace(-3.5, 0.5, 33)]          # selgci q1 grid
ISO_SGRID = np.r_[0.0, np.logspace(-3, 2.5, 34)]        # selgci q2 grid
ESS_MIN = 0.10
PRIMARY = dict(k=20, bw='mean', support='union')        # the ONE kernel carried into T1-T3


# ------------------------------------------------------------- lnPF spaces ----
def moprp_lnpf(ens):
    """per-frame scaled-BV (b') lnPF, (500, 97): max(0.35 Nc + 2.0 Nh + delta, 0) -- the scaled model's metric space."""
    res, h, a, k = D.load(ens)
    delta = Q.POP_MODELS[f'{ens}|shift_clamp0|matched']['delta']
    return np.maximum(D.BC0 * h + D.BH0 * a + delta, 0.0).T


def iso_lnpf(which):
    z = np.load(Q.ISO_B + f'fit_features/features_iso_{which}.npz')
    return (0.35 * np.asarray(z['heavy_contacts'], float) + 2.0 * np.asarray(z['acceptor_contacts'], float)).T


# ------------------------------------------------------------------ kernel ----
def build_kernel(X, k=20, bw='mean', support='union', Dm=None):
    """self-tuning kernel on rows of X. sigma_i = mean (or RMS) distance to k NN (self excluded);
    K_ij = exp(-d^2/(sigma_i sigma_j)) on the symmetric-union kNN support (+ diagonal) or dense; alpha = 1/2 density
    normalisation Kt = K / sqrt(q_i q_j); S = D^-1/2 Kt D^-1/2 (eigen); L_st = I - S. Returns dict."""
    if Dm is None: Dm = cdist(X, X)
    N = len(Dm)
    M = Dm.copy(); np.fill_diagonal(M, np.inf)
    nn = np.argsort(M, axis=1, kind='stable')[:, :k]
    dk = np.take_along_axis(Dm, nn, 1)
    sig = dk.mean(1) if bw == 'mean' else np.sqrt((dk ** 2).mean(1))
    K = np.exp(-Dm ** 2 / np.outer(sig, sig))
    if support == 'union':
        A = np.zeros((N, N), bool); A[np.repeat(np.arange(N), k), nn.ravel()] = True; A |= A.T; np.fill_diagonal(A, True)
        K = K * A
    q = K.sum(1); Kt = K / np.sqrt(np.outer(q, q))
    d = Kt.sum(1); S = Kt / np.sqrt(np.outer(d, d)); S = 0.5 * (S + S.T)
    mu, phi = np.linalg.eigh(S); o = np.argsort(-mu); mu, phi = mu[o], phi[:, o]
    # connected components of the support
    from scipy.sparse.csgraph import connected_components
    ncomp = connected_components(K > 0, directed=False)[0]
    if ncomp > 1:   # degenerate mu = 1 null space: pin mode 1 to the global stationary vector sqrt(d), orthogonalise the rest
        u = np.sqrt(d) / np.linalg.norm(np.sqrt(d))
        Qm, _ = np.linalg.qr(np.column_stack([u, phi[:, :ncomp]]))
        phi[:, :ncomp] = Qm[:, :ncomp] * np.sign(Qm[:, :1].sum())
    return dict(k=k, bw=bw, support=support, sig=sig, n_capped=None, Kt=Kt, d=d, mu=mu, phi=phi, Lst=np.eye(N) - S, ncomp=int(ncomp),
                n_edges=int(((K > 0).sum() - N) // 2))


def walk_G(kern, lazy=True):
    """eigen-weights g of G = I + sum_{k>=2} (g(mu_k)-1) phi phi^T, g = (1+mu')/(1-mu'), mu' = (1+mu)/2 if lazy."""
    mu = kern['mu']; mul = (1 + mu) / 2 if lazy else mu
    g = (1 + mul) / np.clip(1 - mul, 1e-12, None); g[0] = 1.0
    # finite-chain cap: a mode cannot be more correlated than N identical frames (rho_k = 1 for all lags < N) -> g <= 2N-1.
    # Binds only for (near-)disconnected supports (ISO_BI union graph = 2 components = open | closed; MSAss k=10 mu2 .9968).
    return np.minimum(g, 2 * len(mu) - 1)


def walk_cov(kern, F, lazy=True):
    """C_walk = (1/N^2) F~^T G F~ (F~ = F centred at the uniform mean); C_ens = cov(F) (ddof 1)."""
    N = len(F); Ft = F - F.mean(0); Y = kern['phi'].T @ Ft; g = walk_G(kern, lazy)
    Cw = (Y * g[:, None]).T @ Y / N ** 2
    Ce = np.cov(F, rowvar=False)
    infl = np.trace(Cw) / (np.trace(Ce) / N)
    return Cw, Ce, dict(infl=float(infl), lam_geo=float(infl / N), N_eff=float(N / infl))


# ---------------------------------------------------- MoPrP F (rate mode) ----
def moprp_F(m):
    """per-frame quantities of the b' rate model. Returns F_rate (500, 97), Jacobian J (210, 97) of the uptake w.r.t. the
    averaged rate vector at UNIFORM weights, and F_upt (500, 210) per-frame uptake curves (uptake-mode approximation)."""
    Fr = m.Kr.T.copy()
    kbar = m.Kr.mean(1)
    E = m.T[None] * np.exp(-kbar[:, None] * m.T[None])               # (97, 15)
    J = np.einsum('pr,rt->ptr', m.M, E).reshape(-1, m.Kr.shape[0])   # (14*15, 97), row = p*15 + t
    U = 1 - np.exp(-m.Kr.T[:, :, None] * m.T[None, None])            # (500, 97, 15)
    Fu = np.einsum('pr,frt->fpt', m.M, U).reshape(m.nF, -1)
    return Fr, J, Fu


def moprp_cwalk(m, kern, mode='delta', lazy=True):
    """C_walk / C_ens in uptake-cell space (210 x 210, row-major peptide x time).
    mode 'delta': J C_walk(rate) J^T (linearised rate-mode); 'upt': per-frame uptake curves (uptake-mode approximation)."""
    Fr, J, Fu = moprp_F(m)
    if mode == 'delta':
        Cw, Ce, inf = walk_cov(kern, Fr, lazy)
        Cw, Ce = J @ Cw @ J.T, J @ Ce @ J.T
        N = Fr.shape[0]; infl = np.trace(Cw) / (np.trace(Ce) / N)
        inf = dict(infl=float(infl), lam_geo=float(infl / N), N_eff=float(N / infl), infl_rate=inf['infl'])
        return Cw, Ce, inf
    return walk_cov(kern, Fu, lazy)


# ------------------------------------------------------------ Sigma scoring ----
class Scorer:
    """held-out scoring under Sigma = D_exp + lam C_shape for a fixed fold (val rows) and a list of lams.
    model.sig (P, T) gives D_exp; C_shape is (P*T, P*T) row-major."""

    def __init__(self, model, va, C_shape, lams):
        self.va = list(va); NT = model.NT
        self.cells = np.concatenate([np.arange(r * NT, (r + 1) * NT) for r in self.va])
        self.sd_exp = model.sig[self.va].ravel()
        Cs = C_shape[np.ix_(self.cells, self.cells)]
        self.lams = list(lams); self.diag = {}; self.chol = {}; self.cond = {}
        for lam in self.lams:
            Sig = np.diag(self.sd_exp ** 2) + lam * Cs
            self.diag[lam] = np.sqrt(np.diag(Sig))
            self.chol[lam] = cho_factor(Sig, lower=True)
            ev = np.linalg.eigvalsh(Sig); self.cond[lam] = float(ev.max() / ev.min())
        self.n = len(self.cells); self.tn = Q.t_n(self.n)

    def score(self, model, w, KL):
        P = model.pred(w); r = (model.y[self.va] - P[self.va]).ravel()
        out = {}
        for lam in self.lams:
            z = r / self.diag[lam]; fl = np.abs(z) > self.tn
            maha = float(r @ cho_solve(self.chol[lam], r))
            out[lam] = dict(n_flags=int(fl.sum()), flag_mask=fl, half_z2=float(0.5 * np.sum(z ** 2)), half_maha=0.5 * maha,
                            F_diag=float(0.5 * np.sum(z ** 2) + KL), F_full=float(0.5 * maha + KL), chi2_cell=float(np.mean(z ** 2)),
                            maxabsz=float(np.abs(z).max()))
        return out


# ------------------------------------------------------------------- rules ----
def select(g, fcol='F'):
    """rules on one fold's sweep (DataFrame with S, ESS, n_flags, F, rec ...). candidates = ESS >= 10% U {S = 0}.
    A: smallest S among min-flag points; B: argmin F; D: argmin F within min-flag set; D_ess50: D on ESS >= 50% U {S=0};
    ref_p0; ref_oracle (max rec among candidates)."""
    g = g.sort_values('S').reset_index(drop=True)
    c = g[(g.ESS >= ESS_MIN) | (g.S == 0)]
    mf = c.n_flags.min(); fs = c[c.n_flags == mf]
    c50 = c[(c.ESS >= 0.5) | (c.S == 0)]; fs50 = c50[c50.n_flags == c50.n_flags.min()]
    out = {'A_flags': fs.S.idxmin(), 'B_F': c[fcol].idxmin(), 'D_flags_then_F': fs[fcol].idxmin(),
           'D_ess50': fs50[fcol].idxmin(), 'ref_p0': g.S.idxmin(), 'ref_oracle': c.rec.idxmax()}
    rows = []
    for rule, i in out.items():
        r = g.loc[i].to_dict(); r.update(rule=rule, min_flags=int(mf), n_minflag_pts=len(fs), S_index=int(i),
                                         at_grid_edge=bool(i == c.index.max()))
        rows.append(r)
    return rows


# --------------------------------------------------------- pure OMC solver ----
def fit_pomc(model, rows, eta0, S, L, c, gtol=1e-9):
    """min chi2_train(w) + (c/S) w^T L w over w = softmax(eta) (no KL; p0 enters only as the starting logits eta0)."""
    if S == 0:
        raise ValueError('S=0 is the pure-OMC attractor, not computed here')

    def f(eta):
        lw = eta - logsumexp(eta); w = np.exp(lw)
        ch, g = model.chi2_grad(w, rows); Lw = L @ w
        val = ch + (c / S) * (w @ Lw); g = g + 2 * (c / S) * Lw
        return val, w * (g - w @ g)
    r = minimize(f, eta0, jac=True, method='L-BFGS-B', options=dict(maxiter=50000, maxfun=100000, ftol=1e-16, gtol=gtol, maxcor=30))
    lw = r.x - logsumexp(r.x); w = np.exp(lw)
    _, g = model.chi2_grad(w, rows); s = g + 2 * (c / S) * (L @ w); m = w > 1e-8
    stat = float(np.sum(w[m] * np.abs(s[m] - np.sum(w[m] * s[m]) / w[m].sum())) / w[m].sum())   # KKT residual (chi2 units)
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat, fun=float(r.fun), x=r.x)
