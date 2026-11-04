"""[i0] iso_stk/ shared code (read-only analysis; run from repo root with .venv/bin/python, OMP_NUM_THREADS=4).

Reuses (imports, no rewrite):
  stk/s0_common.py   -> self-tuning kernel (build_kernel), walk covariance, Scorer, rules select(), iso_lnpf, S grids
  selgci/q0_common.py -> ISO / MoPrP models, p0, splits, exact ISO dual (gamma = 0), EF solver, kl / ess / t_n
  lap/l0_common.py   -> fit_omc pattern (chi2 + KL/S + gamma w'Lw on softmax logits)

Adds:
  * scalar-distance all-pairs kernels exactly as the policy sidecars build them
      (run_maxent_omc_sidecar.regularizer -> jaxent/src/opt/loss/original_omc_laplacian.build_omc_kernel):
      d~ = d / median(positive upper-triangle d);  K_ij = exp(-min(d~^2 / (2 h^2), 80)).
    Deviation: entries at the exponent cap (exp(-80) = 1.8e-35) are set to exactly 0 here (in the sidecars they are 1.8e-35, i.e.
    numerically inert in sidecar units; with our gamma rescaling they would otherwise turn into a complete graph).
  * three penalty forms on simplex weights w:
      'lap'  : w^T (D - K) w           = 1/2 sum_ij K_ij (w_i - w_j)^2          (coordinator's L = D - K form; PRIMARY)
      'sym'  : w^T (I - D^-1/2 K D^-1/2) w   (L_sym; degree-0 nodes -> row/col 0)
      'orig' : 1/2 sum_ij K_ij w_i w_j (w_i - w_j)^2   (the sidecars' original_omc_laplacian energy without its N^2 factor;
               the N^2 is absorbed by the gamma rescaling)
  * a fast quadratic chi2 for the linear ISO model (A = F_tr^T W F_tr / n precomputed per fold) and a generic softmax-logit
    L-BFGS solver for chi2 + [KL(w||p0)/S] + pen(w) (lap/l0.fit_omc objective, generalised to any penalty).
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
RG = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(RG, 'stk'))
import s0_common as S0          # noqa: E402  (also puts selgci + lap on the path)
Q, D, L0 = S0.Q, S0.D, S0.L0
OUT = HERE + '/'
REPO = os.path.abspath(os.path.join(RG, '..', '..'))
ISO_GEOM = os.path.join(REPO, 'artifacts/iso_policy_sidecars/geometry/')
ISO_KNN = os.path.join(REPO, 'artifacts/iso_policy_sidecars_knn20/ISO_TRI_logpf_l2_knn20.npz')
MO_GEOM = os.path.join(REPO, 'jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_policy_sidecars_fresh_20261023/geometry/')
HS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMAS = [0.1, 0.3, 1.0, 3.0]
ESS_MIN = 0.10


def median_scale(M):
    v = M[np.triu_indices(len(M), 1)]; s = float(np.median(v[v > 0]))
    return M / s, s


def sidecar_kernel(Dn, h):
    """sidecar Gaussian on median-scaled distances; capped entries -> exact 0; zero diagonal (pairwise_exact_zero)."""
    E = Dn ** 2 / (2.0 * h ** 2)
    K = np.where(E >= 80.0, 0.0, np.exp(-np.minimum(E, 80.0)))
    np.fill_diagonal(K, 0.0)
    return K


class Penalty:
    """pen(w) -> (value, grad) for form in {'lap', 'sym', 'orig'} on a fixed symmetric nonnegative K (zero diag),
    or form 'mat' for a given PSD matrix L (self-tuning L_st / sidecar kNN L_sym)."""

    def __init__(self, form, K=None, L=None):
        self.form = form
        if form == 'mat':
            self.L = L
        elif form == 'lap':
            self.L = np.diag(K.sum(1)) - K
        elif form == 'sym':
            d = K.sum(1); dm = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
            self.L = np.diag((d > 0).astype(float)) - dm[:, None] * K * dm[None]
        elif form == 'orig':
            self.K = K
        else:
            raise ValueError(form)

    def __call__(self, w):
        if self.form == 'orig':
            # E = 1/2 sum K_ij w_i w_j (w_i - w_j)^2 = sum_ij K_ij w_i^3 w_j - sum_ij K_ij w_i^2 w_j^2
            KW = self.K @ np.column_stack([w, w ** 2, w ** 3])
            val = float(w ** 3 @ KW[:, 0] - w ** 2 @ KW[:, 1])
            g = 3 * w ** 2 * KW[:, 0] + KW[:, 2] - 4 * w * KW[:, 1]
            return val, g
        Lw = self.L @ w
        return float(w @ Lw), 2 * Lw


class QuadChi2:
    """ISO chi2 restricted to training rows as an explicit quadratic: chi2(w) = w'Aw - 2b'w + c0 (== Q.ISO.chi2_grad)."""

    def __init__(self, m, rows):
        Fr = m.F[:, rows].reshape(m.nF, -1); y = m.y[rows].ravel(); iv = 1 / m.sig[rows].ravel() ** 2; n = y.size
        Fw = Fr * (iv / n)
        self.A = Fw @ Fr.T; self.b = Fw @ y; self.c0 = float(np.sum(iv * y ** 2) / n)

    def __call__(self, w):
        Aw = self.A @ w
        return float(w @ Aw - 2 * self.b @ w + self.c0), 2 * Aw - 2 * self.b


def fit_pen(chi2, logp0, S, pen=None, scale=0.0, use_kl=True, eta0=None, gtol=1e-9):
    """min chi2(w) + [KL(w||p0)/S if use_kl] + coef * pen(w) over w = softmax(eta);
    coef = scale (KL variant: gamma_abs) or scale / S (pure-OMC variant, no KL; 'c/S').
    Returns w, logw, info (stat = KKT residual in chi2 units, as lap/l0.fit_omc and stk.s0.fit_pomc)."""
    coef = scale if use_kl else scale / S

    def f(eta):
        lw = eta - logsumexp(eta); w = np.exp(lw)
        val, g = chi2(w)
        if pen is not None and coef > 0:
            pv, pg = pen(w); val += coef * pv; g = g + coef * pg
        if use_kl:
            val += np.sum(w * (lw - logp0)) / S; g = g + (lw - logp0 + 1) / S
        return val, w * (g - w @ g)
    x0 = (logp0.copy() if eta0 is None else eta0)
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=50000, maxfun=100000, ftol=1e-16, gtol=gtol, maxcor=30))
    lw = r.x - logsumexp(r.x); w = np.exp(lw)
    _, g = chi2(w)
    if pen is not None and coef > 0: g = g + coef * pen(w)[1]
    s = (lw - logp0 + S * g) if use_kl else g
    msk = w > 1e-8
    stat = float(np.sum(w[msk] * np.abs(s[msk] - np.sum(w[msk] * s[msk]) / w[msk].sum())) / w[msk].sum())
    if use_kl: stat /= S        # chi2 units
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat, x=r.x)


def heldout_diag(m, w, va, KL):
    """selgci held-out score at lam = 0 (Sigma = sigma^2 I): Sidak flags and F = 1/2 sum z^2 + KL."""
    P = m.pred(w); z = ((m.y[va] - P[va]) / m.sig[va]).ravel(); tn = Q.t_n(z.size)
    return int(np.sum(np.abs(z) > tn)), float(0.5 * np.sum(z ** 2) + KL), float(np.mean(z ** 2))


def select(g):
    return S0.select(g, 'F')


# --------------------------------------------------------------- geometry ----
def iso_bi_index():
    """BI frames are an exact, order-preserving subset of TRI frames (identical Nc/Nh rows; checked in o1)."""
    return np.load(OUT + 'o1_bi_in_tri.npy')


def iso_distances(which):
    """raw all-pairs distance matrices (work_scale, rmsd, pyrosetta) for ISO_BI / ISO_TRI, then median-scaled on the
    ensemble's own positive pairs (sidecar convention). BI = TRI submatrix."""
    out = {}
    idx = iso_bi_index() if which == 'bi' else None
    for met in ('work_scale', 'rmsd', 'pyrosetta'):
        R = np.load(ISO_GEOM + f'{met}.npz')['raw_distances'].astype(float)
        if idx is not None: R = R[np.ix_(idx, idx)]
        out[met] = median_scale(R)
    return out


def moprp_distances(ens):
    return {met: (np.load(OUT + f'o1_moprp_{ens}_{met}.npz')['normalized'], float(np.load(OUT + f'o1_moprp_{ens}_{met}.npz')['median']))
            for met in ('work_scale', 'dihedral_rmsd')}


def kernel_key(met, h):
    return 'complete' if h >= 100 else f'{met}|h{h:g}'


# ------------------------------------------------- fast exact ISO solvers ----
def sqrt_factor(L, rtol=1e-11):
    """B with B^T B = L (PSD), B = Lambda^1/2 U^T over eigenvalues > rtol * max."""
    ev, U = np.linalg.eigh(0.5 * (L + L.T)); keep = ev > rtol * ev.max()
    return (np.sqrt(ev[keep])[:, None] * U[:, keep].T), float(ev.max())


def fit_dual_aug(m, rows, logp0, S, B, gamma_abs, lam0=None, gtol=1e-10):
    """Exact convex dual of  chi2_train(w) + KL(w||p0)/S + gamma_abs ||B w||^2  (ISO linear model; extends
    selgci q0.fit_dual_linear with the OMC rows M = [F_tr; B], targets [y; 0], weights a = [iv/n; gamma_abs]).
    D(lam) = -(1/S) log sum p0 exp(-S M^T lam) - lam.c - 1/4 sum lam^2 / a;  w = p0 exp(-S M^T lam) / Z."""
    if S == 0:
        return np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0), None
    Fr = m.F[:, rows].reshape(m.nF, -1); y = m.y[rows].ravel(); n = y.size
    a1 = (1 / m.sig[rows].ravel() ** 2) / n
    nb = 0 if (B is None or gamma_abs == 0) else B.shape[0]
    a = np.r_[a1, np.full(nb, gamma_abs)]; c = np.r_[y, np.zeros(nb)]

    def MT(lam):
        v = Fr @ lam[:n]
        if nb: v = v + B.T @ lam[n:]
        return v

    def negD(lam):
        eta = logp0 - S * MT(lam); lz = logsumexp(eta); w = np.exp(eta - lz)
        Mw = np.r_[Fr.T @ w, B @ w] if nb else Fr.T @ w
        Dv = -lz / S - lam @ c - 0.25 * np.sum(lam ** 2 / a)
        return -Dv, -(Mw - c - 0.5 * lam / a)
    x0 = np.zeros(n + nb) if lam0 is None or len(lam0) != n + nb else lam0
    r = minimize(negD, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=100000, maxfun=200000, ftol=1e-16, gtol=gtol, maxcor=30))
    eta = logp0 - S * MT(r.x); lw = eta - logsumexp(eta); w = np.exp(lw)
    _, g = m.chi2_grad(w, rows)
    if nb: g = g + 2 * gamma_abs * (B.T @ (B @ w))
    s = lw - logp0 + S * g; msk = w > 1e-8
    stat = float(np.sum(w[msk] * np.abs(s[msk] - np.sum(w[msk] * s[msk]) / w[msk].sum())) / w[msk].sum()) / S
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat), r.x


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u) - 1; k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]; return np.maximum(v - css[rho] / (rho + 1), 0)


def fit_pomc_qp(qc, L, coef, Lmax_A, Lmax_L, w0, tol=1e-9, maxit=30000):
    """pure OMC (no KL): min_w  w'(A + coef L)w - 2b'w  on the simplex (strictly convex -> unique; start irrelevant).
    FISTA with gradient-restart; stop on the simplex KKT gap  max_i (g_i - min_support... ) via
    gap = sum_i w_i g_i - min_i g_i  (Frank-Wolfe duality gap, >= f(w) - f*)."""
    H = qc.A + coef * L; Lf = 2 * (Lmax_A + coef * Lmax_L); x = w0.copy(); yv = x.copy(); t = 1.0
    for it in range(maxit):
        g = 2 * (H @ yv) - 2 * qc.b
        xn = proj_simplex(yv - g / Lf)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        if (yv - xn) @ (xn - x) > 0: tn = 1.0; yv = xn.copy()      # adaptive restart
        else: yv = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
        if it % 25 == 0:
            gx = 2 * (H @ x) - 2 * qc.b; gap = float(x @ gx - gx.min())
            if gap < tol: break
    gx = 2 * (H @ x) - 2 * qc.b; gap = float(x @ gx - gx.min())
    return x, dict(conv=gap < tol, nit=it + 1, stat=gap)


def fit_pomc_pdas(qc, L, coef, w0, Lmax_A, Lmax_L, tol=1e-8, maxit=60):
    """pure OMC QP  min w'Hw - 2b'w on the simplex (H = A + coef L, PD) by a primal-dual active-set (semismooth Newton) method:
    active A = {i: z_i - w_i > 0}; on the free set solve [2H_FF -1; 1' 0][w_F; mu] = [2b_F; 1]; z = 2Hw - 2b - mu.
    Exact at convergence (set stable, w >= 0, z >= 0). Falls back to FISTA (fit_pomc_qp) warm-started from the best iterate if
    the set cycles. Stat = Frank-Wolfe gap (>= f(w) - f*)."""
    H = qc.A + coef * L; b = qc.b; N = len(b); w = w0.copy()
    g = 2 * (H @ w) - 2 * b; mu = float(np.min(g[w > 0])) if np.any(w > 0) else float(g.min()); z = g - mu
    best = (np.inf, w.copy()); seen = set()
    for it in range(maxit):
        act = (z - w) > 0; F = ~act
        key = act.tobytes()
        if key in seen: break
        seen.add(key)
        idx = np.nonzero(F)[0]; n = len(idx)
        M = np.empty((n + 1, n + 1)); M[:n, :n] = 2 * H[np.ix_(idx, idx)]; M[:n, n] = -1; M[n, :n] = 1; M[n, n] = 0
        sol = np.linalg.solve(M, np.r_[2 * b[idx], 1.0])
        w = np.zeros(N); w[idx] = sol[:n]; mu = sol[n]
        g = 2 * (H @ w) - 2 * b; z = g - mu; z[idx] = 0
        if np.all(w >= -1e-14) and np.all(z[act] >= -1e-12):
            w = np.maximum(w, 0); w /= w.sum(); g = 2 * (H @ w) - 2 * b; gap = float(w @ g - g.min())
            return w, dict(conv=gap < tol, nit=it + 1, stat=gap, method='pdas')
        wp = np.maximum(w, 0); wp /= wp.sum(); gp = 2 * (H @ wp) - 2 * b; gap = float(wp @ gp - gp.min())
        if gap < best[0]: best = (gap, wp)
    w, inf = fit_pomc_qp(qc, L, coef, Lmax_A, Lmax_L, best[1], tol=tol)
    inf['method'] = 'fista-fallback'; return w, inf


def fit_dual_aug_scaled(m, rows, logp0, S, B, gamma_abs, th0=None, gtol=1e-9):
    """fit_dual_aug in whitened dual coordinates theta_j = lam_j / sqrt(a_j): rows G = sqrt(a) M, targets sqrt(a) c,
    D(theta) = -(1/S) log sum p0 exp(-S G^T theta) - theta.(sqrt(a) c) - 1/4 |theta|^2  (uniform curvature; same optimum)."""
    if S == 0:
        return np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0), None
    Fr = m.F[:, rows].reshape(m.nF, -1); y = m.y[rows].ravel(); n = y.size
    sa1 = np.sqrt((1 / m.sig[rows].ravel() ** 2) / n)
    G1 = Fr * sa1[None]                         # (N, n) = F^T rows scaled
    nb = 0 if (B is None or gamma_abs == 0) else B.shape[0]
    G2 = np.sqrt(gamma_abs) * B if nb else None  # (nb, N)
    cy = sa1 * y

    def negD(th):
        v = G1 @ th[:n]
        if nb: v = v + G2.T @ th[n:]
        eta = logp0 - S * v; lz = logsumexp(eta); w = np.exp(eta - lz)
        Gw = np.r_[G1.T @ w, G2 @ w] if nb else G1.T @ w
        Dv = -lz / S - th[:n] @ cy - 0.25 * th @ th
        g = Gw - 0.5 * th; g[:n] -= cy
        return -Dv, -g
    x0 = np.zeros(n + nb) if th0 is None or len(th0) != n + nb else th0
    r = minimize(negD, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=100000, maxfun=200000, ftol=1e-16, gtol=gtol, maxcor=30))
    v = G1 @ r.x[:n]
    if nb: v = v + G2.T @ r.x[n:]
    eta = logp0 - S * v; lw = eta - logsumexp(eta); w = np.exp(lw)
    _, g = m.chi2_grad(w, rows)
    if nb: g = g + 2 * gamma_abs * (B.T @ (B @ w))
    s = lw - logp0 + S * g; msk = w > 1e-8
    stat = float(np.sum(w[msk] * np.abs(s[msk] - np.sum(w[msk] * s[msk]) / w[msk].sum())) / w[msk].sum()) / S
    return w, lw, dict(conv=bool(r.success), nit=int(r.nit), stat=stat), r.x
