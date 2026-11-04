"""Small NumPy/SciPy optimisers for frame weights (no framework fits).
 - maxent_exact: w_i ∝ u_i exp(lam . O_i), dual minimisation (exact moment matching, feasible constraints only)
 - soft_objective: J(w) = beta*sum_s (ln(p_s(w)+eps) - ln t_s)^2 + chi2(w) [optional] + (1/S) KL(w||u), w = softmax(theta)"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import d0_common as D

EPS = 1e-6


def maxent_exact(O, t, l2=1e-8):
    """O (n_con, F) indicator/score rows, t (n_con,) targets. Returns w, lam, info."""
    nF = O.shape[1]; logu = np.full(nF, -np.log(nF))

    def dual(lam):
        z = logu + lam @ O; lz = logsumexp(z); w = np.exp(z - lz)
        return lz - lam @ t + 0.5 * l2 * lam @ lam, O @ w - t + l2 * lam
    r = minimize(dual, np.zeros(len(t)), jac=True, method='BFGS', options=dict(gtol=1e-12, maxiter=20000))
    w = softmax(logu + r.x @ O)
    return w, r.x, dict(success=bool(r.success), max_abs_err=float(np.abs(O @ w - t).max()))


class Data:
    """uptake data term (pooled-sigma chi2, mean over cells) for frame_uptake or rate averaging under a lnPF model"""

    def __init__(self, ens, mode='frame_uptake', model='scale'):
        res, h, a, k = D.load(ens)
        self.M = D.pmap(res); self.mode = mode
        L = D.lnpf(ens, model)
        self.Kr = k[:, None] * np.exp(-L)                  # (res, F) per-frame exchange rates
        if mode == 'frame_uptake':
            self.F = D.frame_uptake_F(L, k, self.M)          # (F, NP, NT)
        self.iv = 1.0 / D.SIG_POOL ** 2; self.N = D.DEXP.size

    def pred(self, w):
        if self.mode == 'frame_uptake':
            return np.einsum('fpt,f->pt', self.F, w)
        kbar = self.Kr @ w
        return self.M @ (1 - np.exp(-kbar[:, None] * D.TIMES[None]))

    def chi2_grad(self, w):
        P = self.pred(w); r = D.DEXP - P
        val = np.sum(self.iv * r ** 2) / self.N
        dP = -2 * self.iv * r / self.N                       # dchi2/dP (NP,NT)
        if self.mode == 'frame_uptake':
            g = np.einsum('fpt,pt->f', self.F, dP)
        else:
            kbar = self.Kr @ w
            E = D.TIMES[None] * np.exp(-kbar[:, None] * D.TIMES[None])   # (res,T) d(1-exp(-k t))/dk
            g = ((self.M.T @ dP) * E).sum(1) @ self.Kr                      # (F,)
        return val, g


def soft_fit(O, t, S, beta=1.0, data=None, x0=None, log_ratio=True):
    """minimise beta*R(w) + chi2(w) [if data] + (1/S) KL(w||u). O rows may be empty (beta=0 / no restraints)."""
    nF = O.shape[1] if O is not None and len(O) else data.F.shape[0] if data is not None and data.mode == 'frame_uptake' else data.Kr.shape[1]
    u = np.full(nF, 1.0 / nF); lt = np.log(t) if O is not None and len(O) else None

    def f(th):
        w = softmax(th); g = np.zeros(nF); val = 0.0
        if O is not None and len(O) and beta > 0:
            p = O @ w
            d = np.log(p + EPS) - lt
            val += beta * np.sum(d ** 2); g += beta * (2 * d / (p + EPS)) @ O
        if data is not None:
            c, gc = data.chi2_grad(w); val += c; g += gc
        lw = np.log(w + 1e-300)
        val += np.sum(w * (lw - np.log(u))) / S; g += (lw - np.log(u) + 1) / S
        return val, w * (g - w @ g)
    r = minimize(f, np.zeros(nF) if x0 is None else x0, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, maxfun=50000, ftol=1e-15, gtol=1e-10))
    return softmax(r.x), r


def summary(ens, w, a=None):
    a = D.assign(ens) if a is None else a
    P6 = D.state_pops(a, w)
    return P6
