"""T3: frame-level fits with decoy-state mass fixed (0, 1%) vs unconstrained, under median-scaled BV.
Objectives: NNLS on mse / wr2 / chi2_pool (quadratic), and MaxEnt  chi2_pool/cell + (1/S) KL(w||uniform).
Decoy frames (MSAss PUF3+unfolded = 26; Filtered PUF2-like = 1) carry the fixed mass uniformly; other frames free.
Solver: NNLS = L-BFGS-B on x>=0, w_free = (1-c) x / sum(x); MaxEnt = L-BFGS on theta, w_free = (1-c) softmax(theta) (JAX value_and_grad, float64); frame_uptake NNLS cross-checked with scipy nnls.
-> p3_t3.csv, p3_weights.npz, p3.log"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.optimize import minimize, nnls
import p0_common as P, p2_landscape as L2

MODELS = ['scale|matched', 'shift|matched', 'affine_iqr|matched']
CONS = [None, 0.0, 0.01]
OBJ = [('nnls', 'mse', None), ('nnls', 'wr2', None), ('nnls', 'chi2_pool', None)] + [('maxent', 'chi2_pool', S) for S in [0.03, 0.1, 0.3, 1, 3, 10, 100]]
OBJ_ORDER = [o for o in OBJ if o[0] == 'maxent'] + [o for o in OBJ if o[0] == 'nnls']
DEX = jnp.asarray(P.DEXP); T = jnp.asarray(P.TIMES)
WMAT = {'mse': np.ones_like(P.DEXP), 'wr2': P.WTS, 'chi2_pool': 1 / P.SIG_POOL ** 2}


def make(Lf, k, M, mode, free, wdec, c, sc, S):
    rates = jnp.asarray(k[:, None] * np.exp(-Lf)); Mj = jnp.asarray(M); W = jnp.asarray(WMAT[sc])
    Dfr = jnp.einsum('pr,rft->fpt', Mj, 1 - jnp.exp(-rates[:, :, None] * T[None, None])) if mode == 'frame_uptake' else None
    N = rates.shape[1]; fi = jnp.asarray(free); wd = jnp.asarray(wdec)

    def weights(x):
        if S is not None:                      # MaxEnt: strictly positive softmax parametrisation (theta = x)
            return wd.at[fi].set((1 - c) * jax.nn.softmax(x))
        return wd.at[fi].set((1 - c) * x / jnp.sum(x))

    def pred(w):
        if mode == 'frame_uptake': return jnp.einsum('fpt,f->pt', Dfr, w)
        kb = rates @ w
        return Mj @ (1 - jnp.exp(-kb[:, None] * T[None]))

    def f(x):
        w = weights(x); r = DEX - pred(w)
        J = jnp.mean(W * r ** 2) + (1e-8 * (jnp.mean(x) - 1) ** 2 if S is None else 0.0)
        if S is not None:
            J = J + (1 / S) * jnp.sum(jax.scipy.special.xlogy(w, w * N))
        return J
    return jax.jit(jax.value_and_grad(f)), weights


rows, Wsave = [], {}
for ens in P.ENSS:
    res, h, a, k = P.load(ens); M = P.pmap(res); asg = P.assign(ens); N = len(asg)
    dec = np.isin(asg, [3, 4, 5])
    for mk in MODELS:
        Lf = L2.lnpf(f'{ens}|{mk}', h, a)
        for mode in P.PRED:
            for c in CONS:
                if c is None: free = np.arange(N); wdec = np.zeros(N); cc = 0.0
                else: free = np.where(~dec)[0]; wdec = np.where(dec, c / dec.sum(), 0.0); cc = c
                theta_prev, w_last = None, None
                for kind, sc, S in OBJ_ORDER:
                    vg, wfun = make(Lf, k, M, mode, free, wdec, cc, sc, S)
                    fun = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x)))
                    opts = dict(maxiter=100000, maxfun=200000, ftol=1e-15, gtol=1e-11)
                    if S is not None:   # MaxEnt path: continuation in S (warm start from previous S)
                        x0 = np.zeros(len(free)) if theta_prev is None else theta_prev
                        r = minimize(fun, x0, jac=True, method='L-BFGS-B', options=opts)
                        theta_prev = r.x; w_last = np.asarray(wfun(jnp.asarray(r.x)))
                    else:               # NNLS: starts from uniform and from the S=100 MaxEnt solution; keep the better
                        starts = [np.ones(len(free))] + ([w_last[free] / max(w_last[free].sum(), 1e-300) * len(free)] if w_last is not None else [])
                        r = min((minimize(fun, x0, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(free), options=opts) for x0 in starts), key=lambda z: z.fun)
                    w = np.asarray(wfun(jnp.asarray(r.x))); w = np.clip(w, 0, None); w /= w.sum()
                    if mode == 'frame_uptake' and kind == 'nnls':       # exact cross-check with scipy nnls (penalised sum)
                        rates = k[:, None] * np.exp(-Lf)
                        F = np.einsum('pr,rft->fpt', M, 1 - np.exp(-rates[:, :, None] * P.TIMES[None, None]))
                        sw = np.sqrt(WMAT[sc]); bfix = np.einsum('fpt,f->pt', F, wdec)
                        A = (F[free] * sw[None]).reshape(len(free), -1).T; b = ((P.DEXP - bfix) * sw).ravel()
                        lam = 1e3; x2, _ = nnls(np.vstack([A, lam * np.ones(len(free))]), np.append(b, lam * (1 - cc)), maxiter=100 * N)
                        w2 = wdec.copy(); w2[free] = x2; w2 /= w2.sum()
                        J1 = np.mean(WMAT[sc] * (P.DEXP - P.PRED[mode](Lf, k, M, w)) ** 2)
                        J2 = np.mean(WMAT[sc] * (P.DEXP - P.PRED[mode](Lf, k, M, w2)) ** 2)
                        nn_check = J1 - J2
                        if J2 < J1: w = w2
                    else:
                        nn_check = np.nan
                    pred = P.PRED[mode](Lf, k, M, w); s = P.scores(pred)
                    pops = P.state_pops(asg, w)
                    row = dict(ensemble=ens, model=mk, mode=mode, constraint='none' if c is None else f'decoy={c}',
                               objective=f'{kind}_{sc}' + (f'_S{S}' if S else ''), converged=bool(r.success), msg=str(r.message)[:40], J=float(r.fun), nnls_check_dJ=nn_check,
                               **{P.STATES[i]: round(pops[i], 4) for i in range(6)},
                               recovery=P.recovery(asg, w), ESS=P.ess(w), n_nonzero=int((w > 1e-6).sum()),
                               **{kk: float(s[kk]) for kk in ['mse', 'wr2', 'chi2_pool', 'maxz', 'spearman', 'mean_res']})
                    rows.append(row); Wsave['|'.join(map(str, [ens, mk, mode, row['constraint'], row['objective']]))] = w
                    print(f"{ens} {mk} {mode} {row['constraint']} {row['objective']}: F/P1/P2/P3/U/P2l "
                          + '/'.join(f'{100*p:.1f}' for p in pops) + f"  rec {row['recovery']:.1f} ESS {row['ESS']:.3f} chi2 {row['chi2_pool']:.2f} conv {r.success} dJ {nn_check:.1e}", flush=True)
D = pd.DataFrame(rows); D.to_csv(P.OUT + 'p3_t3.csv', index=False)
np.savez_compressed(P.OUT + 'p3_weights.npz', **Wsave)
# uniform-prior reference rows
