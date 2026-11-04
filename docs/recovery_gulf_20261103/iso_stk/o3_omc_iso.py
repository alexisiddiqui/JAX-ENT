"""[o3] ISO OMC sweeps (T1 self-tuning / sidecar-kNN graphs + main scalar-distance sidecar kernels), same data / splits / S grid /
noise / stopping as o2 (= stk s5 = selgci q2), uniform p0.
  KL variant  : min chi2_train + KL(w||u)/S + gamma_abs * pen(w), gamma in {0.1, 0.3, 1, 3}; exact augmented dual
                (i0.fit_dual_aug_scaled = i0.fit_dual_aug in whitened dual coordinates; M = [F_tr; B], B^T B = L), warm-started along S, stop at ESS < 5%.
  pure OMC    : min chi2_train + (c/S) * pen(w), no KL (S = 0 row = uniform p0, reference); exact simplex QP by primal-dual active set
                (i0.fit_pomc_pdas, FISTA fallback; strictly convex -> start-independent; warm-started along S), stop at ESS < 5%.  (c/S == 1/S' with S' = S/c.)
  gamma_abs = gamma * c,  c = median over the 36 ISO fold-runs (2 sigma x 3 seeds x 6 folds) of (KL/S)/pen(w) at the gamma = 0
  rule-D pick (o2, lam = 0 scoring), S > 0 picks only  ->  gamma = 1 makes the term equal KL/S at the typical MaxEnt pick.
Graphs / penalties:
  T1   'Lst'      : stk self-tuning L_st (k=20 mean union alpha=1/2), w'L_st w             [3 seeds]
       'knnLsym'  : sidecar logPF-kNN20 union graph, unweighted, L_sym (TRI file; BI = induced subgraph)   [3 seeds]
  main sidecar Gaussian K = exp(-min(d~^2/2h^2,80)) on median-scaled Work Scale / Calpha RMSD / PyRosetta |d ref2015|,
       h in {0.01, 0.1, 1}; h in {10, 100, 1000} == 'complete' graph (rel. deviation 3e-3..8e-3 / <= 8e-5 / <= 8e-7 after
       rescale, o1; on MoPrP the dihedral h = 10 kernel, rel. dev. 1.2e-2, gives rule-D picks identical to 'complete' within 0.01 points, o5); RMSD h=0.01 has no edges (inert == gamma 0).
       form 'lap' = w'(D-K)w (all gamma + pure OMC); form 'sym' = L_sym (gamma=1; h in {0.01, 0.1} only, where degrees are
       non-uniform -- for h >= 1 L_sym ~ L/mean degree). [seed 0; compute budget]
Scoring lam = 0 (selgci) for all; T1 also lam_geo diag (stk union C_shape).
-> o3_scale.csv, o3_parts/<job>.csv (merged: o3_sweep.csv), o3.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from multiprocessing import Pool
import i0_common as C
Q, S0 = C.Q, C.S0
PARTS = C.OUT + 'o3_parts/'
PR = dict(np.load(os.path.join(C.RG, 'stk', 's1_primary.npz')))


def kernels(which):
    """name -> (kind, metric, h, builder) ; builder returns dict form -> L (dense PSD)."""
    out = {'Lst': ('T1', 'self-tuning', np.nan), 'knnLsym': ('T1', 'sidecar-kNN20', np.nan), 'complete': ('main', 'any', 100.0)}
    for met in ('work_scale', 'rmsd', 'pyrosetta'):
        for h in (0.01, 0.1, 1.0):                        # h = 10: rel. dev. from complete 3e-3..8e-3 (o1) -> == 'complete'
            if met == 'rmsd' and h == 0.01: continue      # no edges (o1)
            out[f'{met}|h{h:g}'] = ('main', met, h)
    return out


def build_L(which, name):
    """penalty matrices, each divided by its max absolute row sum (Gershgorin bound on lambda_max) so that tiny-weight kernels
    (e.g. RMSD h=0.1, weights e^-80..e^-50) stay well scaled in the dual; the gamma rescaling c absorbs the factor."""
    return {f: L / np.abs(L).sum(1).max() for f, L in _build_L(which, name).items()}


def _build_L(which, name):
    if name == 'Lst': return {'mat': np.load(C.OUT + 'o1_graphs.npz')[f'{which}|Lst']}
    if name == 'knnLsym': return {'mat': np.load(C.OUT + 'o1_graphs.npz')[f'{which}|knn_Lsym']}
    N = 874 if which == 'bi' else 2225
    if name == 'complete':
        K = np.ones((N, N)); np.fill_diagonal(K, 0)
    else:
        met, h = name.split('|'); Dn, _ = C.iso_distances(which)[met]; K = C.sidecar_kernel(Dn, float(h[1:]))
    return {'lap': C.Penalty('lap', K=K).L, 'sym': C.Penalty('sym', K=K).L}


def kl_safe(w, logp0):
    m = w > 0; return float(np.sum(w[m] * (np.log(w[m]) - logp0[m])))


def gamma_scale():
    o2 = pd.read_csv(C.OUT + 'o2_sweep.csv'); W = np.load(C.OUT + 'o2_weights.npz')
    picks = []
    for key, g in o2.groupby(['system', 'sigma', 'seed', 'split', 'heldout'], sort=False):
        r = [x for x in C.select(g) if x['rule'] == 'D_flags_then_F'][0]; picks.append(r)
    P = pd.DataFrame(picks); rows = []
    for which in ('bi', 'tri'):
        pk = P[(P.system == f'ISO_{which.upper()}') & (P.S > 0)]
        for name, (kind, met, h) in kernels(which).items():
            Ls = build_L(which, name)
            for form, L in Ls.items():
                rat = []
                for _, r in pk.iterrows():
                    w = W[f'{which}|{r.sigma}|{r.seed}|{r.split}|{r.heldout}|{r.S:.6g}'].astype(float); w /= w.sum()
                    rat.append((r.KL / r.S) / float(w @ L @ w))
                rows.append(dict(system=f'ISO_{which.upper()}', kernel=name, kind=kind, metric=met, h=h, form=form, c=float(np.median(rat)),
                                 c_q25=float(np.quantile(rat, 0.25)), c_q75=float(np.quantile(rat, 0.75)), n_picks=len(rat)))
                print(f'[scale] ISO_{which.upper()} {name:18s} {form}: c = {rows[-1]["c"]:.4g} (IQR {rows[-1]["c_q25"]:.3g}-{rows[-1]["c_q75"]:.3g}, {len(rat)} picks)', flush=True)
    return pd.DataFrame(rows)


def run(job):
    which, name, sigma, seed = job
    tag = f'{which}_{name.replace("|", "_")}_{sigma}_{seed}'; fn = PARTS + tag + '.csv'
    if os.path.exists(fn): return fn
    sc_df = pd.read_csv(C.OUT + 'o3_scale.csv'); kind = kernels(which)[name][0]
    m = Q.ISO(which, sigma, seed); N = m.nF; logp0 = np.full(N, -np.log(N)); u = np.exp(logp0)
    Ls = build_L(which, name); fac = {f: C.sqrt_factor(L) for f, L in Ls.items()}
    lg = float(PR[f'ISO_{which}|lam_geo']); CS = PR[f'ISO_{which}|Cw'] / lg
    plan = [(f, 'kl', g) for f in Ls for g in (C.GAMMAS if f in ('lap', 'mat') else [1.0])
            if f != 'sym' or name.endswith(('h0.01', 'h0.1'))] + [(f, 'pomc', 1.0) for f in Ls if f in ('lap', 'mat')]
    rows = []; t0 = time.time()
    for split, held, tr, va in Q.iso_splits(len(m.res)):
        qc = C.QuadChi2(m, tr); lmA = float(np.linalg.eigvalsh(qc.A).max())
        sc = S0.Scorer(m, va, CS, [lg]) if kind == 'T1' else None
        for form, var, gam in plan:
            c = float(sc_df[(sc_df.system == f'ISO_{which.upper()}') & (sc_df.kernel == name) & (sc_df.form == form)].c.iloc[0])
            B, lmL = fac[form]; L = Ls[form]
            lam = None; wq = u.copy()
            grid = S0.ISO_SGRID if var == 'kl' else np.r_[0.0, S0.ISO_SGRID[1:]]
            for S in grid:
                if S == 0:
                    w, inf = u.copy(), dict(conv=True, nit=0, stat=0.0)
                elif var == 'kl':
                    w, lw, inf, lam = C.fit_dual_aug_scaled(m, tr, logp0, S, B, gam * c, th0=lam, gtol=1e-9)
                else:
                    wq, inf = C.fit_pomc_pdas(qc, L, c / S, wq, lmA, lmL); w = wq
                KL = kl_safe(w, logp0); P = m.pops(w); nf, F, chi = C.heldout_diag(m, w, va, KL)
                r = dict(system=f'ISO_{which.upper()}', kernel=name, form=form, variant=var, gamma=gam, c=c, sigma=sigma, seed=seed,
                         split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL, pen=float(w @ L @ w), chi2_train=qc(w)[0], rec=m.recovery(w),
                         pop_open=P[0], pop_closed=P[1], pop_unassigned=P[2], n_flags=nf, F=F, chi2_cell=chi,
                         conv=inf['conv'], nit=inf['nit'], stat=inf['stat'], method=inf.get('method', 'dual'))
                if sc is not None:
                    o = sc.score(m, w, KL)[lg]; r.update(n_flags_lg=o['n_flags'], F_lg=o['F_diag'])
                rows.append(r)
                if Q.ess(w) < 0.05: break
    os.makedirs(PARTS, exist_ok=True); pd.DataFrame(rows).to_csv(fn + '.tmp', index=False); os.replace(fn + '.tmp', fn)
    print(f'{tag}: {len(rows)} rows, max stat {pd.DataFrame(rows).stat.max():.1e}, t={time.time()-t0:.0f}s', flush=True)
    return fn


if __name__ == '__main__':
    if not os.path.exists(C.OUT + 'o3_scale.csv'):
        gamma_scale().to_csv(C.OUT + 'o3_scale.csv', index=False)
    jobs = []
    for which in ('tri', 'bi'):
        for name, (kind, met, h) in kernels(which).items():
            for sigma in (0.01, 0.02):
                for seed in (range(3) if kind == 'T1' else [0]):
                    jobs.append((which, name, sigma, seed))
    print(f'{len(jobs)} jobs', flush=True)
    with Pool(5) as pool:
        fns = pool.map(run, jobs, chunksize=1)
    pd.concat([pd.read_csv(f) for f in fns], ignore_index=True).to_csv(C.OUT + 'o3_sweep.csv', index=False)
