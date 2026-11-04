"""[o4] MoPrP OMC with the sidecars' scalar-distance all-pairs kernels, in the selgci/stk setup: scaled BV b' (shift-clamp),
rate averaging, KL(w||p0) to the binary-DeltaG-openness p0, floored pooled sigma, 3 contiguous 2-fold block splits, SGRID.
gamma = 0 baseline = stk s2 sweep / s3 picks at lam = 0 (== selgci q1/q3; reused, not refitted).
Kernels: sidecar K = exp(-min(d~^2/2h^2, 80)) (capped -> 0) on median-scaled Work Scale and periodic phi/psi RMS (o1;
AF2_MSAss from the stored sidecar geometry, AF2_filtered recomputed with the sidecar recipe), h in {0.01, 0.1, 1, 10} (+ Work
Scale h=100, rel. dev. from complete 8e-3 / 7e-4), 'complete' = h >= 100 (dihedral) / 1000 (both) limit.
Forms: 'lap' w'(D-K)w (gamma in {0.1, 0.3, 1, 3} + pure OMC) for every kernel; sensitivity (compute budget: MSAss Work Scale
kernels + 'complete' only) 'sym' L_sym (gamma 1) and 'orig' = sidecar original_omc_laplacian energy 1/2 sum K_ij w_i w_j (w_i-w_j)^2
(gamma 1). Dihedral h=0.01 dropped: <= 14 of 500 frames connected (o1), c ~ 6e9 (o4_scale) -> degenerate. Penalties divided by their max abs row sum (scale absorbed by c).
KL variant: lap/l0.fit_omc objective via i0.fit_pen (500-logit L-BFGS, warm-started along S). Pure OMC: chi2 + (c/S) pen, no
KL, cold-started from uniform logits at each S (stk 'pomcu'). Stop at ESS < 5%.
c = median over folds of (KL/S)/pen(w) at the gamma = 0 rule-D pick (lam = 0), S > 0 picks; fallback S ~ 0.1 grid point (stk s4).
Follow-up ('followup' argument): dihedral h = 0.1 kernel in the L_sym form, gamma grid + pure OMC (analogue of ISO o3b).
-> o4_scale.csv, o4_parts/*.csv (merged o4_sweep.csv; follow-up o4b_sweep.csv), o4.log / o4b.log / o4c.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from multiprocessing import Pool
import i0_common as C
Q, D, S0 = C.Q, C.D, C.S0
PARTS = C.OUT + 'o4_parts/'
STK = os.path.join(C.RG, 'stk') + '/'


def kernels():
    out = {'complete': ('any', 1000.0)}
    for met, hs in (('work_scale', (0.01, 0.1, 1.0, 10.0, 100.0)), ('dihedral_rmsd', (0.1, 1.0, 10.0))):
        for h in hs: out[f'{met}|h{h:g}'] = (met, h)
    return out


def build_pens(ens, name):
    N = 500
    if name == 'complete':
        K = np.ones((N, N)); np.fill_diagonal(K, 0)
    else:
        met, h = name.split('|'); K = C.sidecar_kernel(C.moprp_distances(ens)[met][0], float(h[1:]))
    out = {}
    for f in ('lap', 'sym'):
        L = C.Penalty(f, K=K).L; out[f] = C.Penalty('mat', L=L / np.abs(L).sum(1).max())
    out['orig'] = C.Penalty('orig', K=K / K.sum(1).max())
    return out


def gamma0_picks():
    s3 = pd.read_csv(STK + 's3_selected.csv')
    return s3[(s3['mode'] == 'upt') & (s3.lam_key.astype(str) == '0') & (s3.version == 'diag') & (s3.rule == 'D_flags_then_F')]


def gamma_scale():
    W = np.load(STK + 's2_weights.npz'); pk = gamma0_picks(); sw = pd.read_csv(STK + 's2_sweep.csv')
    sw = sw[(sw['mode'] == 'upt') & (sw.lam_key.astype(str) == '0')]
    rows = []
    for ens in D.ENSS:
        p = pk[(pk.ensemble == ens) & (pk.S > 0)]
        if len(p): use, how = p, f'D picks ({len(p)}/6 folds S>0)'
        else:
            Sg = S0.SGRID[1:][np.argmin(np.abs(S0.SGRID[1:] - 0.1))]; use = sw[(sw.ensemble == ens) & np.isclose(sw.S, Sg)]; how = 'fallback S~0.1'
        for name in kernels():
            for form, pen in build_pens(ens, name).items():
                rat = []
                for _, r in use.iterrows():
                    w = W[f'{ens}|{r.split}|{r.heldout}|{r.S:.6g}']; rat.append((r.KL / r.S) / pen(w)[0])
                rows.append(dict(ensemble=ens, kernel=name, form=form, c=float(np.median(rat)), how=how, n=len(rat)))
                print(f'[scale] {ens} {name:20s} {form}: c = {rows[-1]["c"]:.4g} ({how})', flush=True)
    return pd.DataFrame(rows)


def run(job):
    ens, name, fold = job[:3]; ptag = job[3] if len(job) > 3 else ''
    tag = f'{ens}_{name.replace("|", "_")}_{fold}{ptag}'; fn = PARTS + tag + '.csv'
    if os.path.exists(fn): return fn
    scdf = pd.read_csv(C.OUT + 'o4_scale.csv')
    m = Q.MoPrP(ens, floor=True); logp0, _ = Q.moprp_p0(ens); pens = build_pens(ens, name)
    split, held, tr, va = Q.moprp_splits()[fold]
    chi = lambda w: m.chi2_grad(w, tr)
    plan = [('lap', 'kl', g) for g in C.GAMMAS] + [('lap', 'pomc', 1.0)]
    if (ens == 'AF2_MSAss' and name.startswith('work_scale')) or name == 'complete':     # form sensitivity (compute budget)
        plan += [('sym', 'kl', 1.0), ('orig', 'kl', 1.0)]
    if ptag == '_sym5':      # follow-up (o3b analogue): full L_sym grid on the dihedral h = 0.1 kernel
        plan = [('sym', 'kl', g) for g in C.GAMMAS] + [('sym', 'pomc', 1.0)]
    rows = []; t0 = time.time()
    for form, var, gam in plan:
        c = float(scdf[(scdf.ensemble == ens) & (scdf.kernel == name) & (scdf.form == form)].c.iloc[0]); pen = pens[form]; eta = None
        for S in S0.SGRID:
            if S == 0:
                w, lw, inf = np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0)
            elif var == 'kl':
                w, lw, inf = C.fit_pen(chi, logp0, S, pen, gam * c, True, eta0=eta); eta = inf['x']
            else:
                w, lw, inf = C.fit_pen(chi, logp0, S, pen, c, False, eta0=np.zeros(m.nF))
            KL = Q.kl(w, lw, logp0); P = m.pops(w); nf, F, chi_c = C.heldout_diag(m, w, va, KL)
            rows.append(dict(ensemble=ens, kernel=name, form=form, variant=var, gamma=gam, c=c, split=split, heldout=held, S=S, ESS=Q.ess(w),
                             KL=KL, pen=pen(w)[0], chi2_train=chi(w)[0], rec=m.recovery(w), n_flags=nf, F=F, chi2_cell=chi_c,
                             conv=inf['conv'], nit=inf['nit'], stat=inf['stat'], **{f'pop_{s}': P[i] for i, s in enumerate(Q.MO_STATES)}))
            if Q.ess(w) < 0.05: break
    os.makedirs(PARTS, exist_ok=True); pd.DataFrame(rows).to_csv(fn + '.tmp', index=False); os.replace(fn + '.tmp', fn)
    print(f'{tag}: {len(rows)} rows, max stat {pd.DataFrame(rows).stat.max():.1e}, t={time.time()-t0:.0f}s', flush=True)
    return fn


if __name__ == '__main__' and sys.argv[1:] == ['followup']:
    # analogue of the ISO RMSD-h0.1 L_sym result: dihedral h = 0.1 kernel, L_sym form, gamma grid + pure OMC, both ensembles
    jobs = [(e, 'dihedral_rmsd|h0.1', f, '_sym5') for e in D.ENSS for f in range(6)]
    with Pool(5) as pool:
        fns = pool.map(run, jobs, chunksize=1)
    pd.concat([pd.read_csv(f) for f in fns], ignore_index=True).to_csv(C.OUT + 'o4b_sweep.csv', index=False)
elif __name__ == '__main__':
    if not os.path.exists(C.OUT + 'o4_scale.csv'):
        gamma_scale().to_csv(C.OUT + 'o4_scale.csv', index=False)
    jobs = [(e, n, f) for e in D.ENSS for n in kernels() for f in range(6)]
    print(f'{len(jobs)} jobs', flush=True)
    with Pool(5) as pool:
        fns = pool.map(run, jobs, chunksize=1)
    pd.concat([pd.read_csv(f) for f in fns], ignore_index=True).to_csv(C.OUT + 'o4_sweep.csv', index=False)
