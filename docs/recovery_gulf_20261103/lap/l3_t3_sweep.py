"""[l3] T3: S sweeps with smoothed-openness references, min chi2_train(w) + KL(w||p0_tau)/S [+ gamma_abs w'Lw].
Same data / model / sigma (floored) / splits / S grid as selgci q1 (b' rate-averaged scaled BV, 14 peptides x 15 t,
3 contiguous 2-fold block splits, S in {0} U logspace(-3.5, 0.5, 33)). Family A openness only.
p0_tau from l2 ('use' solver: exact MaxEnt if feasible, else d3 log-LS soft S=100; tau=0 = selgci q0 p0).
part 'base' (gamma = 0; q0.fit_ef exponential-family solver, as q1):
   lap L and Lsym, tau in {0, 0.1, 0.3, 1, 3, 10}; + sensitivity p0 = soft S=1 at tau in {1, 10} (L, Lsym).
part 'omc' (gamma > 0; full-logit solver l0.fit_omc, warm-started along S): lap L (unnormalised, primary graph) both for
   smoothing and for the OMC term, tau in {0, 1, 10}, gamma in {0.1, 1} x c, where c (per ens x tau) = median over folds of
   (KL/S) / (w'Lw) at the gamma=0 rule-D pick (folds whose D pick is p0 excluded; fallback S=0.1 grid point) -> gamma = 1
   makes the OMC term equal to the KL term at the typical selected point. A gamma=0 full-logit control is run for tau=1.
-> l3_sweep_{part}.csv, l3_units_{part}.npz, l3_{part}.log  (l3_gamma_scale.csv for omc)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from multiprocessing import Pool
import l0_common as L0
Q, D = L0.Q, L0.D

SGRID = np.r_[0.0, np.logspace(-3.5, 0.5, 33)]
PART = sys.argv[1]
P0 = dict(np.load(L0.OUT + 'l2_p0.npz'))
T2 = pd.read_csv(L0.OUT + 'l2_t2.csv')


def p0key(ens, lap, tau, solver):
    if tau == 0: return np.asarray(Q.moprp_p0(ens)[0]), 'q0'
    if solver == 'use':
        r = T2[(T2.ensemble == ens) & (T2.graph == 'primary') & (T2.lap == lap) & (T2.fam == 'A') & (T2.tau == tau) & T2.use]
        solver = r.solver.iloc[0]
    return P0[f'{ens}|primary|{lap}|A|{tau:g}|{solver}'], solver


def run(job):
    ens, lap, tau, solver, gamma, gabs = job
    logp0, sol = p0key(ens, lap, tau, solver)
    m = Q.MoPrP(ens, floor=True); Lu, _ = L0.laplacians(L0.graph(ens))
    rows, UNITS = [], {}
    for split, held, tr, va in Q.moprp_splits():
        t0 = time.time()
        if gabs is None:
            Phi = Q.moprp_basis(m, tr, logp0); mu = None
        eta = None
        for S in SGRID:
            if gabs is None:
                w, lw, inf, mu_new = Q.fit_ef(m, tr, logp0, S, Phi, mu0=mu)
                if S > 0: mu = mu_new
            else:
                w, lw, inf = L0.fit_omc(m, tr, logp0, S, Lu, gabs, eta0=eta)
                if S > 0: eta = inf['x']
            ho, unit = Q.heldout(m, w, va)
            ctr = m.chi2_grad(w, tr)[0]; KL = Q.kl(w, lw, logp0); P = m.pops(w); wLw = float(w @ Lu @ w)
            g_abs = gabs or 0.0
            r = dict(ensemble=ens, lap=lap, tau=tau, p0solver=sol, p0req=solver, gamma=gamma, gamma_abs=g_abs, sigma='floor',
                     split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL, wLw=wLw, chi2_train=ctr,
                     J=ctr + (KL / S if S > 0 else 0) + g_abs * wLw, conv=inf['conv'], nit=inf['nit'], stat=inf['stat'],
                     **{f'ho_{k}': v for k, v in ho.items()}, F=ho['half_sum_z2'] + KL,
                     **{f'pop_{s}': P[i] for i, s in enumerate(Q.MO_STATES)}, rec=m.recovery(w))
            rows.append(r)
            UNITS[f'{ens}|{lap}|{tau:g}|{solver}|{gamma:g}|{split}|{held}|{S:.6g}'] = unit
        g = pd.DataFrame(rows[-len(SGRID):])
        print(f'{ens} {lap} tau={tau:g} p0={sol} gamma={gamma:g} ({g_abs:.3g}) {split} held={held}: ESS {g.ESS.iloc[0]:.2f}->{g.ESS.min():.3f}, '
              f'rec {g.rec.iloc[0]:.1f}..{g.rec.max():.1f}, PUF2 max {g.pop_PUF2.max():.2e}, max stat {g.stat.max():.1e}, t={time.time()-t0:.0f}s', flush=True)
    return rows, UNITS


if __name__ == '__main__':
    jobs = []
    if PART == 'base':
        for ens in D.ENSS:
            jobs.append((ens, 'L', 0.0, 'use', 0.0, None))
            for lap in ('L', 'Lsym'):
                for tau in (0.1, 0.3, 1.0, 3.0, 10.0):
                    jobs.append((ens, lap, tau, 'use', 0.0, None))
                for tau in (1.0, 10.0):
                    jobs.append((ens, lap, tau, 'soft1', 0.0, None))
    else:
        base = pd.read_csv(L0.OUT + 'l3_sweep_base.csv')
        un = dict(np.load(L0.OUT + 'l3_units_base.npz'))
        rng = np.random.default_rng(7); sc = []
        for ens in D.ENSS:
            for tau in (0.0, 1.0, 10.0):
                b = base[(base.ensemble == ens) & (base.lap == 'L') & (base.tau == tau) & (base.p0req == 'use')]
                ratios = []
                for (split, held), g in b.groupby(['split', 'heldout']):
                    sel = L0.select(g, un, lambda S: f'{ens}|L|{tau:g}|use|0|{split}|{held}|{S:.6g}', rng)
                    d = [r for r in sel if r['rule'] == 'D_flags_then_F'][0]
                    if d['S'] > 0 and d['wLw'] > 0: ratios.append(d['KL'] / d['S'] / d['wLw'])
                if ratios: c = float(np.median(ratios)); how = f'D picks ({len(ratios)}/6 folds with S>0)'
                else:
                    g1 = b[np.isclose(b.S, b.S[b.S > 0].iloc[(b.S[b.S > 0] - 0.1).abs().argsort().iloc[0]])]
                    c = float(np.median(g1.KL / g1.S / g1.wLw)); how = 'fallback S~0.1'
                sc.append(dict(ensemble=ens, tau=tau, c=c, how=how))
                for gamma in (0.1, 1.0):
                    jobs.append((ens, 'L', tau, 'use', gamma, gamma * c))
                if tau == 1.0: jobs.append((ens, 'L', tau, 'use', 0.0, 0.0))
        pd.DataFrame(sc).to_csv(L0.OUT + 'l3_gamma_scale.csv', index=False); print(pd.DataFrame(sc).to_string(index=False), flush=True)
    with Pool(5) as pool:
        res = pool.map(run, jobs, chunksize=1)
    rows = [r for rr, _ in res for r in rr]; U = {k: v for _, uu in res for k, v in uu.items()}
    pd.DataFrame(rows).to_csv(L0.OUT + f'l3_sweep_{PART}.csv', index=False); np.savez(L0.OUT + f'l3_units_{PART}.npz', **U)
