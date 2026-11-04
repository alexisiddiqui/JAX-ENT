"""[o3b] follow-up of the one o3 configuration that beat gamma = 0 on ISO_TRI seed 0 (Calpha-RMSD h = 0.1 kernel, L_sym form,
gamma = 1): full gamma grid {0.1, 0.3, 1, 3} + pure OMC in the L_sym form, all 3 noise seeds x 2 sigma x 6 folds, ISO_BI
(control) and ISO_TRI; also the L_sym form of RMSD h = 1 at gamma = 1 (neighbouring bandwidth), 3 seeds.
Identical solver / scale / scoring to o3 (c from o3_scale.csv, form 'sym'). The seed-0 gamma = 1 sym runs are recomputed and
must reproduce o3 exactly (check in o5).
-> o3b_parts/*.csv, o3b_sweep.csv, o3b.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from multiprocessing import Pool
import i0_common as C
import o3_omc_iso as O3
Q, S0 = C.Q, C.S0
PARTS = C.OUT + 'o3b_parts/'


def run(job):
    which, name, sigma, seed, plan = job
    tag = f'{which}_{name.replace("|", "_")}_{sigma}_{seed}_{len(plan)}'; fn = PARTS + tag + '.csv'
    if os.path.exists(fn): return fn
    sc_df = pd.read_csv(C.OUT + 'o3_scale.csv')
    m = Q.ISO(which, sigma, seed); N = m.nF; logp0 = np.full(N, -np.log(N)); u = np.exp(logp0)
    L = O3.build_L(which, name)['sym']; B, lmL = C.sqrt_factor(L)
    c = float(sc_df[(sc_df.system == f'ISO_{which.upper()}') & (sc_df.kernel == name) & (sc_df.form == 'sym')].c.iloc[0])
    rows = []; t0 = time.time()
    for split, held, tr, va in Q.iso_splits(len(m.res)):
        qc = C.QuadChi2(m, tr); lmA = float(np.linalg.eigvalsh(qc.A).max())
        for var, gam in plan:
            th = None; wq = u.copy()
            for S in S0.ISO_SGRID:
                if S == 0:
                    w, inf = u.copy(), dict(conv=True, nit=0, stat=0.0)
                elif var == 'kl':
                    w, lw, inf, th = C.fit_dual_aug_scaled(m, tr, logp0, S, B, gam * c, th0=th, gtol=1e-9)
                else:
                    wq, inf = C.fit_pomc_pdas(qc, L, c / S, wq, lmA, lmL); w = wq
                KL = O3.kl_safe(w, logp0); P = m.pops(w); nf, F, chi = C.heldout_diag(m, w, va, KL)
                rows.append(dict(system=f'ISO_{which.upper()}', kernel=name, form='sym', variant=var, gamma=gam, c=c, sigma=sigma, seed=seed,
                                 split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL, pen=float(w @ L @ w), chi2_train=qc(w)[0], rec=m.recovery(w),
                                 pop_open=P[0], pop_closed=P[1], pop_unassigned=P[2], n_flags=nf, F=F, chi2_cell=chi,
                                 conv=inf['conv'], nit=inf['nit'], stat=inf['stat'], method=inf.get('method', 'dual')))
                if Q.ess(w) < 0.05: break
    os.makedirs(PARTS, exist_ok=True); pd.DataFrame(rows).to_csv(fn + '.tmp', index=False); os.replace(fn + '.tmp', fn)
    print(f'{tag}: {len(rows)} rows, max stat {pd.DataFrame(rows).stat.max():.1e}, t={time.time()-t0:.0f}s', flush=True)
    return fn


if __name__ == '__main__':
    full = [('kl', g) for g in C.GAMMAS] + [('pomc', 1.0)]
    jobs = [(w, 'rmsd|h0.1', s, sd, full) for w in ('tri', 'bi') for s in (0.01, 0.02) for sd in range(3)] + \
           [(w, 'rmsd|h1', s, sd, [('kl', 1.0)]) for w in ('tri', 'bi') for s in (0.01, 0.02) for sd in range(3)]
    with Pool(5) as pool:
        fns = pool.map(run, jobs, chunksize=1)
    pd.concat([pd.read_csv(f) for f in fns], ignore_index=True).to_csv(C.OUT + 'o3b_sweep.csv', index=False)
