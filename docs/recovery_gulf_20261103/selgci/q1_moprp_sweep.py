"""[q1] MoPrP sweep: per ensemble x sigma variant (floored at 0.0586 / unfloored) x 3 contiguous-block 2-fold splits x 2 folds,
min chi2_train(w) + KL(w||p0)/S over S in {0} U logspace(-3.5, 0.5, 33), b' rate-averaged scaled BV, p0 = dg T2 A|S11 MaxEnt.
Exponential-family solver (q0.fit_ef), warm-started along S. Records held-out z statistics, flags, F, KL, pops, recovery.
-> q1_sweep.csv, q1_units.npz (per-unit held-out mean z^2 for the bootstrap), q1_weights.npz, q1.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import q0_common as Q

SGRID = np.r_[0.0, np.logspace(-3.5, 0.5, 33)]
rows, UNITS, W = [], {}, {}
for ens in Q.D.ENSS:
    logp0, info = Q.moprp_p0(ens)
    print(f'{ens}: p0 constraints {info["constraints"]} dropped {info["dropped"]} max dev vs d3 {info["max_dev_vs_d3"]:.1e}', flush=True)
    for floor in (True, False):
        m = Q.MoPrP(ens, floor=floor); sv = 'floor' if floor else 'unfloored'
        for split, held, tr, va in Q.moprp_splits():
            Phi = Q.moprp_basis(m, tr, logp0); mu = None; t0 = time.time()
            for S in SGRID:
                w, lw, inf, mu_new = Q.fit_ef(m, tr, logp0, S, Phi, mu0=mu)
                if S > 0: mu = mu_new
                ho, unit = Q.heldout(m, w, va)
                ctr = m.chi2_grad(w, tr)[0]; KL = Q.kl(w, lw, logp0); P = m.pops(w)
                r = dict(system='MoPrP', ensemble=ens, sigma=sv, split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL,
                         chi2_train=ctr, J=ctr + (KL / S if S > 0 else 0), conv=inf['conv'], nit=inf['nit'], stat=inf['stat'],
                         **{f'ho_{k}': v for k, v in ho.items()}, F=ho['half_sum_z2'] + KL,
                         **{f'pop_{s}': P[i] for i, s in enumerate(Q.MO_STATES)}, rec=m.recovery(w),
                         ntrain_pep=len(tr), nval_pep=len(va))
                rows.append(r); key = f'{ens}|{sv}|{split}|{held}|{S:.6g}'; UNITS[key] = unit; W[key] = w.astype(np.float32)
            g = pd.DataFrame(rows[-len(SGRID):])
            print(f'  {sv} {split} held={held}: ESS {g.ESS.max():.2f}->{g.ESS.min():.3f}, ho chi2 {g.ho_chi2_cell.iloc[0]:.2f}->min {g.ho_chi2_cell.min():.2f}, '
                  f'flags {g.ho_n_flags.iloc[0]}->min {g.ho_n_flags.min()}, rec {g.rec.iloc[0]:.1f}..{g.rec.max():.1f}, '
                  f'all conv {g.conv.all()}, max stat {g.stat.max():.1e}, t={time.time()-t0:.0f}s', flush=True)
df = pd.DataFrame(rows)
df.to_csv(Q.OUT + 'q1_sweep.csv', index=False)
np.savez(Q.OUT + 'q1_units.npz', **UNITS); np.savez_compressed(Q.OUT + 'q1_weights.npz', **W)
