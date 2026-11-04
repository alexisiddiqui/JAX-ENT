"""[q2] ISO control sweep: ISO_BI (874 frames, truth 40/60 open/closed) and ISO_TRI (2225 frames, 1351 unassigned),
self-consistent features, BV(0.35,2.0) frame_uptake, residue-level target + Gaussian noise sigma in {0.01, 0.02}
(3 noise seeds), p0 = uniform, min chi2_train + KL(w||u)/S over S in {0} U logspace(-3, 2.5, 34), exact convex dual (q0.fit_dual_linear,
L-BFGS, warm-started along S; each fold's sweep stops at the first S with ESS < 5%). Three contiguous residue-block 2-fold splits (cuts at 1/2, 1/3, 2/3 of the 294 residues).
sigma used for z = the noise sigma (known).  usage: q2_iso_sweep.py bi|tri
-> q2_sweep_{bi,tri}.csv, q2_units_{bi,tri}.npz, q2_{bi,tri}.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import q0_common as Q

which = sys.argv[1]
SGRID = np.r_[0.0, np.logspace(-3, 2.5, 34)]
rows, UNITS = [], {}
for sigma in (0.01, 0.02):
    for seed in range(3):
        m = Q.ISO(which, sigma, seed); nF = m.nF; logp0 = np.full(nF, -np.log(nF))
        for split, held, tr, va in Q.iso_splits(len(m.res)):
            lam = None; t0 = time.time()
            for S in SGRID:
                w, lw, inf, lam_new = Q.fit_dual_linear(m, tr, logp0, S, lam0=lam, gtol=1e-12)
                if S > 0: lam = lam_new
                ho, unit = Q.heldout(m, w, va)
                ctr = m.chi2_grad(w, tr)[0]; KL = Q.kl(w, lw, logp0); P = m.pops(w)
                rows.append(dict(system=f'ISO_{which.upper()}', ensemble=f'ISO_{which.upper()}', sigma=f'{sigma}', seed=seed, split=split, heldout=held,
                                 S=S, ESS=Q.ess(w), KL=KL, chi2_train=ctr, conv=inf['conv'], nit=inf['nit'], stat=inf['stat'],
                                 **{f'ho_{k}': v for k, v in ho.items()}, F=ho['half_sum_z2'] + KL,
                                 pop_open=P[0], pop_closed=P[1], pop_unassigned=P[2], rec=m.recovery(w),
                                 ntrain_pep=len(tr), nval_pep=len(va)))
                UNITS[f'{sigma}|{seed}|{split}|{held}|{S:.6g}'] = unit
                if Q.ess(w) < 0.05: break          # sweep only needs ESS down to ~10%; stop once below 5%
            g = pd.DataFrame([r for r in rows if r['sigma'] == f'{sigma}' and r['seed'] == seed and r['split'] == split and r['heldout'] == held])
            print(f'sigma={sigma} seed={seed} {split} held={held}: ESS {g.ESS.max():.2f}->{g.ESS.min():.3f}, ho chi2 {g.ho_chi2_cell.iloc[0]:.2f}->min {g.ho_chi2_cell.min():.3f}, '
                  f'flags {g.ho_n_flags.iloc[0]}->min {g.ho_n_flags.min()}, rec {g.rec.iloc[0]:.1f}..{g.rec.max():.1f}, conv {g.conv.mean():.2f}, max stat {g.stat.max():.1e}, '
                  f't={time.time()-t0:.0f}s', flush=True)
df = pd.DataFrame(rows)
df.to_csv(Q.OUT + f'q2_sweep_{which}.csv', index=False)
np.savez(Q.OUT + f'q2_units_{which}.npz', **UNITS)
