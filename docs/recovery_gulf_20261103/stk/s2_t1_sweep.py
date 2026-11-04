"""[s2] T1 sweep: selgci q1 MoPrP sweep (floored pooled sigma, b' rate, KL to binary-openness p0, 3 contiguous 2-fold block
splits, S in {0} U logspace(-3.5, 0.5, 33), q0.fit_ef warm-started) re-scored on the held-out cells under
Sigma = D_exp + lam C_shape, C_shape = C_walk / lambda_geo of the primary kernel (s1: k=20, mean, union, lazy, alpha=1/2;
C_walk at UNIFORM weights), for F-mode 'upt' (per-frame uptake; primary), 'delta' (linearised rate; sensitivity), 'uptD' (dense kernel; sensitivity),
lam in {0, 1/500, lambda_geo/3, lambda_geo, 3 lambda_geo}. Records per point: diag flags (Sidak), F_diag, F_full, chi2/cell.
-> s2_sweep.csv, s2_weights.npz (float64 w per point), s2_flags.npz (held-out flag masks), s2.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s0_common as S0
Q, D = S0.Q, S0.D

PR = dict(np.load(S0.OUT + 's1_primary.npz'))
MODES = ('upt', 'delta', 'uptD')   # uptD = per-frame uptake, DENSE primary-k kernel (the form that passes ISO_BI in s5)


def lam_set(lg, N=500):
    return {'0': 0.0, '1/N': 1.0 / N, 'lg/3': lg / 3, 'lg': lg, '3lg': 3 * lg}


rows, W, FL = [], {}, {}
for ens in D.ENSS:
    logp0, _ = Q.moprp_p0(ens); m = Q.MoPrP(ens, floor=True)
    CS = {md: PR[f'{ens}|Cw|{md}'] / float(PR[f'{ens}|lam_geo|{md}']) for md in MODES}
    LS = {md: lam_set(float(PR[f'{ens}|lam_geo|{md}'])) for md in MODES}
    for md in MODES:
        print(f'{ens} {md}: lambda_geo {float(PR[f"{ens}|lam_geo|{md}"]):.4f} lams ' + ', '.join(f'{k}={v:.4g}' for k, v in LS[md].items()), flush=True)
    for split, held, tr, va in Q.moprp_splits():
        t0 = time.time()
        SC = {md: S0.Scorer(m, va, CS[md], list(LS[md].values())) for md in MODES}
        for md in MODES:
            print(f'   {split} {held} {md}: n={SC[md].n} cond(Sigma) ' + ', '.join(f'{k}:{SC[md].cond[v]:.3g}' for k, v in LS[md].items()), flush=True)
        Phi = Q.moprp_basis(m, tr, logp0); mu = None
        for S in S0.SGRID:
            w, lw, inf, mu_new = Q.fit_ef(m, tr, logp0, S, Phi, mu0=mu)
            if S > 0: mu = mu_new
            KL = Q.kl(w, lw, logp0); P = m.pops(w); base = dict(system='MoPrP', ensemble=ens, split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL,
                                                                 chi2_train=m.chi2_grad(w, tr)[0], stat=inf['stat'], rec=m.recovery(w),
                                                                 **{f'pop_{s}': P[i] for i, s in enumerate(Q.MO_STATES)})
            W[f'{ens}|{split}|{held}|{S:.6g}'] = w
            for md in MODES:
                sc = SC[md].score(m, w, KL)
                for lk, lam in LS[md].items():
                    o = sc[lam]
                    rows.append(dict(**base, mode=md, lam_key=lk, lam=lam, n_cells=SC[md].n, t_n=SC[md].tn, n_flags=o['n_flags'],
                                     F_diag=o['F_diag'], F_full=o['F_full'], half_z2=o['half_z2'], half_maha=o['half_maha'],
                                     chi2_cell=o['chi2_cell'], maxabsz=o['maxabsz']))
                    FL[f'{ens}|{split}|{held}|{S:.6g}|{md}|{lk}'] = np.where(o['flag_mask'])[0].astype(np.int16)
        np.save(S0.OUT + f'_cells_{ens}_{split}_{held.replace(" ", "")}.npy', SC['upt'].cells)
        g = pd.DataFrame([r for r in rows if r['ensemble'] == ens and r['split'] == split and r['heldout'] == held and r['mode'] == 'upt'])
        print(f'{ens} {split} held={held}: ' + ' | '.join(f'{lk}: flags p0 {gg.n_flags.iloc[0]} min {gg.n_flags.min()}' for lk, gg in g.groupby('lam_key', sort=False))
              + f' | t={time.time()-t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(S0.OUT + 's2_sweep.csv', index=False)
np.savez_compressed(S0.OUT + 's2_weights.npz', **W); np.savez_compressed(S0.OUT + 's2_flags.npz', **FL)
