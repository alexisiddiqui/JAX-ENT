"""[s5] T3 ISO control: selgci q2 sweeps (self-consistent ISO_BI / ISO_TRI, BV(0.35,2.0) frame_uptake, residue target + noise
sigma in {0.01, 0.02} x 3 seeds, p0 = uniform, exact convex dual warm-started along S, stop at ESS < 5%, 3 contiguous residue-
block 2-fold splits) re-scored under Sigma = sigma^2 I + lam C_shape with the SAME kernel construction in ISO lnPF space
(s1 primary: k=20, mean, union, lazy, alpha=1/2, g capped at 2N-1; ISO_BI union graph = 2 components open|closed).
lam keys: 0, 1/N, 5/N (gci2 h3), lg/3, lg, 3lg (union C_shape); lgD = dense-kernel C_walk (connected; sensitivity).
Rules A / B / D (+ D_ess50, refs) as s3, diag and full versions; pass = rec >= 95 on BI.
-> s5_sweep.csv, s5_selected.csv, s5_summary.csv, s5.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s0_common as S0
Q = S0.Q

PR = dict(np.load(S0.OUT + 's1_primary.npz'))
rows = []
for which in ('bi', 'tri'):
    lg = float(PR[f'ISO_{which}|lam_geo']); lgD = float(PR[f'ISO_{which}|lam_geo|dense'])
    CS = PR[f'ISO_{which}|Cw'] / lg; CSD = PR[f'ISO_{which}|Cw|dense'] / lgD
    for sigma in (0.01, 0.02):
        for seed in range(3):
            m = Q.ISO(which, sigma, seed); N = m.nF; logp0 = np.full(N, -np.log(N))
            LS = {'0': 0.0, '1/N': 1 / N, '5/N': 5 / N, 'lg/3': lg / 3, 'lg': lg, '3lg': 3 * lg}
            if seed == 0 and sigma == 0.01:
                print(f'ISO_{which}: N={N} lambda_geo(union) {lg:.4g} = {lg * N:.1f}/N = {lg / (5 / N):.1f} x 5/N; dense {lgD:.4g} = {lgD * N:.2f}/N', flush=True)
            for split, held, tr, va in Q.iso_splits(len(m.res)):
                t0 = time.time()
                sc = S0.Scorer(m, va, CS, list(LS.values())); scD = S0.Scorer(m, va, CSD, [lgD])
                lam = None
                for S in S0.ISO_SGRID:
                    w, lw, inf, lam_new = Q.fit_dual_linear(m, tr, logp0, S, lam0=lam, gtol=1e-12)
                    if S > 0: lam = lam_new
                    KL = Q.kl(w, lw, logp0); P = m.pops(w)
                    base = dict(system=f'ISO_{which.upper()}', sigma=sigma, seed=seed, split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL,
                                stat=inf['stat'], rec=m.recovery(w), pop_open=P[0], pop_closed=P[1], pop_unassigned=P[2])
                    o = sc.score(m, w, KL); oD = scD.score(m, w, KL)
                    for lk, lv in list(LS.items()) + [('lgD', lgD)]:
                        oo = oD[lgD] if lk == 'lgD' else o[lv]
                        rows.append(dict(**base, lam_key=lk, lam=lv, n_cells=sc.n, n_flags=oo['n_flags'], F_diag=oo['F_diag'], F_full=oo['F_full'],
                                         chi2_cell=oo['chi2_cell']))
                    if Q.ess(w) < 0.05: break
                print(f'ISO_{which} sigma={sigma} seed={seed} {split} {held}: t={time.time()-t0:.0f}s', flush=True)
df = pd.DataFrame(rows); df.to_csv(S0.OUT + 's5_sweep.csv', index=False)

sel = []
for (sysn, sg, seed, lk, split, held), g in df.groupby(['system', 'sigma', 'seed', 'lam_key', 'split', 'heldout'], sort=False):
    for ver, fcol in (('diag', 'F_diag'), ('full', 'F_full')):
        for r in S0.select(g, fcol):
            r.update(version=ver, F=r[fcol]); sel.append(r)
S = pd.DataFrame(sel); S.to_csv(S0.OUT + 's5_selected.csv', index=False)

# reproduction check vs selgci q3 (ISO, lam = 0)
q3 = pd.read_csv(os.path.join(S0.RG, 'selgci', 'q3_selected.csv'), dtype={'sigma': str})
q3 = q3[q3.system.str.startswith('ISO') & q3.rule.isin(['A_flags', 'B_F', 'D_flags_then_F'])].copy(); q3['sigma'] = q3.sigma.astype(float)
mine = S[(S.lam_key == '0') & (S.version == 'diag')]
mg = q3.merge(mine, on=['system', 'sigma', 'seed', 'split', 'heldout', 'rule'], suffixes=('_q3', ''))
print(f'[check] lam=0 vs selgci q3 ISO: {len(mg)} picks, S identical {np.isclose(mg.S_q3, mg.S).sum()}/{len(mg)}, max |drec| {np.abs(mg.rec_q3 - mg.rec).max():.2e}', flush=True)

keys = ['system', 'sigma', 'lam_key', 'version', 'rule']; grp = S.groupby(keys, sort=False)
summ = grp.agg(lam=('lam', 'first'), S_med=('S', 'median'), ESS_mean=('ESS', 'mean'), ESS_sd=('ESS', 'std'), rec_mean=('rec', 'mean'),
               rec_sd=('rec', 'std'), rec_min=('rec', 'min'), flags_mean=('n_flags', 'mean'), pop_open=('pop_open', 'mean'),
               pop_unassigned=('pop_unassigned', 'mean')).reset_index()
summ['n'] = grp.size().values; summ['n_rec_ge95'] = grp.rec.apply(lambda x: int((x >= 95).sum())).values
summ['n_p0'] = grp.S.apply(lambda x: int((x == 0).sum())).values
summ.to_csv(S0.OUT + 's5_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 1000)
sh = summ[summ.rule.isin(['A_flags', 'B_F', 'D_flags_then_F', 'D_ess50', 'ref_oracle'])]
print(sh.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
