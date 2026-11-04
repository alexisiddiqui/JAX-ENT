"""[s6] T2 selection: rules A / B / D (+ D_ess50, refs) on the s4 OMC sweeps and the gamma = 0 sweep (s2, upt), under the T1
two-sided Sigma at lambda_geo (diag = primary; full) and lam = 0 (diag; selgci scoring) for contrast. Candidates ESS >= 10% U {S=0}.
Also: pure-OMC drift -- populations along the pomc path vs p0 and vs the L_st attractor w* ∝ sqrt(d).
-> s6_selected.csv, s6_summary.csv, s6.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s0_common as S0
Q, D = S0.Q, S0.D

s4 = pd.read_csv(S0.OUT + 's4_sweep.csv')
s2 = pd.read_csv(S0.OUT + 's2_sweep.csv'); s2 = s2[(s2['mode'] == 'upt') & s2.lam_key.isin(['0', 'lg'])].copy()
s2['variant'] = 'kl'; s2['gamma'] = 0.0
cols = [c for c in s4.columns if c in s2.columns]
df = pd.concat([s2[cols], s4[cols]], ignore_index=True)
PR = dict(np.load(S0.OUT + 's1_primary.npz'))

sel = []
for (ens, var, gam, lk, split, held), g in df.groupby(['ensemble', 'variant', 'gamma', 'lam_key', 'split', 'heldout'], sort=False):
    for ver, fcol in (('diag', 'F_diag'), ('full', 'F_full')):
        if lk == '0' and ver == 'full': continue
        for r in S0.select(g, fcol):
            r.update(version=ver, F=r[fcol]); sel.append(r)
S = pd.DataFrame(sel); S.to_csv(S0.OUT + 's6_selected.csv', index=False)
popc = [c for c in S.columns if c.startswith('pop_')]
keys = ['ensemble', 'variant', 'gamma', 'lam_key', 'version', 'rule']; grp = S.groupby(keys, sort=False)
summ = grp.agg(S_med=('S', 'median'), ESS_mean=('ESS', 'mean'), ESS_sd=('ESS', 'std'), rec_mean=('rec', 'mean'), rec_sd=('rec', 'std'),
               rec_min=('rec', 'min'), flags_mean=('n_flags', 'mean'), KL_mean=('KL', 'mean'), **{c: (c, 'mean') for c in popc}).reset_index()
summ['n_ESS_gt50'] = grp.ESS.apply(lambda x: int((x > 0.5).sum())).values
summ['n_rec_ge95'] = grp.rec.apply(lambda x: int((x >= 95).sum())).values
summ['n_p0'] = grp.S.apply(lambda x: int((x == 0).sum())).values
summ.to_csv(S0.OUT + 's6_summary.csv', index=False)

for ens in D.ENSS:
    print(f'\n=== {ens} (gamma scale c = {s4[s4.ensemble == ens].c.iloc[0]:.4g}) ===')
    for lk, ver in (('lg', 'diag'), ('lg', 'full'), ('0', 'diag')):
        print(f'-- Sigma: lam={lk} {ver}')
        g = summ[(summ.ensemble == ens) & (summ.lam_key == lk) & (summ.version == ver) & summ.rule.isin(['A_flags', 'B_F', 'D_flags_then_F', 'D_ess50', 'ref_oracle'])]
        for _, r in g.iterrows():
            print(f'{r.variant:5s} g={r.gamma:<4g} {r.rule:15s} S {r.S_med:<8.3g} ESS {100 * r.ESS_mean:3.0f}±{100 * r.ESS_sd:2.0f} rec {r.rec_mean:6.2f}±{r.rec_sd:4.2f} '
                  f'(min {r.rec_min:5.1f}) KL {r.KL_mean:.3f} fl {r.flags_mean:4.1f} F/P1/P2/P3/U {100 * r.pop_Folded:.1f}/{100 * r.pop_PUF1:.2f}/{100 * r.pop_PUF2:.2f}/'
                  f'{100 * r.pop_PUF3:.2f}/{100 * r.pop_unfolded:.2f} E50 {r.n_ESS_gt50} p0 {r.n_p0}')

# pure-OMC drift: populations along the path (mean over folds) vs p0 and attractor
print('\n[pure OMC drift] mean over folds, by S (lam irrelevant here)')
for ens in D.ENSS:
    m = Q.MoPrP(ens); att = PR[f'{ens}|attractor']; p0 = np.exp(Q.moprp_p0(ens)[0])
    print(f'{ens}: p0 pops {np.round(100 * m.pops(p0), 2)} rec {m.recovery(p0):.1f} ESS {Q.ess(p0):.2f} | attractor pops {np.round(100 * m.pops(att), 2)} rec {m.recovery(att):.1f}')
    for var in ('pomc', 'pomcu'):
        g = s4[(s4.ensemble == ens) & (s4.variant == var) & (s4.lam_key == 'lg')].groupby('S').agg(
            ESS=('ESS', 'mean'), rec=('rec', 'mean'), KL=('KL', 'mean'), Folded=('pop_Folded', 'mean'), PUF1=('pop_PUF1', 'mean'),
            PUF2=('pop_PUF2', 'mean'), U=('pop_unfolded', 'mean'), stat=('stat', 'max')).reset_index()
        print(f'  {var}:'); print(g.iloc[::4].round(4).to_string(index=False))
