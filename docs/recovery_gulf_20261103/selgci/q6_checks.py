"""[q6] checks/diagnostics on q1/q2 sweeps and q3 picks -> q6.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import q0_common as Q
S = pd.read_csv(Q.OUT + 'q3_selected.csv', dtype={'sigma': str})
d1 = pd.read_csv(Q.OUT + 'q1_sweep.csv')
d2 = pd.concat([pd.read_csv(Q.OUT + f'q2_sweep_{w}.csv', dtype={'sigma': str}) for w in ('bi', 'tri')])
for nm, d in (('MoPrP', d1), ('ISO', d2)):
    c = d[(d.ESS >= 0.10) & (d.S > 0)]
    print(f'{nm}: candidate fits {len(c)}; stationarity residual max {c.stat.max():.1e}, 99th pct {c.stat.quantile(.99):.1e}, median {c.stat.median():.1e}; '
          f'L-BFGS success flag {c.conv.mean():.2f} (non-success = ABNORMAL line-search stop at ftol=1e-16; stationarity is the check)')
key = ['system', 'ensemble', 'sigma', 'seed', 'split', 'heldout']
S['seed'] = S['seed'].fillna(-1)
P = S.pivot_table(index=key, columns='rule', values='S', aggfunc='first').reset_index()
for sysn, g in P.groupby('system'):
    print(f'{sysn}: B==C pick in {(g.B_F == g.C_minchi2).sum()}/{len(g)} replicates; D==B in {(g.D_flags_then_F == g.B_F).sum()}/{len(g)}; '
          f'C1SE==p0 in {(g.C_1SE == 0).sum()}/{len(g)}; A==p0 in {(g.A_flags == 0).sum()}/{len(g)}')
# KL vs half-sum-z2 at B picks
b = S[S.rule == 'B_F']
for sysn, g in b.groupby('system'):
    print(f'{sysn}: at B pick KL = {g.KL.min():.3f}-{g.KL.max():.3f}; 0.5*sum z^2 = {g.ho_half_sum_z2.min():.0f}-{g.ho_half_sum_z2.max():.0f}')
# MoPrP: irreducible flags
m = S[(S.system == 'MoPrP')]
for (e, sv), g in m.groupby(['ensemble', 'sigma']):
    p0 = g[g.rule == 'ref_p0']; print(f'{e} {sv}: flags at p0 {p0.ho_n_flags.sum()} over 6 folds; minimum over sweep {p0.min_flags.sum()} '
                                       f'({100 * p0.min_flags.sum() / p0.ho_n_flags.sum():.0f}% irreducible); min Folded over all picks {m[(m.ensemble == e) & (m.sigma == sv)].pop_Folded.min():.3f}')
# persistent flagged peptides at p0 (floor)
from collections import Counter
cnt = Counter()
for s in m[(m.rule == 'ref_p0') & (m.sigma == 'floor')].ho_flagged_units.dropna():
    for t in s.split(';'): p, n = t.rstrip(')').split('('); cnt[p] += int(n)
tot = sum(cnt.values()); print('floor p0 flags by peptide (both ensembles, all folds):', dict(cnt.most_common()), f'total {tot}; 60-74+68-75+83-90+11-26+27-31 share',
      f"{100 * sum(cnt[k] for k in ['60-74', '68-75', '83-90', '11-26', '27-31']) / tot:.0f}%")
# MoPrP pooled over both ensembles per rule (floor and unfloored)
for sv in ('floor', 'unfloored'):
    g = m[m.sigma == sv]
    print(f'MoPrP both ensembles, sigma {sv}: ' + '; '.join(f'{r} rec {x.rec.mean():.1f}±{x.rec.std():.1f} ESS {100*x.ESS.mean():.0f}% (ESS>50 {int((x.ESS>0.5).sum())}/12)' for r, x in g.groupby('rule', sort=False)))
# floor sensitivity: per-replicate change in recovery of each rule's pick, floor -> unfloored
for r in ['A_flags', 'B_F', 'C_minchi2', 'C_1SE', 'D_flags_then_F']:
    a = m[(m.rule == r) & (m.sigma == 'floor')].set_index(['ensemble', 'split', 'heldout']); u = m[(m.rule == r) & (m.sigma == 'unfloored')].set_index(['ensemble', 'split', 'heldout'])
    dr = (u.rec - a.rec); dS = u.S - a.S
    print(f'floor->unfloored {r}: rec change mean {dr.mean():+.2f}, mean |change| {dr.abs().mean():.2f}, max |change| {dr.abs().max():.2f}; same S pick {(dS == 0).sum()}/12; '
          f'ESS change mean {100 * (u.ESS - a.ESS).mean():+.0f} pts')
