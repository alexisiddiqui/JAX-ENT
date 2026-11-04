"""[l4] Apply selgci's rules (A, B, C, C1SE, D + D|ESS>=50%, refs p0/oracle) to the l3 sweeps (base and omc), per fold.
Candidate set = {ESS >= 10%} U {S = 0} (l0.select = q3.select + that tweak + D_ess50). Bootstrap rng seed 7 as q3.
-> l4_selected.csv (every pick), l4_summary.csv (mean +- SD over the 6 folds), l4.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import l0_common as L0

KEY = ['ensemble', 'lap', 'tau', 'p0req', 'gamma', 'gamma_abs']
sel = []
for part in ('base', 'omc'):
    p = L0.OUT + f'l3_sweep_{part}.csv'
    if not os.path.exists(p): continue
    d = pd.read_csv(p); un = dict(np.load(L0.OUT + f'l3_units_{part}.npz')); d['part'] = part
    rng = np.random.default_rng(7)
    for k, g0 in d.groupby(KEY + ['part'], sort=False):
        for (split, held), g in g0.groupby(['split', 'heldout'], sort=False):
            ens, lap, tau, req, gamma = k[0], k[1], k[2], k[3], k[4]
            sel += L0.select(g, un, lambda S: f'{ens}|{lap}|{tau:g}|{req}|{gamma:g}|{split}|{held}|{S:.6g}', rng)
S = pd.DataFrame(sel)
S.to_csv(L0.OUT + 'l4_selected.csv', index=False)
G = KEY + ['part', 'p0solver', 'rule']
agg = {'S': ['median', 'min', 'max'], 'ESS': ['mean', 'std', 'min'], 'rec': ['mean', 'std', 'min', 'max'], 'ho_chi2_cell': ['mean', 'std'],
       'ho_n_flags': ['mean', 'std'], 'F_rel': ['mean'], 'wLw': ['mean'], 'stat': ['max']}
agg.update({f'pop_{s}': 'mean' for s in L0.Q.MO_STATES})
summ = S.groupby(G, sort=False).agg(agg); summ.columns = ['_'.join(c) for c in summ.columns]
gb = S.groupby(G, sort=False)
summ['n'] = gb.size(); summ['n_ESS_ge50'] = gb.ESS.apply(lambda x: int((x >= 0.5).sum()))
summ['n_picks_p0'] = gb.S.apply(lambda x: int((x == 0).sum())); summ['n_rec_ge95'] = gb.rec.apply(lambda x: int((x >= 95).sum()))
summ = summ.reset_index(); summ.to_csv(L0.OUT + 'l4_summary.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 2000)
x = summ.copy()
for c in ['ESS_mean', 'ESS_std', 'ESS_min'] + [f'pop_{s}_mean' for s in L0.Q.MO_STATES]: x[c] = 100 * x[c]
cols = ['ensemble', 'lap', 'tau', 'p0req', 'p0solver', 'gamma', 'gamma_abs', 'rule', 'S_median', 'ESS_mean', 'ESS_std', 'rec_mean', 'rec_std', 'rec_min',
        'ho_chi2_cell_mean', 'ho_n_flags_mean', 'pop_Folded_mean', 'pop_PUF1_mean', 'pop_PUF2_mean', 'pop_PUF3_mean', 'pop_unfolded_mean',
        'n_ESS_ge50', 'n_picks_p0', 'stat_max']
print(x[cols].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
