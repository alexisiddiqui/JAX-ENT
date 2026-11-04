"""[q3] Apply the selection rules to the q1 (MoPrP) and q2 (ISO) sweeps.
Candidate set per replicate = sweep points with ESS >= 10% (always includes S=0, i.e. p0).
 A   flags: n_flags = #held-out cells with |z| > t_n (Sidak, alpha=0.05, n = #held-out cells); most regularised (smallest S)
     among the points with the minimum flag count.
 B   free energy: F = 0.5*sum_heldout z^2 + KL(w||p0); argmin.
 C   argmin held-out chi2/cell;  C1SE = smallest S with chi2 <= min + SE, SE = peptide(unit)-bootstrap SD (1000 draws) of the
     held-out chi2/cell at the argmin.
 D   A then B: argmin F within the minimum-flag set.
 refs: p0 (S=0), oracle = max recovery in the candidate set.
-> q3_selected.csv, q3_summary.csv, q3.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import q0_common as Q

ESS_MIN = 0.10
RULES = ['A_flags', 'B_F', 'C_minchi2', 'C_1SE', 'D_flags_then_F', 'ref_p0', 'ref_oracle']


def select(g, units, keyfmt, rng):
    g = g.sort_values('S').reset_index(drop=True)
    c = g[g.ESS >= ESS_MIN]
    out = {}
    mf = c.ho_n_flags.min(); fs = c[c.ho_n_flags == mf]
    out['A_flags'] = fs.S.idxmin()
    out['B_F'] = c.F.idxmin()
    i0 = c.ho_chi2_cell.idxmin(); out['C_minchi2'] = i0
    U = np.stack([units[keyfmt(S)] for S in c.S])          # (nS, n_units)
    nU = U.shape[1]; idx = rng.integers(0, nU, (1000, nU))
    boot = U[:, idx].mean(2)                                # (nS, 1000)
    se = boot[list(c.index).index(i0)].std()
    ok = c[c.ho_chi2_cell <= c.ho_chi2_cell.min() + se]
    out['C_1SE'] = ok.S.idxmin()
    out['D_flags_then_F'] = fs.F.idxmin()
    out['ref_p0'] = g.S.idxmin()
    out['ref_oracle'] = c.rec.idxmax()
    rows = []
    for rule, i in out.items():
        r = g.loc[i].to_dict(); r.update(rule=rule, se_boot=se, min_flags=int(mf), n_minflag_pts=len(fs),
                                         F_rel=g.loc[i, 'F'] - g.loc[g.S.idxmin(), 'F'], S_index=int(i), at_grid_edge=bool(i == c.index.max()))
        rows.append(r)
    return rows


rng = np.random.default_rng(7)
sel = []
d1 = pd.read_csv(Q.OUT + 'q1_sweep.csv'); u1 = dict(np.load(Q.OUT + 'q1_units.npz'))
for (ens, sv, split, held), g in d1.groupby(['ensemble', 'sigma', 'split', 'heldout']):
    sel += select(g, u1, lambda S: f'{ens}|{sv}|{split}|{held}|{S:.6g}', rng)
for which in ('bi', 'tri'):
    p = Q.OUT + f'q2_sweep_{which}.csv'
    if not os.path.exists(p): continue
    d2 = pd.read_csv(p, dtype={'sigma': str}); u2 = dict(np.load(Q.OUT + f'q2_units_{which}.npz'))
    for (sg, seed, split, held), g in d2.groupby(['sigma', 'seed', 'split', 'heldout']):
        sel += select(g, u2, lambda S: f'{sg}|{seed}|{split}|{held}|{S:.6g}', rng)
S = pd.DataFrame(sel)
S.to_csv(Q.OUT + 'q3_selected.csv', index=False)

popc = [c for c in S.columns if c.startswith('pop_')]
agg = {'S': ['mean', 'std', 'min', 'max'], 'ESS': ['mean', 'std', 'min'], 'rec': ['mean', 'std', 'min', 'max'], 'ho_chi2_cell': ['mean', 'std'],
       'ho_n_flags': ['mean', 'std'], 'F_rel': ['mean', 'std'], 'at_grid_edge': 'mean'}
agg.update({c: 'mean' for c in popc})
summ = S.groupby(['system', 'ensemble', 'sigma', 'rule'], sort=False).agg(agg)
summ.columns = ['_'.join(c) for c in summ.columns]
summ['n'] = S.groupby(['system', 'ensemble', 'sigma', 'rule'], sort=False).size()
summ['frac_rec_ge95'] = S.groupby(['system', 'ensemble', 'sigma', 'rule'], sort=False).rec.apply(lambda x: float((x >= 95).mean()))
summ['frac_ESS_gt50'] = S.groupby(['system', 'ensemble', 'sigma', 'rule'], sort=False).ESS.apply(lambda x: float((x > 0.5).mean()))
summ = summ.reset_index()
summ.to_csv(Q.OUT + 'q3_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
print(summ.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
