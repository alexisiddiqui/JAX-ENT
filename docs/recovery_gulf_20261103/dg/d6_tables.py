"""Markdown summary tables for dg_report.md from d2/d3/d4/d5 outputs (no new computation). -> d6_tables.md"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import d0_common as D
T = D.seg_table().set_index('seg')
t2 = pd.read_csv(D.OUT + 'd3_t2.csv'); t4 = pd.read_csv(D.OUT + 'd5_t4.csv')
out = []
f = lambda v: '–' if pd.isna(v) else (f'{v:.3f}' if abs(v) >= 0.001 or v == 0 else f'{v:.1e}')
for ens in D.ENSS:
    pick = {'uniform': t2[(t2.ensemble == ens) & (t2.method == 'uniform')].iloc[0],
            'T2 A-S11 exact': t2[(t2.ensemble == ens) & (t2.family == 'A') & (t2.set == 'S11') & (t2.method == 'exact')].iloc[0],
            'T2 A-S11 soft S=100': t2[(t2.ensemble == ens) & (t2.family == 'A') & (t2.set == 'S11') & (t2.method == 'soft') & (t2.S == 100)].iloc[0],
            'T2 B70-S11 soft S=100': t2[(t2.ensemble == ens) & (t2.family == 'B70') & (t2.set == 'S11') & (t2.method == 'soft') & (t2.S == 100)].iloc[0],
            'T4 A S=0.1 β=10 frame': t4[(t4.ensemble == ens) & (t4['mode'] == 'frame_uptake') & (t4.family == 'A') & (t4.S == 0.1) & (t4.beta == 10)].iloc[0]}
    out.append(f'\n### {ens}: achieved open fraction per segment (indicator A | B70) vs p_exp = K/(1+K)\n')
    out.append('| segment | p_exp | ' + ' | '.join(pick) + ' |'); out.append('|---|---|' + '---|' * len(pick))
    for s in D.S11:
        out.append(f'| {s} | {T.loc[s, "p_open"]:.4f} | ' + ' | '.join(f'{f(r[f"A:{s}"])} \\| {f(r[f"B70:{s}"])}' for r in pick.values()) + ' |')
open(D.OUT + 'd6_tables.md', 'w').write('\n'.join(out)); print('\n'.join(out))
