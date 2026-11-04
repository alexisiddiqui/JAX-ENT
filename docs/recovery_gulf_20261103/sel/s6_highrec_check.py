"""[s6] What are the >80%-recovery candidates in the MoPrP P2 pools, and where do the scores rank them? (scope old_val) -> s6_highrec.csv"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel')
import s0_common as S
rows = []
for arm in ('full_uptake', 'rate'):
    d = pd.read_parquet(S.SEL / f'work/s3_scores_{arm}.parquet'); d = d[d.scope == 'old_val']
    keys = ['ensemble', 'model', 'split_type', 'split_idx']
    for c in ('f1_wmse', 'f2_mse', 'f10_offset_mse', 'f5a_neg_spearman'):
        d[c + '_pct'] = d.groupby(keys)[c].rank(pct=True)
    for (e, m), g in d.groupby(['ensemble', 'model']):
        h = g[g.recovery_percent > 80]
        rows.append(dict(arm=arm, ensemble=e, model=m, n=len(g), frac_gt80=len(h) / len(g), ess_median_gt80=h.ess_percent.median(),
                         frac_gt80_singleframe=(h.ess_percent < 0.5).mean() if len(h) else np.nan,
                         frac_gt80_ess_ge10=(h.ess_percent >= 10).mean() if len(h) else np.nan,
                         **{f'{c}_pctrank_median_gt80': h[c + '_pct'].median() for c in ('f1_wmse', 'f2_mse', 'f10_offset_mse', 'f5a_neg_spearman')},
                         max_rec_pool=g.recovery_percent.max(), max_rec_ess_ge10=g[g.ess_percent >= 10].recovery_percent.max()))
o = pd.DataFrame(rows); o.to_csv(S.SEL / 's6_highrec.csv', index=False)
pd.set_option('display.width', 250); print(o.round(3).to_string())
