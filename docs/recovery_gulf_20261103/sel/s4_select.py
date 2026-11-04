"""[s4] Apply every selection function per split replicate (min over the P2 pool of one model, across S, bandwidth, step),
for every arm x scope x ensemble x model x split type. Also per-replicate Spearman(-score, recovery) within the pool.
-> sel_selected.csv, sel_summary.csv, sel_summary6.csv (both split types pooled, n=6), sel_spearman.csv, sel_spearman_summary.csv"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel')
import s0_common as S
SC = ['f1_wmse', 'f2_mse', 'f3_chi2_pooled', 'f4_logerr', 'f5a_neg_spearman', 'f5b_neg_pearson', 'f6_maxabsz', 'f7a_Qn_pooled',
      'f7b_Qn_homo', 'f8_pep_norm_mse', 'f9_meancentred_mse', 'f10_offset_mse', 'f11_wmse_ess_b0.01', 'f11_wmse_ess_b0.1',
      'f11_mse_ess_b0.01', 'f11_mse_ess_b0.1']
F12 = [('f12_ws_wmse_5pct', 'f1_wmse', .05), ('f12_ws_wmse_10pct', 'f1_wmse', .10), ('f12_ws_mse_5pct', 'f2_mse', .05), ('f12_ws_mse_10pct', 'f2_mse', .10)]
KEYS = ['arm', 'scope', 'ensemble', 'model', 'split_type', 'split_idx']
TIE = ['strength', 'bandwidth', 'step', 'candidate_index']


def pick(g, col):
    return g.sort_values([col] + TIE, kind='stable').iloc[0]


def select(df, pops):
    out = []
    for key, g in df.groupby(KEYS, sort=True):
        sels = [(c, pick(g, c)) for c in SC]
        for name, base, tol in F12:
            m = g[base].min(); sub = g[g[base] <= m * (1 + tol)]
            sels.append((name, pick(sub, 'work_scale')))
        if 'val_closed_sigma_mse' in g and g.val_closed_sigma_mse.notna().all():
            sels.append(('iso_closed_sigma', pick(g, 'val_closed_sigma_mse')))
        sels.append(('ref_min_work_scale', pick(g, 'work_scale')))
        o = g.sort_values(['recovery_percent'] + TIE, ascending=[False] + [True] * 4, kind='stable').iloc[0]
        sels.append(('ref_oracle_max_recovery', o))
        for name, r in sels:
            d = dict(zip(KEYS, key)); d.update(selector=name, run_id=r.run_id, candidate_index=r.candidate_index, strength=r.strength,
                                                bandwidth=r.bandwidth, step=r.step, recovery=r.recovery_percent, ess=r.ess_percent,
                                                work_scale=r.work_scale, n_pool=len(g))
            d.update({p: r[p] for p in pops})
            out.append(d)
    return pd.DataFrame(out)


def spear(df):
    out = []
    for key, g in df.groupby(KEYS, sort=True):
        d = dict(zip(KEYS, key))
        for c in SC + ['work_scale'] + (['val_closed_sigma_mse'] if g.get('val_closed_sigma_mse') is not None and g.val_closed_sigma_mse.notna().all() else []):
            d[c] = spearmanr(-g[c], g.recovery_percent).statistic
        out.append(d)
    return pd.DataFrame(out)


def summarize(sel, pops, by):
    agg = dict(rec_mean=('recovery', 'mean'), rec_sd=('recovery', 'std'), rec_min=('recovery', 'min'), ess_mean=('ess', 'mean'),
               S_sel=('strength', lambda s: '/'.join(f'{x:g}' for x in s)), h_sel=('bandwidth', lambda s: '/'.join(f'{x:g}' for x in s)),
               step_med=('step', 'median'), n=('recovery', 'size'))
    agg.update({f'{p}_mean': (p, 'mean') for p in pops})
    return sel.groupby(by, sort=False).agg(**agg).reset_index()


if __name__ == '__main__':
    allsel, allsp = [], []
    for arm in ('full_uptake', 'rate', 'iso'):
        df = pd.read_parquet(S.SEL / f'work/s3_scores_{arm}.parquet')
        pops = S.STATES if arm != 'iso' else ['open', 'closed', 'intermediate']
        sel = select(df, pops); allsel.append(sel); allsp.append(spear(df))
        print(arm, len(sel), flush=True)
    sel = pd.concat(allsel, ignore_index=True); sp = pd.concat(allsp, ignore_index=True)
    sel.to_csv(S.SEL / 'sel_selected.csv', index=False); sp.to_csv(S.SEL / 'sel_spearman.csv', index=False)
    pops = [p for p in S.STATES + ['open', 'closed', 'intermediate']]
    sel[pops] = sel[pops] * 100
    summarize(sel, pops, ['arm', 'scope', 'ensemble', 'model', 'split_type', 'selector']).to_csv(S.SEL / 'sel_summary.csv', index=False)
    summarize(sel, pops, ['arm', 'scope', 'ensemble', 'model', 'selector']).to_csv(S.SEL / 'sel_summary6.csv', index=False)
    cols = [c for c in sp.columns if c not in KEYS]
    sp.groupby(['arm', 'scope', 'ensemble', 'model'])[cols].mean().reset_index().to_csv(S.SEL / 'sel_spearman_summary.csv', index=False)
    print('done')
