"""T2: restrained reweighting to per-segment K_op (no BV, no uptake).
Families: A (rule features; exact on grouped identical indicators), B70 (Q_seg<0.7), C (continuous scaled-BV K_BV,f; target K).
Sets: S9 (brief's segments, corrected numbering), S11 (+205-212, +217-223), SN (N-subdomain segments only).
Exact MaxEnt: identical indicator rows merged (geometric-mean target), rows with no open frame dropped (infeasible).
Soft: min sum_s (ln p_s - ln t_s)^2 + (1/S) KL(w||u), S in {1,10,100,1000}, one row per segment.
-> d3_t2.csv, d3_weights.npz, d3.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import d0_common as D, d0b_solvers as V

T = D.seg_table().set_index('seg'); IND = dict(np.load(D.OUT + 'd2_indicators.npz'))
TG = D.alt_targets()
rows, W = [], {}
SN = ['127-132', '133-143', '133-148', '149-153', '154-167', '155-162']   # N-subdomain only (no alpha2/alpha3 segments)


def rowsfor(ens, fam, segs):
    O, t, names = [], [], []
    for s in segs:
        o = IND[f'{ens}|{s}|{fam}']
        if np.isnan(o).all(): continue           # not featured (A: 133-143)
        O.append(o); t.append(T.loc[s, 'K'] if fam == 'C' else T.loc[s, 'p_open']); names.append(s)
    return np.array(O), np.array(t), names


def merge_for_exact(O, t, names):
    keep, tt, nn, dropped = [], [], [], []
    seen = {}
    for o, ti, n in zip(O, t, names):
        key = o.tobytes()
        if key in seen: seen[key][1].append(ti); seen[key][2].append(n)
        else: seen[key] = [o, [ti], [n]]
    for o, ts, ns in seen.values():
        g = float(np.exp(np.mean(np.log(ts))))
        if not (o.min() < g < o.max()): dropped.append('+'.join(ns)); continue
        keep.append(o); tt.append(g); nn.append('+'.join(ns))
    return np.array(keep), np.array(tt), nn, dropped


def record(ens, fam, sset, method, S, w, extra=''):
    a = D.assign(ens); P6 = D.state_pops(a, w)
    r = dict(ensemble=ens, family=fam, set=sset, method=method, S=S, ESS=D.ess(w),
             **{f'pop_{s}': P6[i] for i, s in enumerate(D.STATES)},
             **{f'rec_{k}': D.recovery_vs(P6, v) for k, v in TG.items()}, note=extra)
    assert abs(r['rec_ORIG'] - D.recovery(a, w)) < 1e-8
    # achieved per-segment open fractions under each family (cross-check)
    for s in D.S11:
        for f2 in ['A', 'B70', 'C']:
            o = IND[f'{ens}|{s}|{f2}']
            r[f'{f2}:{s}'] = float(o @ w) if not np.isnan(o).all() else np.nan
    # log-ratio mismatch on own family
    for f2 in ['A', 'B70', 'C']:
        O, t, names = rowsfor(ens, f2, D.S11)
        r[f'rms_logratio_{f2}'] = float(np.sqrt(np.mean((np.log(O @ w + V.EPS) - np.log(t)) ** 2)))
    rows.append(r); W[f'{ens}|{fam}|{sset}|{method}|{S}'] = w


for ens in D.ENSS:
    u = np.ones(500) / 500
    record(ens, '-', '-', 'uniform', np.nan, u)
    for fam in ['A', 'B70', 'C']:
        for sset, segs in [('S9', D.S9), ('S11', D.S11), ('SN', SN)]:
            O, t, names = rowsfor(ens, fam, segs)
            Oe, te, ne, dropped = merge_for_exact(O, t, names)
            if len(Oe):
                w, lam, info = V.maxent_exact(Oe, te)
                record(ens, fam, sset, 'exact', np.inf, w, f"constraints={ne}; dropped(infeasible)={dropped}; max_err={info['max_abs_err']:.1e}")
            for S in [1, 10, 100, 1000]:
                w, r = V.soft_fit(O, t, S)
                record(ens, fam, sset, 'soft', S, w)
df = pd.DataFrame(rows)
df.to_csv(D.OUT + 'd3_t2.csv', index=False); np.savez(D.OUT + 'd3_weights.npz', **W)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 200)
cols = ['ensemble', 'family', 'set', 'method', 'S', 'ESS', 'pop_Folded', 'pop_PUF1', 'pop_PUF2', 'pop_PUF3', 'pop_unfolded', 'pop_PUF2-like',
        'rec_ORIG', 'rec_ALT1', 'rec_ALT2', 'rms_logratio_A', 'rms_logratio_B70', 'rms_logratio_C']
print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.4g}'))
print()
for _, r in df[df.method == 'exact'].iterrows(): print(r.ensemble, r.family, r.set, r.note)
