"""Markdown tables for pop_report.md from p1/p2/p3/p4 CSVs. -> p5_tables.log"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import p0_common as P
O = P.OUT
MN = {'scale|matched': '(a)=(c) scale', 'shift|matched': '(b) shift', 'shift_clamp0|matched': "(b') shift,lnPF≥0", 'affine_iqr|matched': "(c') affine-IQR", 'none|-': 'unscaled'}
MN.update({k.split('|')[0]: v for k, v in list(MN.items())})
EN = {'AF2_MSAss': 'MSAss', 'AF2_filtered': 'Filt'}
MO = {'frame_uptake': 'frame', 'rate': 'rate'}


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def pops(r, dec=None):
    d = dec if dec is not None else r['decoys']
    return f"{100*r['Folded']:.0f}/{100*r['PUF1']:.1f}/{100*r['PUF2']:.1f}/{100*d:.1f} → **{r['recovery']:.0f}**"


A = pd.read_csv(O + 'p4_summary.csv')
# --- T2 reference (target / uniform) ---
R = pd.read_csv(O + 'p2_ref.csv')
rows = []
for (e, m, mo), g in R.groupby(['ensemble', 'model', 'mode'], sort=False):
    t = g[g.point == 'target'].iloc[0]; u = g[g.point == 'uniform'].iloc[0]
    rows.append({'ens': EN[e], 'model': MN[m], 'mode': MO[mo],
                 'χ²/cell tgt / uni': f"{t.chi2_pool:.2f} / {u.chi2_pool:.2f}", 'MSE×100 tgt / uni': f"{100*t.mse:.2f} / {100*u.mse:.2f}",
                 'ρ tgt / uni': f"{t.spearman:.3f} / {u.spearman:.3f}", 'max abs z tgt / uni': f"{t.maxz:.1f} / {u.maxz:.1f}",
                 '% decoy-free grid better than tgt (χ² / MSE / ρ)': f"{t.pct0_chi2_pool:.1f} / {t.pct0_mse:.1f} / {t.pct0_spearman:.1f}"})
print('## T2 ref\n' + md(pd.DataFrame(rows)) + '\n')
# --- T2 best per score, constraint none / decoy<=0 ---
st = A[A.level == 'state(T2)']
for c in ['none', 'decoy<=0.0', 'decoy<=0.01', 'decoy<=0.05']:
    g = st[st.constraint == c]; rows = []
    for (e, m, mo), h in g.groupby(['ensemble', 'model', 'mode'], sort=False):
        row = {'ens': EN[e], 'model': MN[m], 'mode': MO[mo]}
        for sc in ['chi2_pool', 'mse', 'wr2', 'spearman', 'maxz']:
            row[sc] = pops(h[h.selection == sc].iloc[0])
        rows.append(row)
    print(f'## T2 best, constraint {c}  (F/PUF1/PUF2/decoy % → recovery)\n' + md(pd.DataFrame(rows)) + '\n')
# --- sharpness ---
S = pd.read_csv(O + 'p4_sharp.csv'); S = S[S.constraint == 'decoy<=0.0']; rows = []
for (e, m, mo), h in S.groupby(['ensemble', 'model', 'mode'], sort=False):
    row = {'ens': EN[e], 'model': MN[m], 'mode': MO[mo]}
    for sc in ['chi2_pool', 'mse', 'spearman']:
        r = h[h.score == sc].iloc[0]
        row[sc] = f"F {100*r.F_min:.0f}–{100*r.F_max:.0f}, P1 {100*r.PUF1_min:.1f}–{100*r.PUF1_max:.1f}, rec {r.rec_min:.0f}–{r.rec_max:.0f}"
    rows.append(row)
print('## sharpness (decoy-free grid, near-optimal set)\n' + md(pd.DataFrame(rows)) + '\n')
# --- T3 ---
T3 = pd.read_csv(O + 'p3_t3.csv'); T3['decoys'] = T3[['PUF3', 'unfolded', 'PUF2-like']].sum(1)
objs = ['nnls_chi2_pool', 'nnls_mse', 'nnls_wr2', 'maxent_chi2_pool_S0.03', 'maxent_chi2_pool_S0.3', 'maxent_chi2_pool_S3', 'maxent_chi2_pool_S100']
for c in ['decoy=0.0', 'decoy=0.01', 'none']:
    rows = []
    for (e, m, mo), h in T3[T3.constraint == c].groupby(['ensemble', 'model', 'mode'], sort=False):
        row = {'ens': EN[e], 'model': MN[m], 'mode': MO[mo]}
        for o in objs:
            r = h[h.objective == o].iloc[0]
            row[o.replace('maxent_chi2_pool_', 'ME χ² ').replace('nnls_', 'NNLS ')] = pops(r) + f" (ESS {100*r.ESS:.1f}%, χ² {r.chi2_pool:.1f})"
        rows.append(row)
    print(f'## T3 frame-level, constraint {c}  (F/PUF1/PUF2/decoy % → recovery (ESS, χ²/cell))\n' + md(pd.DataFrame(rows)) + '\n')
# --- top configurations ---
top = A.sort_values('recovery', ascending=False)
top = top[top.constraint.isin(['none', 'decoy<=0.0', 'decoy=0.01', 'decoy<=0.01'])].head(25)
top = top.assign(ens=top.ensemble.map(EN), model=top.model.map(MN), mode=top['mode'].map(MO), pops=top.apply(pops, axis=1),
                 ESS=top.ESS.map(lambda x: '' if pd.isna(x) else f'{100*x:.1f}%'), chi2=top.chi2_pool.round(2))
print('## top-25 by recovery\n' + md(top[['level', 'ens', 'model', 'mode', 'selection', 'constraint', 'pops', 'ESS', 'chi2', 'script']]) + '\n')
