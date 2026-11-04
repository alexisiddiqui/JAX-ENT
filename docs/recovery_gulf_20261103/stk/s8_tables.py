"""[s8] formatting only: markdown tables from s1 / s3 / s5 / s6 outputs -> s8_tables.md"""
import os
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
L = []
f = lambda r: f'{r.rec_mean:.2f} ± {r.rec_sd:.2f} (min {r.rec_min:.1f}) · ESS {100 * r.ESS_mean:.0f}%'

# T0
k1 = pd.read_csv(OUT + 's1_kernel.csv')
L.append('## T0. Kernel λ_geo / N_eff per ensemble `[s1]` (lazy walk, α = ½, g ≤ 2N−1)\n')
L.append('| ensemble | F | k | σ_i | union: λ_geo (N_eff) | dense: λ_geo (N_eff) | union μ₂ | union non-lazy λ_geo (N_eff) |')
L.append('|---|---|---|---|---|---|---|---|')
for e in ['AF2_MSAss', 'AF2_filtered', 'ISO_BI', 'ISO_TRI']:
    for F in (['upt', 'delta'] if e.startswith('AF2') else ['upt(exact)']):
        for k in (10, 20):
            for bw in ('mean', 'rms'):
                g = k1[(k1.ensemble == e) & (k1.F == F) & (k1.k == k) & (k1.bw == bw) & (k1.space != 'gci2-uptake(default BV)')]
                u = g[(g.support == 'union') & g.lazy].iloc[0]; d = g[(g.support == 'dense') & g.lazy].iloc[0]; un = g[(g.support == 'union') & ~g.lazy].iloc[0]
                L.append(f'| {e} | {F} | {k} | {bw} | {u.lam_geo:.4g} ({u.N_eff:.1f}) | {d.lam_geo:.4g} ({d.N_eff:.1f}) | {u.mu2:.4f} | {un.lam_geo:.4g} ({un.N_eff:.1f}) |')
g = k1[k1.space == 'gci2-uptake(default BV)']
L.append('\ngci2 h2 construction re-run (uptake space, default BV, dense, mean, c = 1, lazy): ' +
         ', '.join(f'{r.ensemble} k={r.k}: {r.lam_geo:.4f}' for _, r in g.iterrows()) + '\n')

# T1
s3 = pd.read_csv(OUT + 's3_summary.csv')
LK = ['0', '1/N', 'lg/3', 'lg', '3lg']
for md in ('upt', 'uptD', 'delta'):
    L.append(f'\n## T1 ({md}). Rule picks vs λ, mean ± SD over 6 folds `[s3]`\n')
    L.append('Cell: recovery % ± SD (min) · ESS. Below: flags at the pick under Σ(λ) / under λ = 0 at the same w; #ESS>50; #picks = p0; #folds whose S differs from the λ = 0 pick.\n')
    for e in ('AF2_MSAss', 'AF2_filtered'):
        lgv = s3[(s3.ensemble == e) & (s3['mode'] == md) & (s3.lam_key == 'lg')].lam.iloc[0]
        L.append(f'**{e}** (λ_geo = {lgv:.4g})\n')
        L.append('| rule | version | ' + ' | '.join(f'λ = {k}' for k in LK) + ' |'); L.append('|---|---|' + '---|' * len(LK))
        for rule in ('A_flags', 'B_F', 'D_flags_then_F', 'D_ess50'):
            for ver in ('diag', 'full'):
                if ver == 'full' and rule == 'A_flags': continue
                cells, cells2 = [], []
                for k in LK:
                    v = 'diag' if k == '0' else ver
                    r = s3[(s3.ensemble == e) & (s3['mode'] == md) & (s3.lam_key == k) & (s3.version == v) & (s3.rule == rule)].iloc[0]
                    cells.append(f(r)); cells2.append(f'fl {r.flags_mean:.1f}/{r.flags_lam0_mean:.1f} · E50 {r.n_ESS_gt50} · p0 {r.n_p0} · Δ {r.n_changed_vs_lam0}')
                L.append(f'| {rule} | {ver} | ' + ' | '.join(cells) + ' |')
                L.append(f'| | | ' + ' | '.join(cells2) + ' |')
        L.append('')
fp = pd.read_csv(OUT + 's3_flags_p0.csv')
L.append('\n## T1 flags at p0 `[s3]` (sum over the folds that hold each peptide out; upt)\n')
L.append('| ensemble | λ | flags λ=0 | flags λ | disappeared |'); L.append('|---|---|---|---|---|')
for (e, k), g in fp[fp['mode'] == 'upt'].groupby(['ensemble', 'lam_key']):
    L.append(f'| {e} | {k} | {g.flags_lam0.sum()} | {g.flags_lam.sum()} | {g.gone.sum()} |')
L.append('\nPer peptide at λ_geo (upt): flags λ=0 → λ_geo, median λ_geo·C_ii/σ_ii² of disappeared | persisting cells\n')
L.append('| ensemble | peptide | flags λ=0 | flags λ_geo | gone | ratio gone | ratio kept |'); L.append('|---|---|---|---|---|---|---|')
t = fp[(fp['mode'] == 'upt') & (fp.lam_key == 'lg')].groupby(['ensemble', 'peptide'], sort=False).agg(a=('flags_lam0', 'sum'), b=('flags_lam', 'sum'), c=('gone', 'sum'),
                                                                                                       d=('ratio_gone_med', 'median'), e=('ratio_kept_med', 'median'))
for (e, p), r in t.iterrows():
    L.append(f'| {e} | {p} | {r.a} | {r.b} | {r.c} | {r.d:.2f} | {r.e:.2f} |'.replace('nan', '–'))

# T2
s6 = pd.read_csv(OUT + 's6_summary.csv')
XS = [('kl', 0.0), ('kl', 0.1), ('kl', 0.3), ('kl', 1.0), ('kl', 3.0), ('pomc', 1.0), ('pomcu', 1.0)]
XL = ['γ=0', 'γ=0.1', 'γ=0.3', 'γ=1', 'γ=3', 'pure OMC (start p0)', 'pure OMC (start uniform)']
L.append('\n## T2. OMC with L_st: picks vs γ `[s6]` (Σ at λ_geo, upt; recovery ± SD (min) · ESS)\n')
for lk, ver in (('lg', 'diag'), ('lg', 'full'), ('0', 'diag')):
    L.append(f'\n**Σ: λ = {lk}, {ver}**\n')
    L.append('| ensemble | rule | ' + ' | '.join(XL) + ' |'); L.append('|---|---|' + '---|' * len(XS))
    for e in ('AF2_MSAss', 'AF2_filtered'):
        for rule in ('A_flags', 'B_F', 'D_flags_then_F', 'D_ess50'):
            row = []
            for v, gm in XS:
                r = s6[(s6.ensemble == e) & (s6.variant == v) & (s6.gamma == gm) & (s6.lam_key == lk) & (s6.version == ver) & (s6.rule == rule)].iloc[0]
                row.append(f(r))
            L.append(f'| {e} | {rule} | ' + ' | '.join(row) + ' |')
L.append('\nPopulations at rule D (λ_geo diag), Folded / PUF1 / PUF2 / PUF3 / U %:\n')
L.append('| ensemble | ' + ' | '.join(XL) + ' |'); L.append('|---|' + '---|' * len(XS))
for e in ('AF2_MSAss', 'AF2_filtered'):
    row = []
    for v, gm in XS:
        r = s6[(s6.ensemble == e) & (s6.variant == v) & (s6.gamma == gm) & (s6.lam_key == 'lg') & (s6.version == 'diag') & (s6.rule == 'D_flags_then_F')].iloc[0]
        row.append(f'{100 * r.pop_Folded:.1f} / {100 * r.pop_PUF1:.2f} / {100 * r.pop_PUF2:.2f} / {100 * r.pop_PUF3:.2f} / {100 * r.pop_unfolded:.2f}')
    L.append(f'| {e} | ' + ' | '.join(row) + ' |')

# T3
s5 = pd.read_csv(OUT + 's5_summary.csv')
LK5 = ['0', '1/N', '5/N', 'lg/3', 'lg', '3lg', 'lgD']
L.append('\n## T3. ISO control `[s5]`: recovery mean ± SD (min) · #rec ≥ 95 of 18\n')
for sysn in ('ISO_BI', 'ISO_TRI'):
    for sg in (0.01, 0.02):
        L.append(f'\n**{sysn}, σ = {sg}**\n')
        L.append('| rule | version | ' + ' | '.join(LK5) + ' |'); L.append('|---|---|' + '---|' * len(LK5))
        for rule in ('A_flags', 'B_F', 'D_flags_then_F', 'D_ess50'):
            for ver in ('diag', 'full'):
                row = []
                for k in LK5:
                    v = 'diag' if k == '0' else ver
                    r = s5[(s5.system == sysn) & (s5.sigma == sg) & (s5.lam_key == k) & (s5.version == v) & (s5.rule == rule)].iloc[0]
                    row.append(f'{r.rec_mean:.1f} ± {r.rec_sd:.1f} ({r.rec_min:.1f}) · {r.n_rec_ge95}/18' + (f' · p0 {r.n_p0}' if r.n_p0 else ''))
                L.append(f'| {rule} | {ver} | ' + ' | '.join(row) + ' |')
open(OUT + 's8_tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:12]))
