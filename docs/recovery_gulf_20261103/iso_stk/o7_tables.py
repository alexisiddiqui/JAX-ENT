"""[o7] formatting only: markdown tables from o5 summaries (+ o3/o4 scale files) -> o7_tables.md"""
import os
import numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
SI = pd.read_csv(OUT + 'o5_summary_iso.csv'); SM = pd.read_csv(OUT + 'o5_summary_moprp.csv')
sc3 = pd.read_csv(OUT + 'o3_scale.csv'); sc4 = pd.read_csv(OUT + 'o4_scale.csv')
L = []
f = lambda r: f'{r.rec:.2f} ± {r.rec_sd:.2f} ({r.rec_min:.1f})'
VAR = [('kl', 0.1, 'lap'), ('kl', 0.3, 'lap'), ('kl', 1.0, 'lap'), ('kl', 3.0, 'lap'), ('pomc', 1.0, 'lap'), ('kl', 1.0, 'sym'), ('kl', 1.0, 'orig')]
VN = ['γ 0.1', 'γ 0.3', 'γ 1', 'γ 3', 'pure OMC', 'L_sym γ 1', 'orig γ 1']

L.append('## T-ISO-main. Scalar-distance sidecar kernels on ISO, rule D (Σ = σ²I), seed 0 × 6 folds per σ [o5]\n')
L.append('Cell: recovery mean ± SD (min) · #≥95 of 6 (BI) or unassigned % (TRI) · ESS % · paired Δ vs γ = 0 (+better/−worse folds, |Δ| > 0.5).\n')
v = SI[(SI.seeds == 'seed0') & (SI.rule == 'D_flags_then_F') & (SI.scoring == 'lam0')]
for (sysn, sg), d in v.groupby(['system', 'sigma'], sort=False):
    b = d[d.kernel == 'none'].iloc[0]
    L.append(f'\n### {sysn} σ = {sg} — γ = 0 (MaxEnt): {f(b)} · ' + (f'{int(b.n_ge95)}/6 ≥95' if sysn == 'ISO_BI' else f'unassigned {100 * b.unassigned:.1f}%')
             + f' · ESS {100 * b.ESS:.0f}%\n')
    L.append('| kernel | ' + ' | '.join(VN[:6]) + ' |'); L.append('|---' * 7 + '|')
    for k in [k for k in d.kernel.unique() if k not in ('none', 'Lst', 'knnLsym')]:
        cells = []
        for var, g, fm in VAR[:6]:
            r = d[(d.kernel == k) & (d.variant == var) & (d.gamma == g) & (d.form == fm)]
            if not len(r): cells.append('–'); continue
            r = r.iloc[0]
            cells.append(f'{r.rec:.1f} ({r.rec_min:.1f}) · ' + (f'{int(r.n_ge95)}/6' if sysn == 'ISO_BI' else f'U{100 * r.unassigned:.0f}')
                         + f' · E{100 * r.ESS:.0f} · Δ{r.d_mean:+.1f} (+{int(r.n_better)}/−{int(r.n_worse)})')
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')

L.append('\n\n## T-ISO-T1. Self-tuning L_st vs sidecar kNN20 (unweighted L_sym), 3 seeds × 6 folds per σ [o5]\n')
for sco in ('lam0', 'lam_geo'):
    v = SI[(SI.seeds == 'seeds012') & (SI.scoring == sco) & SI.rule.isin(['D_flags_then_F', 'A_flags', 'B_F'])]
    L.append(f'\n### scoring {sco} — rule D (A, B in brackets): recovery mean ± SD (min) · #≥95/18 or unassigned · ESS · open %\n')
    L.append('| system | σ | graph | variant | rule D | A | B |'); L.append('|---' * 7 + '|')
    for (sysn, sg), d in v.groupby(['system', 'sigma'], sort=False):
        for (k, var, g), dd in d.groupby(['kernel', 'variant', 'gamma'], sort=False):
            rr = {r.rule: r for _, r in dd.iterrows()}
            if 'D_flags_then_F' not in rr: continue
            r = rr['D_flags_then_F']
            extra = f'{int(r.n_ge95)}/18' if sysn == 'ISO_BI' else f'U {100 * r.unassigned:.1f}%'
            dl = f' · Δ{r.d_mean:+.2f}±{r.d_se:.2f}' if sco == 'lam0' and k != 'none' else ''
            L.append(f'| {sysn} | {sg} | {k if k != "none" else "— (γ=0 MaxEnt)"} | {var} {g:g} | {f(r)} · {extra} · E{100 * r.ESS:.0f} · open {100 * r.open:.1f}{dl} | '
                     f'{rr["A_flags"].rec:.1f} ({rr["A_flags"].rec_min:.1f}) | {rr["B_F"].rec:.1f} ({rr["B_F"].rec_min:.1f}) |')

L.append('\n\n## T-MoPrP. Scalar-distance sidecar kernels on MoPrP (b′ + rate + KL to ΔG p0), rule D, 6 folds [o5]\n')
L.append('Cell: recovery mean (min) · #folds ESS ≥ 50% · Folded/PUF1 % · paired Δ vs γ = 0 (+/−).\n')
for ens, d in SM[SM.rule == 'D_flags_then_F'].groupby('ensemble', sort=False):
    b = d[d.kernel == 'none'].iloc[0]
    L.append(f'\n### {ens} — γ = 0: {f(b)} · ESS≥50 {int(b.n_ESS50)}/6 · Folded/PUF1 {100 * b.pop_Folded:.1f}/{100 * b.pop_PUF1:.2f} · p0 picked {int(b.n_p0)}/6\n')
    L.append('| kernel | ' + ' | '.join(VN) + ' |'); L.append('|---' * 8 + '|')
    for k in [k for k in d.kernel.unique() if k != 'none']:
        cells = []
        for var, g, fm in VAR:
            r = d[(d.kernel == k) & (d.variant == var) & (d.gamma == g) & (d.form == fm)]
            if not len(r): cells.append('–'); continue
            r = r.iloc[0]
            cells.append(f'{r.rec:.1f} ({r.rec_min:.1f}) · E{int(r.n_ESS50)} · {100 * r.pop_Folded:.1f}/{100 * r.pop_PUF1:.1f} · Δ{r.d_mean:+.1f} (+{int(r.n_better)}/−{int(r.n_worse)})')
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')

L.append('\n\n## T-MoPrP-followup. Dihedral h = 0.1 kernel in the L_sym form (analogue of the ISO RMSD h = 0.1 L_sym result) [o4 followup, o5]\n')
L.append('| ensemble | variant | rule D: recovery ± SD (min) · ESS≥50 · Folded/PUF1 · Δ vs γ=0 |'); L.append('|---|---|---|')
for _, r in SM[(SM.rule == 'D_flags_then_F') & (SM.kernel == 'dihedral_rmsd|h0.1') & (SM.form == 'sym')].iterrows():
    L.append(f'| {r.ensemble} | {r.variant} γ {r.gamma:g} | {f(r)} · E{int(r.n_ESS50)}/6 · {100 * r.pop_Folded:.1f}/{100 * r.pop_PUF1:.2f} · '
             f'Δ{r.d_mean:+.2f} ± {r.d_se:.2f} (+{int(r.n_better)}/−{int(r.n_worse)}) |')

L.append('\n\n## T-scale. γ scale c (penalty normalised by its max absolute row sum) [o3, o4]\n')
L.append('| system | kernel | form | c |'); L.append('|---|---|---|---|')
for _, r in sc3.iterrows(): L.append(f'| {r.system} | {r.kernel} | {r.form} | {r.c:.4g} (IQR {r.c_q25:.3g}–{r.c_q75:.3g}) |')
for _, r in sc4.iterrows(): L.append(f'| {r.ensemble} | {r.kernel} | {r.form} | {r.c:.4g} ({r.how}) |')
open(OUT + 'o7_tables.md', 'w').write('\n'.join(L) + '\n')
print('o7_tables.md written', len(L), 'lines')
