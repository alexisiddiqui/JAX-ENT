"""[s5] Verdict tables + figures from s4 outputs.
-> sel_verdict.csv (per selector x scope), sel_iso_check.csv, sel_pops_maxent.csv, F_<arm>_<scope>.png, F_iso_val.png, F_spearman.png,
   tables_*.md (markdown fragments pasted into sel_report.md)"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel')
import s0_common as S
O = S.SEL
sel = pd.read_csv(O / 'sel_selected.csv'); summ = pd.read_csv(O / 'sel_summary.csv'); sp = pd.read_csv(O / 'sel_spearman.csv')
SELS = ['f1_wmse', 'f2_mse', 'f3_chi2_pooled', 'f4_logerr', 'f5a_neg_spearman', 'f5b_neg_pearson', 'f6_maxabsz', 'f7a_Qn_pooled', 'f7b_Qn_homo',
        'f8_pep_norm_mse', 'f9_meancentred_mse', 'f10_offset_mse', 'f11_wmse_ess_b0.01', 'f11_wmse_ess_b0.1', 'f11_mse_ess_b0.01', 'f11_mse_ess_b0.1',
        'f12_ws_wmse_5pct', 'f12_ws_wmse_10pct', 'f12_ws_mse_5pct', 'f12_ws_mse_10pct', 'iso_closed_sigma', 'ref_min_work_scale', 'ref_oracle_max_recovery']
MODELS = [('AF2_MSAss', 'maxent'), ('AF2_MSAss', 'omc_work_k0'), ('AF2_MSAss', 'omc_dihed_k0'), ('AF2_MSAss', 'omc_work_k20'),
          ('AF2_MSAss', 'omc_dihed_k20'), ('AF2_filtered', 'maxent')]
ST = ['sequence_cluster', 'spatial']

# ---- ISO check (val scope = what a validation selector sees)
iso = summ[(summ.arm == 'iso') & (summ.scope == 'val')]
bi = iso[iso.ensemble == 'ISO_BI'].pivot_table(index='selector', columns='split_type', values=['rec_mean', 'rec_sd', 'rec_min'])
tri = iso[(iso.ensemble == 'ISO_TRI') & (iso.model == 'maxent')].pivot_table(index='selector', columns='split_type', values='rec_mean')
tri_omc = iso[(iso.ensemble == 'ISO_TRI') & (iso.model != 'maxent')].groupby('selector').rec_mean.mean()
chk = pd.DataFrame({'BI_seq': bi[('rec_mean', 'sequence_cluster')], 'BI_seq_sd': bi[('rec_sd', 'sequence_cluster')],
                    'BI_spa': bi[('rec_mean', 'spatial')], 'BI_spa_sd': bi[('rec_sd', 'spatial')],
                    'BI_min_rep': bi['rec_min'].min(1), 'TRI_maxent_seq': tri['sequence_cluster'], 'TRI_maxent_spa': tri['spatial'],
                    'TRI_omc_mean': tri_omc})
chk['iso_pass'] = (chk[['BI_seq', 'BI_spa']].min(1) >= 95) & (chk.BI_min_rep >= 90)
chk = chk.reindex([s for s in SELS if s in chk.index]); chk.to_csv(O / 'sel_iso_check.csv')

# ---- MoPrP verdict per selector x scope (24 cells = 2 arms x 6 models x 2 split types; 72 replicate selections)
mo = sel[sel.arm != 'iso'].copy()
mo['folded_major'] = mo.Folded > 0.5
mo['bad'] = mo.PUF3 + mo.unfolded
rows = []
for (scope, s), g in mo.groupby(['scope', 'selector']):
    cell = g.groupby(['arm', 'ensemble', 'model', 'split_type']).recovery.agg(['mean', 'std'])
    me = g[g.model == 'maxent']
    rows.append(dict(scope=scope, selector=s, rec_mean=g.recovery.mean(), rec_median=g.recovery.median(), cell_sd_mean=cell['std'].mean(),
                     n_cells_ge80=int((cell['mean'] >= 80).sum()), n_reps_ge80=int((g.recovery >= 80).sum()), n_reps=len(g),
                     folded_major_frac=g.folded_major.mean(), puf3_u_mean=100 * g.bad.mean(), folded_mean=100 * g.Folded.mean(),
                     rec_full_uptake=g[g.arm == 'full_uptake'].recovery.mean(), rec_rate=g[g.arm == 'rate'].recovery.mean(),
                     rec_maxent=me.recovery.mean(), rec_omc=g[g.model != 'maxent'].recovery.mean(), ess_mean=g.ess.mean()))
V = pd.DataFrame(rows)
V['iso_pass'] = V.selector.map(chk.iso_pass).fillna(False)
V.to_csv(O / 'sel_verdict.csv', index=False)

# ---- populations for MaxEnt cells (primary scope)
pm = summ[(summ.arm != 'iso') & (summ.scope.isin(['old_val', 'new_all'])) & (summ.model == 'maxent')]
pm.to_csv(O / 'sel_pops_maxent.csv', index=False)


def fmt(m, s):
    return f'{m:.1f} ± {s:.1f}' if np.isfinite(s) else f'{m:.1f}'


def md(df):
    cols = list(df.columns)
    return '\n'.join(['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in df.itertuples(index=False)])


with open(O / 'tables_iso.md', 'w') as f:
    t = chk.reset_index().rename(columns={'index': 'selector'})
    t = pd.DataFrame({'selector': t.selector, 'BI seq': [fmt(a, b) for a, b in zip(t.BI_seq, t.BI_seq_sd)],
                      'BI spatial': [fmt(a, b) for a, b in zip(t.BI_spa, t.BI_spa_sd)], 'BI worst rep': t.BI_min_rep.round(1),
                      'TRI MaxEnt seq/spa': [f'{a:.1f} / {b:.1f}' for a, b in zip(t.TRI_maxent_seq, t.TRI_maxent_spa)],
                      'TRI OMC (mean of 3 graphs × 2 splits)': t.TRI_omc_mean.round(1), 'ISO pass': np.where(t.iso_pass, 'PASS', 'fail')})
    f.write(md(t))

for scope in ('old_val', 'old_all', 'new_val', 'new_all'):
    with open(O / f'tables_{scope}.md', 'w') as f:
        for arm in ('full_uptake', 'rate'):
            s = summ[(summ.arm == arm) & (summ.scope == scope)]
            cols = {}
            for e, m in MODELS:
                for st in ST:
                    x = s[(s.ensemble == e) & (s.model == m) & (s.split_type == st)].set_index('selector')
                    cols[f"{e.replace('AF2_', '')} {m.replace('omc_', '')} {'seq' if st == 'sequence_cluster' else 'spa'}"] = \
                        [fmt(x.loc[q, 'rec_mean'], x.loc[q, 'rec_sd']) if q in x.index else '' for q in SELS if q != 'iso_closed_sigma']
            t = pd.DataFrame(cols); t.insert(0, 'selector', [q for q in SELS if q != 'iso_closed_sigma'])
            f.write(f'\n**{arm} arm, scope {scope}: recovery % (mean ± SD over 3 replicates) [s4/s5]**\n\n' + md(t) + '\n')

with open(O / 'tables_maxent_pops.md', 'w') as f:
    for scope in ('old_val', 'new_all'):
        s = pm[pm.scope == scope]
        keep = ['f1_wmse', 'f2_mse', 'f3_chi2_pooled', 'f4_logerr', 'f5a_neg_spearman', 'f6_maxabsz', 'f8_pep_norm_mse', 'f9_meancentred_mse',
                'f10_offset_mse', 'f12_ws_wmse_5pct', 'f12_ws_mse_10pct', 'f11_mse_ess_b0.1', 'ref_min_work_scale', 'ref_oracle_max_recovery']
        s = s[s.selector.isin(keep)].copy()
        s['split'] = s.split_type.map({'sequence_cluster': 'seq', 'spatial': 'spa'})
        t = pd.DataFrame({'arm': s.arm, 'ens': s.ensemble.str.replace('AF2_', ''), 'split': s.split, 'selector': s.selector,
                          'recovery': [fmt(a, b) for a, b in zip(s.rec_mean, s.rec_sd)], 'ESS %': s.ess_mean.round(1),
                          'F/PUF1/PUF2/PUF3/U %': [f'{a:.0f}/{b:.0f}/{c:.0f}/{d:.0f}/{e:.0f}' for a, b, c, d, e in
                                                   zip(s.Folded_mean, s.PUF1_mean, s.PUF2_mean, s.PUF3_mean, s.unfolded_mean)],
                          'S selected (rep 0/1/2)': s.S_sel, 'median step': s.step_med.astype(int)})
        t['_o'] = t.selector.map({k: i for i, k in enumerate(keep)})
        t = t.sort_values(['arm', 'ens', 'split', '_o']).drop(columns='_o')
        f.write(f'\n**MaxEnt selections, scope {scope} [s4/s5]**\n\n' + md(t) + '\n')

with open(O / 'tables_verdict.md', 'w') as f:
    for scope in ('old_val', 'old_all', 'new_val', 'new_all'):
        t = V[V.scope == scope].copy()
        t['_o'] = t.selector.map({k: i for i, k in enumerate(SELS)}); t = t.sort_values(['iso_pass', 'rec_mean'], ascending=[False, False])
        t = pd.DataFrame({'selector': t.selector, 'ISO': np.where(t.iso_pass, 'PASS', 'fail'), 'MoPrP mean rec': t.rec_mean.round(1),
                          'full-uptake': t.rec_full_uptake.round(1), 'rate': t.rec_rate.round(1), 'MaxEnt': t.rec_maxent.round(1),
                          'OMC': t.rec_omc.round(1), 'mean rep SD': t.cell_sd_mean.round(1), 'cells ≥80 (/24)': t.n_cells_ge80,
                          'reps ≥80 (/72)': t.n_reps_ge80, 'Folded-majority reps': (100 * t.folded_major_frac).round(0).astype(int).astype(str) + '%',
                          'Folded %': t.folded_mean.round(1), 'PUF3+U %': t.puf3_u_mean.round(1), 'ESS %': t.ess_mean.round(1)})
        f.write(f'\n**Verdict, scope {scope} (24 MoPrP cells = 2 arms × 6 ensemble/model × 2 split types; 72 replicate selections) [s5]**\n\n' + md(t) + '\n')

# spearman table
cols = [c for c in sp.columns if c.startswith('f') or c in ('work_scale', 'val_closed_sigma_mse')]
g = sp.groupby(['arm', 'scope'])[cols].mean().T
with open(O / 'tables_spearman.md', 'w') as f:
    t = g.round(2).reset_index().rename(columns={'index': 'score'})
    t.columns = ['score'] + [f'{a}/{b}' for a, b in g.columns]
    f.write(md(t))
    me = sp[sp.model == 'maxent'].groupby(['arm', 'scope', 'ensemble'])[['f1_wmse', 'f2_mse', 'f4_logerr', 'f9_meancentred_mse', 'f10_offset_mse', 'f5a_neg_spearman', 'work_scale']].mean().round(2).reset_index()
    f.write('\n\n**MaxEnt pools only [s4]**\n\n' + md(me))

# ---- figures
plt.rcParams.update({'font.size': 8})
FS = [s for s in SELS if s != 'iso_closed_sigma']
for arm in ('full_uptake', 'rate'):
    for scope in ('old_val', 'new_all', 'old_all', 'new_val'):
        fig, axs = plt.subplots(6, 1, figsize=(13, 17), sharex=True)
        s = summ[(summ.arm == arm) & (summ.scope == scope)]
        x = np.arange(len(FS))
        for ax, (e, m) in zip(axs, MODELS):
            for j, (st, col) in enumerate(zip(ST, ('#0072B2', '#D55E00'))):
                q = s[(s.ensemble == e) & (s.model == m) & (s.split_type == st)].set_index('selector').reindex(FS)
                ax.bar(x + (j - .5) * .38, q.rec_mean, .38, yerr=q.rec_sd, capsize=2, color=col, label=st, alpha=.85)
            ax.axhline(S.PRIOR[e], color='k', ls='--', lw=1, label=f'uniform prior {S.PRIOR[e]}')
            ax.axhline(S.CAP, color='g', ls=':', lw=1.2, label=f'all-Folded cap {S.CAP}')
            ax.axhline(80, color='r', ls='-', lw=.6, alpha=.5)
            ax.set_ylim(0, 105); ax.set_ylabel('recovery %'); ax.set_title(f'{e} · {m}', loc='left')
            ax.grid(axis='y', alpha=.2)
        axs[0].legend(ncol=5, fontsize=7, loc='upper left')
        axs[-1].set_xticks(x); axs[-1].set_xticklabels(FS, rotation=60, ha='right')
        fig.suptitle(f'MoPrP {arm} · selection scope {scope} · recovery of selected model (mean ± SD, 3 replicates) [s5]')
        fig.tight_layout(); fig.savefig(O / f'F_{arm}_{scope}.png', dpi=110); plt.close(fig)

fig, axs = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
s = summ[(summ.arm == 'iso') & (summ.scope == 'val')]
IS = SELS
x = np.arange(len(IS))
for ax, (e, m) in zip(axs, [('ISO_BI', 'maxent'), ('ISO_TRI', 'maxent'), ('ISO_TRI', 'omc_pyrosetta'), ('ISO_TRI', 'omc_rmsd'), ('ISO_TRI', 'omc_work_scale')]):
    for j, (st, col) in enumerate(zip(ST, ('#0072B2', '#D55E00'))):
        q = s[(s.ensemble == e) & (s.model == m) & (s.split_type == st)].set_index('selector').reindex(IS)
        ax.bar(x + (j - .5) * .38, q.rec_mean, .38, yerr=q.rec_sd, capsize=2, color=col, label=st, alpha=.85)
    ax.axhline(62.6 if e == 'ISO_BI' else 31.1, color='k', ls='--', lw=1, label='uniform prior')
    ax.set_ylim(0, 105); ax.set_title(f'{e} · {m}', loc='left'); ax.set_ylabel('recovery %'); ax.grid(axis='y', alpha=.2)
axs[0].legend(fontsize=7); axs[-1].set_xticks(x); axs[-1].set_xticklabels(IS, rotation=60, ha='right')
fig.suptitle('ISO (ground truth 40/60) · validation-scope selection · recovery of selected model [s5]'); fig.tight_layout()
fig.savefig(O / 'F_iso_val.png', dpi=110); plt.close(fig)

# spearman heatmap
cols2 = [c for c in cols if c != 'val_closed_sigma_mse']
G = sp.groupby(['arm', 'scope'])[cols2].mean()
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(G.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
ax.set_yticks(range(len(G))); ax.set_yticklabels([f'{a}/{b}' for a, b in G.index]); ax.set_xticks(range(len(cols2))); ax.set_xticklabels(cols2, rotation=60, ha='right')
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        ax.text(j, i, f'{G.values[i, j]:.2f}', ha='center', va='center', fontsize=6)
plt.colorbar(im, label='mean within-replicate Spearman(−score, recovery)'); ax.set_title('Is a better score a higher recovery? (+ = yes) [s4/s5]')
fig.tight_layout(); fig.savefig(O / 'F_spearman.png', dpi=110); plt.close(fig)
print(V[V.scope == 'old_val'].sort_values('rec_mean', ascending=False).round(1).to_string())
print(chk.round(1).to_string())
