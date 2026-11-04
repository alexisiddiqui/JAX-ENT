"""[q4] Figures: per system/ensemble x sigma variant, rows = recovery, ESS, held-out chi2/cell, n_flags, F_rel vs S;
columns = the three block splits (both folds overlaid); each rule's pick marked. Grey band = ESS < 10% (outside sweep).
-> F_<system>_<ensemble>_<sigma>.png"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import q0_common as Q

MK = {'A_flags': ('^', '#d95f02'), 'B_F': ('s', '#1b9e77'), 'C_minchi2': ('o', '#7570b3'), 'C_1SE': ('D', '#e7298a'), 'D_flags_then_F': ('*', '#000000')}
SEL = pd.read_csv(Q.OUT + 'q3_selected.csv', dtype={'sigma': str})
SMIN = 1e-4


def xs(S):
    return np.where(np.asarray(S) == 0, SMIN, S)


def panel(df, sel, title, fn):
    splits = list(dict.fromkeys(df.split))
    metrics = [('rec', 'recovery (%)'), ('ESS', 'ESS (Kish/N)'), ('ho_chi2_cell', 'held-out χ²/cell'), ('ho_n_flags', 'held-out flags (|z|>t_n)'),
               ('F_rel', 'F − F(p0)')]
    df = df.copy(); df['F_rel'] = df.F - df.groupby(['split', 'heldout'] + (['seed'] if 'seed' in df else [])).F.transform(lambda x: x.iloc[np.argmin(df.loc[x.index, 'S'].values)])
    fig, ax = plt.subplots(len(metrics), len(splits), figsize=(5.2 * len(splits), 2.6 * len(metrics)), sharex=True, squeeze=False)
    for j, sp in enumerate(splits):
        for k, (held, ls) in enumerate(zip(list(dict.fromkeys(df[df.split == sp].heldout)), ['-', '--'])):
            g = df[(df.split == sp) & (df.heldout == held)].sort_values('S')
            s = sel[(sel.split == sp) & (sel.heldout == held)]
            g0 = g.copy(); g0['F_rel'] = g.F - g.F.iloc[0]
            for i, (mcol, lab) in enumerate(metrics):
                ax[i, j].plot(xs(g0.S), g0[mcol], ls, color='C%d' % k, lw=1.4, label=f'held-out {held}')
                for rule, (mk, col) in MK.items():
                    r = s[s.rule == rule]
                    if len(r):
                        ii = g0.index[np.argmin(np.abs(g0.S.values - r.S.values[0]))]
                        ax[i, j].plot(xs(g0.loc[ii, 'S']), g0.loc[ii, mcol], mk, color=col, ms=9 if mk == '*' else 6, mfc='none' if k else col,
                                      label=rule if (i == 0 and k == 0) else None)
                ax[i, j].set_ylabel(lab, fontsize=8)
            lo = g0[g0.ESS < 0.10].S.min()
            if np.isfinite(lo):
                for i in range(len(metrics)): ax[i, j].axvspan(lo, g0.S.max() * 1.5, color='0.9', zorder=0)
        ax[0, j].set_title(sp, fontsize=10); ax[-1, j].set_xscale('log'); ax[-1, j].set_xlabel(f'S (S=0 / p0 plotted at {SMIN:g})')
        ax[1, j].set_yscale('log')
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title + '   (filled = fold 1, open = fold 2; grey = ESS < 10% in at least one fold; candidates = points with ESS ≥ 10%)', fontsize=10)
    fig.tight_layout(); fig.savefig(Q.OUT + fn, dpi=110); plt.close(fig)
    print('wrote', fn)


d1 = pd.read_csv(Q.OUT + 'q1_sweep.csv')
for ens in Q.D.ENSS:
    for sv in ('floor', 'unfloored'):
        panel(d1[(d1.ensemble == ens) & (d1.sigma == sv)], SEL[(SEL.ensemble == ens) & (SEL.sigma == sv)],
              f'MoPrP {ens}, b′ rate, p0 = dg T2 A|S11 MaxEnt, σ {sv}', f'F_MoPrP_{ens}_{sv}.png')
for which in ('bi', 'tri'):
    p = Q.OUT + f'q2_sweep_{which}.csv'
    if not os.path.exists(p): continue
    d2 = pd.read_csv(p, dtype={'sigma': str})
    for sg in ('0.01', '0.02'):
        g = d2[(d2.sigma == sg) & (d2.seed == 0)]
        s = SEL[(SEL.system == f'ISO_{which.upper()}') & (SEL.sigma == sg) & (SEL.seed == 0)]
        panel(g, s, f'ISO_{which.upper()} noise σ={sg} (seed 0), p0 uniform', f'F_ISO_{which.upper()}_s{sg}.png')
