"""[s7] T4 figures. (a) F_a_neff_vs_k.png: N_eff and lambda_geo vs k per bandwidth x support, per ensemble (lazy walk; MoPrP
'upt' solid, 'delta' dashed); gci2 adopted values marked. (b) F_b_ruleD_vs_lam.png: rule-D recovery and ESS (mean ± SD over 6
folds) vs lam (diag / full, upt) per ensemble, lam = 0 baseline marked; ISO_BI rule-D recovery panel. (c) F_c_ruleD_vs_gamma.png:
rule-D recovery / ESS vs gamma (Sigma at lambda_geo, diag) incl. pure-OMC variants."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s0_common as S0

OUT = S0.OUT
k1 = pd.read_csv(OUT + 's1_kernel.csv'); k1 = k1[k1.lazy & (k1.space != 'gci2-uptake(default BV)')]
GCI2 = {'AF2_MSAss': 0.366, 'AF2_filtered': 0.074}
ENS = ['AF2_MSAss', 'AF2_filtered', 'ISO_BI', 'ISO_TRI']
COL = {('mean', 'union'): '#1f77b4', ('rms', 'union'): '#9ecae1', ('mean', 'dense'): '#d62728', ('rms', 'dense'): '#fc9272'}

# (a)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for j, e in enumerate(ENS):
    for (bw, sup), c in COL.items():
        for F, ls in (('upt', '-'), ('delta', '--'), ('upt(exact)', '-')):
            g = k1[(k1.ensemble == e) & (k1.bw == bw) & (k1.support == sup) & (k1.F == F)].sort_values('k')
            if g.empty: continue
            lab = f'{bw}/{sup}' + (' (delta)' if F == 'delta' else '')
            ax[0, j].plot(g.k, g.N_eff, ls, color=c, marker='o', label=lab)
            ax[1, j].plot(g.k, g.lam_geo, ls, color=c, marker='o', label=lab)
    N = 500 if e.startswith('AF2') else (874 if e == 'ISO_BI' else 2225)
    if e in GCI2:
        ax[1, j].axhline(GCI2[e], color='k', ls=':', label=f'gci2 adopted {GCI2[e]} (uptake space)')
    else:
        ax[1, j].axhline(5 / N, color='k', ls=':', label='5/N (gci2 h3)')
    ax[1, j].axhline(1 / N, color='grey', ls='-.', lw=0.8, label='1/N (independent)')
    for i in range(2): ax[i, j].set_yscale('log'); ax[i, j].set_xticks([10, 20]); ax[i, j].set_xlabel('k (kNN)')
    ax[0, j].set_title(f'{e} (N={N})'); ax[0, j].set_ylabel('N_eff'); ax[1, j].set_ylabel('lambda_geo = tr C_walk / tr C_ens')
    ax[1, j].legend(fontsize=6.5)
fig.suptitle('[s1] self-tuning kernel in L2-lnPF space, lazy walk, alpha=1/2 (ISO_BI union graph: 2 components, g capped at 2N-1)')
fig.tight_layout(); fig.savefig(OUT + 'F_a_neff_vs_k.png', dpi=130); plt.close(fig)

# (b)
s3 = pd.read_csv(OUT + 's3_summary.csv'); LK = ['0', '1/N', 'lg/3', 'lg', '3lg']
s5 = pd.read_csv(OUT + 's5_summary.csv') if os.path.exists(OUT + 's5_summary.csv') else None
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for j, e in enumerate(['AF2_MSAss', 'AF2_filtered']):
    for md, ls in (('upt', '-'), ('delta', ':')):
        for ver, c in (('diag', '#1f77b4'), ('full', '#d62728')):
            for rule, mk, off in (('D_flags_then_F', 'o', 0.0), ('D_ess50', 's', 0.12)):
                g = s3[(s3.ensemble == e) & (s3['mode'] == md) & (s3.version == ver) & (s3.rule == rule)].set_index('lam_key').reindex(LK)
                if ver == 'full': g.loc['0'] = s3[(s3.ensemble == e) & (s3['mode'] == md) & (s3.version == 'diag') & (s3.rule == rule) & (s3.lam_key == '0')].iloc[0]
                x = np.arange(len(LK)) + off + (0.05 if ver == 'full' else 0)
                lab = f'{rule.split("_")[0] if rule != "D_ess50" else "D|ESS50"} {ver} {md}'
                ax[0, j].errorbar(x, g.rec_mean, g.rec_sd, ls=ls, color=c, marker=mk, ms=4, capsize=2, alpha=1 if md == 'upt' else 0.5, label=lab)
                ax[1, j].errorbar(x, 100 * g.ESS_mean, 100 * g.ESS_sd, ls=ls, color=c, marker=mk, ms=4, capsize=2, alpha=1 if md == 'upt' else 0.5, label=lab)
    for i in range(2):
        ax[i, j].axvspan(-0.3, 0.3, color='grey', alpha=0.15); ax[i, j].set_xticks(range(len(LK))); ax[i, j].set_xticklabels(['0 (selgci)', '1/N', 'lg/3', 'lg', '3lg'])
    lgv = s3[(s3.ensemble == e) & (s3['mode'] == 'upt') & (s3.lam_key == 'lg')].lam.iloc[0]
    ax[0, j].set_title(f'{e}: rule D vs lam (lambda_geo upt = {lgv:.3g})'); ax[0, j].set_ylabel('recovery % (mean ± SD, 6 folds)')
    ax[1, j].set_ylabel('ESS %'); ax[1, j].axhline(50, color='k', lw=0.6, ls='--')
    ax[0, j].axhline(95, color='k', lw=0.6, ls='--')
ax[0, 0].legend(fontsize=6.5, ncol=2)
if s5 is not None:
    LK5 = ['0', '1/N', '5/N', 'lg/3', 'lg', '3lg', 'lgD']
    for i, sysn in enumerate(['ISO_BI', 'ISO_TRI']):
        for sg, c in ((0.01, '#2ca02c'), (0.02, '#9467bd')):
            for ver, ls in (('diag', '-'), ('full', '--')):
                g = s5[(s5.system == sysn) & (s5.sigma == sg) & (s5.version == ver) & (s5.rule == 'D_flags_then_F')].set_index('lam_key').reindex(LK5)
                ax[i, 2].errorbar(np.arange(len(LK5)), g.rec_mean, g.rec_sd, ls=ls, color=c, marker='o', ms=4, capsize=2, label=f'sigma={sg} {ver}')
        ax[i, 2].set_xticks(range(len(LK5))); ax[i, 2].set_xticklabels(LK5); ax[i, 2].axvspan(-0.3, 0.3, color='grey', alpha=0.15)
        ax[i, 2].set_title(f'{sysn}: rule D recovery vs lam (18 replicates)'); ax[i, 2].set_ylabel('recovery %')
        if sysn == 'ISO_BI': ax[i, 2].axhline(95, color='k', lw=0.6, ls='--')
    ax[0, 2].legend(fontsize=7)
fig.suptitle('[s3, s5] two-sided Sigma = D_exp + lam C_shape in the selection rules (grey band = lam 0 baseline)')
fig.tight_layout(); fig.savefig(OUT + 'F_b_ruleD_vs_lam.png', dpi=130); plt.close(fig)

# (c)
if os.path.exists(OUT + 's6_summary.csv'):
    s6 = pd.read_csv(OUT + 's6_summary.csv')
    XS = [('kl', 0.0), ('kl', 0.1), ('kl', 0.3), ('kl', 1.0), ('kl', 3.0), ('pomc', 1.0), ('pomcu', 1.0)]
    XL = ['0', '0.1', '0.3', '1', '3', 'pure OMC\n(start p0)', 'pure OMC\n(start unif)']
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for j, e in enumerate(['AF2_MSAss', 'AF2_filtered']):
        for rule, c, mk in (('A_flags', '#ff7f0e', '^'), ('B_F', '#2ca02c', 's'), ('D_flags_then_F', '#1f77b4', 'o'), ('D_ess50', '#9467bd', 'D')):
            for lk, ls in (('lg', '-'), ('0', ':')):
                g = s6[(s6.ensemble == e) & (s6.rule == rule) & (s6.lam_key == lk) & (s6.version == 'diag')].set_index(['variant', 'gamma'])
                y = [g.loc[k] if k in g.index else None for k in XS]
                rec = np.array([r.rec_mean if r is not None else np.nan for r in y]); rsd = np.array([r.rec_sd if r is not None else np.nan for r in y])
                ess = np.array([100 * r.ESS_mean if r is not None else np.nan for r in y]); esd = np.array([100 * r.ESS_sd if r is not None else np.nan for r in y])
                x = np.arange(len(XS)) + (0.06 if lk == '0' else 0)
                ax[0, j].errorbar(x, rec, rsd, ls=ls, color=c, marker=mk, ms=4, capsize=2, label=f'{rule} lam={lk}', alpha=1 if lk == 'lg' else 0.5)
                ax[1, j].errorbar(x, ess, esd, ls=ls, color=c, marker=mk, ms=4, capsize=2, alpha=1 if lk == 'lg' else 0.5)
        for i in range(2): ax[i, j].set_xticks(range(len(XS))); ax[i, j].set_xticklabels(XL, fontsize=8)
        ax[0, j].set_title(f'{e}: picks vs gamma\n(gamma=1 <=> OMC term = KL/S at D pick)', fontsize=10)
        ax[0, j].set_ylabel('recovery %'); ax[1, j].set_ylabel('ESS %'); ax[1, j].set_xlabel('gamma'); ax[1, j].axhline(50, color='k', lw=0.6, ls='--')
    ax[0, 0].legend(fontsize=6.5)
    fig.suptitle('[s6] chi2 + KL/S + gamma w^T L_st w; rules with two-sided z at lambda_geo (solid) and lam = 0 (dotted)')
    fig.tight_layout(); fig.savefig(OUT + 'F_c_ruleD_vs_gamma.png', dpi=130); plt.close(fig)
print('figures written')
