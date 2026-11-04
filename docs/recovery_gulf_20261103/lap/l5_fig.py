"""[l5] Figures. F_p0_vs_tau.png: p0_tau recovery / ESS / PUF2 weight vs tau (family A, primary graph, both ensembles,
L and L_sym; solid = production p0 ('use'), dashed = soft S=1 sensitivity). Source l2_t2.csv.
F_ruleD_vs_tau.png: rule-D pick recovery and ESS vs tau (mean +- SD over 6 folds), gamma = 0 (L, L_sym) and gamma = 0.1, 1
(OMC term, L). Source l4_summary.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
X0 = 0.03                      # where tau = 0 is drawn on the log axis
ENS = ['AF2_MSAss', 'AF2_filtered']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e4e4e4',
                     'grid.linewidth': 0.6})


def xt(t): return np.where(np.asarray(t) == 0, X0, t)


def fix_x(ax):
    ax.set_xscale('log'); ticks = [X0, 0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(ticks); ax.set_xticklabels(['0', '0.1', '0.3', '1', '3', '10', '30']); ax.set_xlabel('τ')


d = pd.read_csv(OUT + 'l2_t2.csv'); d = d[(d.fam == 'A') & (d.graph == 'primary')]
fig, axs = plt.subplots(3, 2, figsize=(9, 8.2), sharex=True)
for j, ens in enumerate(ENS):
    for lap, col in (('L', C1), ('Lsym', C2)):
        u = d[(d.ensemble == ens) & (d.lap == lap) & d.use].sort_values('tau')
        s1 = d[(d.ensemble == ens) & (d.lap == lap) & (d.solver == 'soft1')].sort_values('tau')
        lab = 'L = D − A' if lap == 'L' else 'L_sym'
        for i, (col_, f) in enumerate((('rec', 1), ('ESS', 100), ('pop_PUF2', 100))):
            y = u[col_] * f; y1 = s1[col_] * f
            if col_ == 'pop_PUF2': y = np.maximum(y, 1e-10); y1 = np.maximum(y1, 1e-10)
            axs[i, j].plot(xt(u.tau), y, '-o', color=col, lw=2, ms=5, label=f'{lab}, production p0')
            axs[i, j].plot(xt(s1.tau), y1, '--', color=col, lw=1.2, label=f'{lab}, soft S=1')
    axs[0, j].axhline(91.7 if ens == 'AF2_MSAss' else 96.1, color='#888', lw=0.8, ls=':')
    axs[0, j].set_title(ens.replace('AF2_', 'AF2 '))
    axs[2, j].axhline(0.517, color='#888', lw=0.8, ls=':'); axs[2, j].text(X0, 0.6, 'target 0.52%', color='#666', fontsize=7)
    axs[2, j].set_yscale('log'); axs[2, j].set_ylim(5e-11, 50)
    for i in range(3): fix_x(axs[i, j])
axs[0, 0].set_ylabel('p0 recovery %'); axs[1, 0].set_ylabel('p0 ESS %'); axs[2, 0].set_ylabel('p0 weight on PUF2 frames %\n(floored at 1e-10)')
axs[0, 0].legend(fontsize=7, loc='lower left')
fig.suptitle('Smoothed-openness reference p0_τ (family A). Dotted: τ=0 recovery; PUF2 target', fontsize=10)
fig.tight_layout(); fig.savefig(OUT + 'F_p0_vs_tau.png', dpi=150); plt.close(fig)

s = pd.read_csv(OUT + 'l4_summary.csv')
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
for j, ens in enumerate(ENS):
    variants = [('base', 'L', 0.0, C1, 'γ=0, L', -0.04), ('base', 'Lsym', 0.0, C2, 'γ=0, L_sym', -0.02),
                ('omc', 'L', 0.1, C3, 'γ=0.1 (OMC), L', 0.02), ('omc', 'L', 1.0, C4, 'γ=1 (OMC), L', 0.04)]
    for part, lap, gam, col, lab, dx in variants:
        x = s[(s.ensemble == ens) & (s.part == part) & (s.lap == lap) & (s.gamma == gam) & (s.p0req == 'use') & (s.rule == 'D_flags_then_F')]
        if part == 'base' and lap == 'Lsym':      # tau=0 is shared
            x = pd.concat([s[(s.ensemble == ens) & (s.part == 'base') & (s.lap == 'L') & (s.tau == 0) & (s.rule == 'D_flags_then_F')], x])
        x = x.sort_values('tau'); xx = xt(x.tau) * (1 + dx * 4)
        axs[0, j].errorbar(xx, x.rec_mean, x.rec_std, fmt='-o', color=col, lw=1.6, ms=5, capsize=2, label=lab)
        axs[1, j].errorbar(xx, 100 * x.ESS_mean, 100 * x.ESS_std, fmt='-o', color=col, lw=1.6, ms=5, capsize=2, label=lab)
    axs[1, j].axhline(50, color='#888', lw=0.8, ls=':'); axs[0, j].set_title(ens.replace('AF2_', 'AF2 '))
    for i in range(2): fix_x(axs[i, j])
axs[0, 0].set_ylabel('rule-D recovery % (mean ± SD, 6 folds)'); axs[1, 0].set_ylabel('rule-D ESS %')
axs[0, 0].legend(fontsize=7, loc='lower left')
fig.suptitle('Rule D (min flags, then min F) on χ² + KL(w‖p0_τ)/S [+ γ wᵀLw]', fontsize=10)
fig.tight_layout(); fig.savefig(OUT + 'F_ruleD_vs_tau.png', dpi=150); plt.close(fig)
print('wrote F_p0_vs_tau.png, F_ruleD_vs_tau.png')
