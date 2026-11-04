"""[o6] figures from o5 summaries.
F1_T1_gamma.png   : T1 (self-tuning L_st vs sidecar kNN20 L_sym), rule-D recovery and ESS vs gamma (+ pure OMC), ISO BI/TRI, both sigma
                    (3 seeds x 6 folds; mean ± SD), gamma = 0 = MaxEnt.
F2_iso_heat.png   : main grid, ISO rule-D recovery per metric x h (rows) x variant (cols), seed 0, per system x sigma; colour =
                    paired difference vs gamma = 0 (diverging blue (better) / red (worse), grey 0), text = recovery and #>=95 on BI.
F3_moprp_heat.png : same for MoPrP (6 folds), text = recovery and #ESS>=50.
-> F1_T1_gamma.png, F2_iso_heat.png, F3_moprp_heat.png"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import i0_common as C

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e4e3df'
DIV = LinearSegmentedColormap.from_list('div', ['#c0392b', '#e34948', '#f0efec', '#2a78d6', '#1c5aa6'])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})
SI = pd.read_csv(C.OUT + 'o5_summary_iso.csv'); SM = pd.read_csv(C.OUT + 'o5_summary_moprp.csv')

# ---------------------------------------------------------------- F1 ----
XS = [0, 0.1, 0.3, 1, 3]; XL = ['0\n(MaxEnt)', '0.1', '0.3', '1', '3', 'pure\nOMC']
fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
v = SI[(SI.seeds == 'seeds012') & (SI.rule == 'D_flags_then_F') & (SI.scoring == 'lam0')]
for i, sysn in enumerate(['ISO_BI', 'ISO_TRI']):
    for kern, col, lab in (('Lst', BLUE, 'self-tuning L_st'), ('knnLsym', ORANGE, 'sidecar kNN20 L_sym (unweighted)')):
        for sg, ls, mk in ((0.01, '-', 'o'), (0.02, '--', 's')):
            b = v[(v.system == sysn) & (v.sigma == sg) & (v.kernel == 'none')].iloc[0]
            k = v[(v.system == sysn) & (v.sigma == sg) & (v.kernel == kern)]
            kl = k[k.variant == 'kl'].sort_values('gamma'); po = k[k.variant == 'pomc']
            ys = [b.rec] + list(kl.rec); es = [b.ESS] + list(kl.ESS); sd = [b.rec_sd] + list(kl.rec_sd)
            x = np.arange(len(ys))
            ax[i, 0].errorbar(x + (0.06 if kern == 'knnLsym' else -0.06), ys, yerr=sd, color=col, ls=ls, marker=mk, ms=6, lw=2, capsize=2,
                              label=f'{lab}, σ={sg}')
            ax[i, 1].plot(x, 100 * np.array(es), color=col, ls=ls, marker=mk, ms=6, lw=2)
            if len(po):
                p = po.iloc[0]
                ax[i, 0].errorbar([5 + (0.06 if kern == 'knnLsym' else -0.06)], [p.rec], yerr=[p.rec_sd], color=col, marker=mk, ms=7, capsize=2, ls='none', mfc='white')
                ax[i, 1].plot([5], [100 * p.ESS], color=col, marker=mk, ms=7, ls='none', mfc='white')
    if sysn == 'ISO_BI': ax[i, 0].axhline(95, color=MUTED, lw=1, ls=':'); ax[i, 0].text(5.3, 95.3, 'BI pass 95', color=MUTED, ha='right', fontsize=8)
    ax[i, 1].axhline(50, color=MUTED, lw=1, ls=':')
    for j in range(2):
        ax[i, j].set_xticks(range(6)); ax[i, j].set_xticklabels(XL)
    ax[i, 0].set_ylabel(f'{sysn}\nrule-D recovery (%)'); ax[i, 1].set_ylabel('rule-D ESS (%)')
ax[1, 0].set_xlabel('γ (OMC term = γ · KL/S at the MaxEnt pick)'); ax[1, 1].set_xlabel('γ')
ax[0, 0].legend(fontsize=7, loc='lower left', frameon=False)
fig.suptitle('T1 · self-tuning vs sidecar kNN OMC on ISO (rule D, Σ = σ²I; 3 seeds × 6 folds; open marker = pure OMC)', color=INK)
fig.savefig(C.OUT + 'F1_T1_gamma.png', dpi=150)


# ---------------------------------------------------------------- F2/F3 ----
def heat(ax, d, rows, cols, title, annot):
    Mv = np.full((len(rows), len(cols)), np.nan); Mt = [['' for _ in cols] for _ in rows]
    for i, (rk, rf) in enumerate(rows):
        for j, (cl, cv, cg, cf) in enumerate(cols):
            r = d[(d.kernel == rk) & (d.variant == cv) & (d.gamma == cg) & (d.form == (cf or rf))]
            if len(r): r = r.iloc[0]; Mv[i, j] = r.d_mean; Mt[i][j] = annot(r)
    lim = 8.0                                   # colour clipped at ±8 points (text carries the value)
    im = ax.imshow(Mv, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    for i in range(len(rows)):
        for j in range(len(cols)):
            if Mt[i][j]: ax.text(j, i, Mt[i][j], ha='center', va='center', fontsize=6.5, color=INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0].replace('work_scale', 'WS').replace('pyrosetta', 'PyR').replace('dihedral_rmsd', 'dihRMS')
                                                          .replace('rmsd', 'RMSD') for r in rows], fontsize=7)
    ax.set_title(title, fontsize=9, color=INK); ax.grid(False)
    return im


cols_iso = [('γ0.1', 'kl', 0.1, 'lap'), ('γ0.3', 'kl', 0.3, 'lap'), ('γ1', 'kl', 1.0, 'lap'), ('γ3', 'kl', 3.0, 'lap'), ('pure', 'pomc', 1.0, 'lap'),
            ('L_sym γ1', 'kl', 1.0, 'sym')]
fig, axs = plt.subplots(1, 4, figsize=(17, 6.5), constrained_layout=True)
v = SI[(SI.seeds == 'seed0') & (SI.rule == 'D_flags_then_F') & (SI.scoring == 'lam0')]
for a, (sysn, sg) in zip(axs, [('ISO_BI', 0.01), ('ISO_BI', 0.02), ('ISO_TRI', 0.01), ('ISO_TRI', 0.02)]):
    d = v[(v.system == sysn) & (v.sigma == sg)]; b = d[d.kernel == 'none'].iloc[0]
    rows = [(k, 'lap') for k in d[~d.kernel.isin(['none', 'Lst', 'knnLsym'])].kernel.unique()]
    ann = (lambda r: f'{r.rec:.1f}\n{int(r.n_ge95)}/6') if sysn == 'ISO_BI' else (lambda r: f'{r.rec:.1f}\nU{100 * r.unassigned:.0f}')
    im = heat(a, d, rows, cols_iso, f'{sysn} σ={sg} · MaxEnt {b.rec:.1f}' + (f' ({int(b.n_ge95)}/6 ≥95)' if sysn == 'ISO_BI' else ''), ann)
fig.colorbar(im, ax=axs, shrink=0.6, extend='both', label='Δ rule-D recovery vs γ = 0 (paired, points; clipped ±8)')
fig.suptitle('Scalar-distance all-pairs OMC (sidecar Gaussian, median-scaled) on ISO · rule D, Σ = σ²I, seed 0 × 6 folds · '
             'text: recovery / BI #≥95 or TRI unassigned %', color=INK)
fig.savefig(C.OUT + 'F2_iso_heat.png', dpi=150)

cols_mo = cols_iso + [('orig γ1', 'kl', 1.0, 'orig')]
fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
v = SM[SM.rule == 'D_flags_then_F']
for a, ens in zip(axs, C.D.ENSS):
    d = v[v.ensemble == ens]; b = d[d.kernel == 'none'].iloc[0]
    rows = [(k, 'lap') for k in d[d.kernel != 'none'].kernel.unique()]
    im = heat(a, d, rows, cols_mo, f'{ens} · MaxEnt (γ=0) {b.rec:.2f}, ESS≥50 {int(b.n_ESS50)}/6',
              lambda r: f'{r.rec:.1f}\nE{int(r.n_ESS50)}')
fig.colorbar(im, ax=axs, shrink=0.6, extend='both', label='Δ rule-D recovery vs γ = 0 (paired, points; clipped ±8)')
fig.suptitle('Scalar-distance all-pairs OMC on MoPrP (b′ + rate + KL to ΔG p0) · rule D · text: recovery / #folds ESS≥50%', color=INK)
fig.savefig(C.OUT + 'F3_moprp_heat.png', dpi=150)
print('figures written')
