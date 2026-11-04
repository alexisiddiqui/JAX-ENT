"""[o5] selection: rules A / B / D (+ D_ess50, refs p0 / oracle; stk s0.select, candidates ESS >= 10% U {S = 0}) on
  ISO : o2 (gamma = 0) + o3 (OMC) sweeps, lam = 0 scoring (primary); T1 graphs also under lam_geo diag (stk union C_shape).
  MoPrP: stk s2 gamma = 0 sweep (upt, lam = 0) + o4 OMC sweeps, lam = 0 scoring; ESS >= 50% as acceptance check on rule D.
Paired comparison vs gamma = 0 on identical folds (ISO main grid: seed 0 only; T1: 3 seeds).
-> o5_selected_iso.csv, o5_summary_iso.csv, o5_selected_moprp.csv, o5_summary_moprp.csv, o5.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import i0_common as C
Q, D, S0 = C.Q, C.D, C.S0
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 2000); pd.set_option('display.max_columns', 40)
RULES = ['A_flags', 'B_F', 'D_flags_then_F', 'D_ess50', 'ref_oracle']

def summ(df, keys):
    grp = df.groupby(keys, sort=False)
    out = grp.agg(n=('rec', 'size'), S_med=('S', 'median'), ESS=('ESS', 'mean'), rec=('rec', 'mean'), rec_sd=('rec', 'std'), rec_min=('rec', 'min'),
                  flags=('n_flags', 'mean'), **({'open': ('pop_open', 'mean'), 'unassigned': ('pop_unassigned', 'mean')} if 'pop_open' in df else {}),
                  **({f'pop_{s}': (f'pop_{s}', 'mean') for s in ('Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded')} if 'pop_Folded' in df else {})).reset_index()
    out['n_ge95'] = grp.rec.apply(lambda x: int((x >= 95).sum())).values
    out['n_ESS50'] = grp.ESS.apply(lambda x: int((x >= 0.5).sum())).values
    out['n_p0'] = grp.S.apply(lambda x: int((x == 0).sum())).values
    return out



PARTS_RUN = sys.argv[1:] or ['iso', 'moprp']
if 'iso' in PARTS_RUN:
    # ------------------------------------------------------------------ ISO ----
    o2 = pd.read_csv(C.OUT + 'o2_sweep.csv'); o2 = o2.assign(kernel='none', form='-', variant='kl', gamma=0.0, c=np.nan)
    o3 = pd.read_csv(C.OUT + 'o3_sweep.csv')
    if os.path.exists(C.OUT + 'o3b_sweep.csv'):        # follow-up (3 seeds) of RMSD L_sym kernels
        o3b = pd.read_csv(C.OUT + 'o3b_sweep.csv')
        KK = ['system', 'kernel', 'form', 'variant', 'gamma', 'sigma', 'seed', 'split', 'heldout', 'S']
        dup = o3.merge(o3b, on=KK, suffixes=('', '_b'))
        print(f'[check o3b vs o3] {len(dup)} overlapping rows; max |d rec| {np.abs(dup.rec - dup.rec_b).max():.2e}, '
              f'max |d ESS| {np.abs(dup.ESS - dup.ESS_b).max():.2e}', flush=True)
        o3b = o3b.merge(o3[KK], on=KK, how='left', indicator=True); o3b = o3b[o3b._merge == 'left_only'].drop(columns='_merge')
        o3 = pd.concat([o3, o3b], ignore_index=True)
    iso = pd.concat([o2, o3], ignore_index=True)
    FK = ['system', 'kernel', 'form', 'variant', 'gamma', 'sigma', 'seed', 'split', 'heldout']
    sel = []
    for key, g in iso.groupby(FK, sort=False):
        for scoring, (nf, fc) in (('lam0', ('n_flags', 'F')), ('lam_geo', ('n_flags_lg', 'F_lg'))):
            if scoring == 'lam_geo' and (g[nf].isna().any() if nf in g else True): continue
            gg = g.copy(); gg['n_flags'] = gg[nf]; gg['F'] = gg[fc]
            for r in C.select(gg): r.update(scoring=scoring); sel.append(r)
    SI = pd.DataFrame(sel)
    SI.to_csv(C.OUT + 'o5_selected_iso.csv', index=False)


    SUM = []
    for seeds, lab in (([0], 'seed0'), ([0, 1, 2], 'seeds012')):
        d = SI[SI.seed.isin(seeds) & SI.rule.isin(RULES)]
        s = summ(d, ['system', 'sigma', 'scoring', 'kernel', 'form', 'variant', 'gamma', 'rule']); s['seeds'] = lab
        s = s[(s.seeds == 'seed0') | (s.n == 18)]          # 3-seed rows only where all 18 fold-runs exist
        SUM.append(s)
    SUMI = pd.concat(SUM, ignore_index=True)

    # paired delta vs gamma = 0 (rule D, lam0) on identical folds
    base = SI[(SI.kernel == 'none') & (SI.rule == 'D_flags_then_F') & (SI.scoring == 'lam0')].set_index(['system', 'sigma', 'seed', 'split', 'heldout']).rec
    dd = SI[(SI.rule == 'D_flags_then_F') & (SI.scoring == 'lam0') & (SI.kernel != 'none')].copy()
    dd['d_rec'] = dd.rec.values - base.loc[list(zip(dd.system, dd.sigma, dd.seed, dd.split, dd.heldout))].values
    dd = pd.concat([dd[dd.seed == 0].assign(seeds='seed0'), dd.assign(seeds='seeds012')], ignore_index=True)
    pdel = dd.groupby(['system', 'sigma', 'seeds', 'kernel', 'form', 'variant', 'gamma'], sort=False).agg(
        d_mean=('d_rec', 'mean'), d_se=('d_rec', lambda x: x.std() / np.sqrt(len(x))), n_better=('d_rec', lambda x: int((x > 0.5).sum())),
        n_worse=('d_rec', lambda x: int((x < -0.5).sum())), n_pair=('d_rec', 'size')).reset_index()
    SUMI = SUMI.merge(pdel, on=['system', 'sigma', 'seeds', 'kernel', 'form', 'variant', 'gamma'], how='left')
    SUMI.loc[SUMI.rule != 'D_flags_then_F', ['d_mean', 'd_se', 'n_better', 'n_worse']] = np.nan
    SUMI.to_csv(C.OUT + 'o5_summary_iso.csv', index=False)

    print('=== ISO rule D (lam = 0), seed 0 (main grid) — rec mean±sd (min) · ESS · #>=95 · open/unassigned · paired d vs gamma0 ===')
    v = SUMI[(SUMI.rule == 'D_flags_then_F') & (SUMI.scoring == 'lam0') & (SUMI.seeds == 'seed0')]
    for _, r in v.iterrows():
        print(f'{r.system:7s} s={r.sigma:<5g} {r.kernel:18s} {r.form:4s} {r.variant:4s} g={r.gamma:<4g} rec {r.rec:6.2f}±{r.rec_sd:4.2f} ({r.rec_min:5.1f}) '
              f'ESS {100 * r.ESS:3.0f}% ge95 {r.n_ge95}/{r.n} open {100 * r.open:5.1f} unas {100 * r.unassigned:5.1f} p0 {r.n_p0} '
              f'd {r.d_mean:+.2f}±{r.d_se:.2f} (+{r.n_better}/-{r.n_worse})' if r.kernel != 'none' else
              f'{r.system:7s} s={r.sigma:<5g} {"gamma=0 (MaxEnt)":18s}                rec {r.rec:6.2f}±{r.rec_sd:4.2f} ({r.rec_min:5.1f}) ESS {100 * r.ESS:3.0f}% '
              f'ge95 {r.n_ge95}/{r.n} open {100 * r.open:5.1f} unas {100 * r.unassigned:5.1f}')
    print('\n=== ISO T1 (3 seeds) rule D / A / B, lam0 and lam_geo ===')
    v = SUMI[(SUMI.seeds == 'seeds012') & SUMI.rule.isin(['A_flags', 'B_F', 'D_flags_then_F', 'D_ess50'])]
    for _, r in v.iterrows():
        print(f'{r.system:7s} s={r.sigma:<5g} {r.scoring:7s} {r.kernel:8s} {r.variant:4s} g={r.gamma:<4g} {r.rule:15s} rec {r.rec:6.2f}±{r.rec_sd:4.2f} ({r.rec_min:5.1f}) '
              f'ESS {100 * r.ESS:3.0f}% ge95 {r.n_ge95}/{r.n} open {100 * r.open:5.1f} unas {100 * r.unassigned:5.1f} p0 {r.n_p0}'
              + (f' d {r.d_mean:+.2f}±{r.d_se:.2f}' if r.rule == 'D_flags_then_F' and r.scoring == 'lam0' and r.kernel != 'none' else ''))
    print('\n[ISO solver] max stat (chi2 units) over all o3 rows: %.2e; at rule picks: %.2e; non-converged rows %d / %d' % (
        o3.stat.max(), SI[SI.kernel != 'none'].stat.max(), int((~o3.conv.astype(bool)).sum()), len(o3)))


if 'moprp' in PARTS_RUN:
    # ---------------------------------------------------------------- MoPrP ----
    STK = os.path.join(C.RG, 'stk') + '/'
    s2 = pd.read_csv(STK + 's2_sweep.csv'); s2 = s2[(s2['mode'] == 'upt') & (s2.lam_key.astype(str) == '0')].copy()
    s2 = s2.rename(columns={'F_diag': 'F'}).assign(kernel='none', form='-', variant='kl', gamma=0.0, c=np.nan)
    o4 = pd.read_csv(C.OUT + 'o4_sweep.csv')
    if os.path.exists(C.OUT + 'o4b_sweep.csv'):
        o4 = pd.concat([o4, pd.read_csv(C.OUT + 'o4b_sweep.csv')], ignore_index=True)
    cols = [c for c in o4.columns if c in s2.columns]
    mo = pd.concat([s2[cols], o4[cols]], ignore_index=True)
    sel = []
    for key, g in mo.groupby(['ensemble', 'kernel', 'form', 'variant', 'gamma', 'split', 'heldout'], sort=False):
        for r in C.select(g): sel.append(r)
    SM = pd.DataFrame(sel); SM.to_csv(C.OUT + 'o5_selected_moprp.csv', index=False)
    SUMM = summ(SM[SM.rule.isin(RULES)], ['ensemble', 'kernel', 'form', 'variant', 'gamma', 'rule'])
    b = SM[(SM.kernel == 'none') & (SM.rule == 'D_flags_then_F')].set_index(['ensemble', 'split', 'heldout']).rec
    dm = SM[(SM.rule == 'D_flags_then_F') & (SM.kernel != 'none')].copy()
    dm['d_rec'] = dm.rec.values - b.loc[list(zip(dm.ensemble, dm.split, dm.heldout))].values
    pdm = dm.groupby(['ensemble', 'kernel', 'form', 'variant', 'gamma'], sort=False).agg(
        d_mean=('d_rec', 'mean'), d_se=('d_rec', lambda x: x.std() / np.sqrt(len(x))), n_better=('d_rec', lambda x: int((x > 0.5).sum())),
        n_worse=('d_rec', lambda x: int((x < -0.5).sum()))).reset_index()
    SUMM = SUMM.merge(pdm, on=['ensemble', 'kernel', 'form', 'variant', 'gamma'], how='left')
    SUMM.loc[SUMM.rule != 'D_flags_then_F', ['d_mean', 'd_se', 'n_better', 'n_worse']] = np.nan
    SUMM.to_csv(C.OUT + 'o5_summary_moprp.csv', index=False)
    print('\n=== MoPrP rule D (lam = 0): rec mean±sd (min) · ESS · #ESS>=50 of 6 · F/PUF1/PUF2/U · paired d vs gamma0 ===')
    for _, r in SUMM[SUMM.rule == 'D_flags_then_F'].iterrows():
        print(f'{r.ensemble:12s} {r.kernel:20s} {r.form:4s} {r.variant:4s} g={r.gamma:<4g} rec {r.rec:6.2f}±{r.rec_sd:4.2f} ({r.rec_min:5.1f}) ESS {100 * r.ESS:3.0f}% '
              f'E50 {r.n_ESS50}/6 F/P1/P2/U {100 * r.pop_Folded:.1f}/{100 * r.pop_PUF1:.2f}/{100 * r.pop_PUF2:.2f}/{100 * r.pop_unfolded:.2f} p0 {r.n_p0}'
              + (f' d {r.d_mean:+.2f}±{r.d_se:.2f} (+{int(r.n_better)}/-{int(r.n_worse)})' if r.kernel != 'none' else ''))
    print('\n[MoPrP solver] max stat over o4 rows %.2e; at rule picks %.2e' % (o4.stat.max(), SM[SM.kernel != 'none'].stat.max()))
