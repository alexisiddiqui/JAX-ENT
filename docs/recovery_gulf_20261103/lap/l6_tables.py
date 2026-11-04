"""[l6] Formatting only: markdown tables for lap_report.md from l1/l2/l4 outputs -> l6_tables.md"""
import os
import numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
o = []
# ---- T1: per-state mean openness (A, primary graph) for MSAss b2/a1 rows and Filtered b2 row
st = pd.read_csv(OUT + 'l1_t1_states.csv'); sm = pd.read_csv(OUT + 'l1_t1_summary.csv')
REG = {'127-132': 'β1', '133-148+149-153': 'α1', '154-167+155-162': 'β2', '182-196+197-204+197-206+205-212+217-223': 'core'}
o.append('## T1a. Per-state mean smoothed openness, family A, primary graph `[l1]`\n')
o.append('Cells: Folded / PUF1 / PUF2 / PUF3 / U (Filtered: Folded / PUF1 / PUF2 / PUF2-like).\n')
o.append('| ensemble | L | τ | β1 | α1 | β2 | core |\n|---|---|---|---|---|---|---|')
for ens in ('AF2_MSAss', 'AF2_filtered'):
    sts = ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded'] if ens == 'AF2_MSAss' else ['Folded', 'PUF1', 'PUF2', 'PUF2-like']
    for lap in ('L', 'Lsym'):
        for tau in (0, 0.1, 0.3, 1, 3, 10, 30):
            if tau == 0 and lap == 'Lsym': continue
            cells = []
            for row, nm in REG.items():
                x = st[(st.ensemble == ens) & (st.graph == 'primary') & (st.lap == lap) & (st.fam == 'A') & (st.tau == tau) & (st.row == row)]
                cells.append(' / '.join(f'{x[x.state == s].mean_o.iloc[0]:.2f}' for s in sts) if len(x) else '– (no open frame)')
            o.append(f'| {ens} | {lap if tau else "binary"} | {tau:g} | ' + ' | '.join(cells) + ' |')
o.append('\n## T1b. Smoothing scale and value distribution, family A rows, primary graph `[l1]`\n')
o.append('n_frac = frames with 1e-3 < o_τ < 1 − 1e-3. Δ = mean |o_τ − o|. "min open / max closed" = smallest o_τ on a raw-open frame / '
         'largest o_τ on a raw-closed frame. Column sum change = relative change of Σ_i o_τ,i.\n')
o.append('| ensemble | L | τ | row (n open) | n_frac | min o_τ (all frames) | min open / max closed | Δ | column-sum change | target |\n|---|---|---|---|---|---|---|---|---|---|')
for ens in ('AF2_MSAss', 'AF2_filtered'):
    for lap in ('L', 'Lsym'):
        for tau in (0.1, 1, 10):
            x = sm[(sm.ensemble == ens) & (sm.graph == 'primary') & (sm.lap == lap) & (sm.fam == 'A') & (sm.tau == tau)]
            for _, r in x.iterrows():
                if r.n_open_raw == 0: continue
                o.append(f'| {ens} | {lap} | {tau:g} | {REG[r.row]} ({r.n_open_raw}) | {r.n_fractional} | {r["min"]:.4f} | {r.min_on_open:.2f} / {r.max_on_closed:.2f} | '
                         f'{r.mean_abs_change:.3f} | {r.colsum_rel_change:+.1e} | {r.target_geo:.4f} |')
sp = pd.read_csv(OUT + 'l1_spectrum.csv')
o.append('\n## T1c. Spectral scale: λ_2, median λ, λ_max and the smoother trace (effective number of modes kept, of 500) `[l1]`\n')
o.append('Filter on mode k is 1/(1+τλ_k).\n')
o.append('| ensemble | graph | L | mean degree | λ_2 | λ_med | λ_max | 1/(1+τλ_med) at τ=0.1 / 1 / 10 | trace τ=0.1 / 0.3 / 1 / 3 / 10 / 30 |\n|---|---|---|---|---|---|---|---|---|')
for _, r in sp.iterrows():
    f = ' / '.join(f'{1 / (1 + t * r.lam_med):.3f}' for t in (0.1, 1, 10))
    tr = ' / '.join(f'{r[f"dof_t{t:g}"]:.0f}' for t in (0.1, 0.3, 1, 3, 10, 30))
    o.append(f'| {r.ensemble} | {r.graph} | {r.lap} | {f"{r.mean_degree:.1f}" if r.mean_degree == r.mean_degree else "–"} | {r.lam2:.3g} | {r.lam_med:.3g} | {r.lam_max:.3g} | {f} | {tr} |')
# ---- T2
d = pd.read_csv(OUT + 'l2_t2.csv')
o.append('\n## T2. Reference p0_τ: feasibility, populations, recovery, ESS, PUF2 weight (family A unless noted) `[l2]`\n')
o.append('LP: "feas" = interior-feasible (exact MaxEnt exists); otherwise l1 = min over the simplex of Σ_s |O_s w − t_s|/t_s. '
         'drop = constraints dropped because the target lies below every frame\'s smoothed openness (unattainable). '
         'prod = production p0 (exact if feasible and no extra drop, else d3 log-LS soft S=100); soft1 = log-LS soft S=1. Pops in %.\n')
o.append('| ensemble | graph | L | O | τ | LP | drop | p0 | ESS % | Folded / PUF1 / PUF2 / PUF3 / U (PUF2-like) | rec | PUF2 weight % | max |ach/t−1| |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
def fmtp(r):
    s = f'{100*r.pop_Folded:.1f} / {100*r.pop_PUF1:.2f} / {100*r.pop_PUF2:.2g} / {100*r.pop_PUF3:.2g} / {100*r.pop_unfolded:.2g}'
    return s + (f' ({100*r["pop_PUF2-like"]:.2g})' if r.ensemble == 'AF2_filtered' else '')
for (ens, graph, fam), g0 in d.groupby(['ensemble', 'graph', 'fam'], sort=False):
    for lap in ('L', 'Lsym'):
        for tau in (0, 0.1, 0.3, 1, 3, 10, 30):
            if tau == 0 and lap == 'Lsym': continue
            g = g0[(g0.lap == lap) & (g0.tau == tau)]
            if not len(g): continue
            for _, r in g[g.use | (g.solver == 'soft1')].sort_values('use', ascending=False).iterrows():
                lp = 'feas' if r.lp_feasible else f'l1 {r.lp_l1rel:.2f}'
                nd = len([x for x in str(r.dropped).split(';') if x and x != 'nan'])
                o.append(f'| {ens} | {graph} | {lap if tau else "binary"} | {fam} | {tau:g} | {lp} | {nd} | {"prod:" + r.solver if r.use else "soft1"} | '
                         f'{100*r.ESS:.1f} | {fmtp(r)} | {r.rec:.1f} | {100*r.pop_PUF2:.2g} | {r.max_rel_err:.2g} |')
# ---- T3
s = pd.read_csv(OUT + 'l4_summary.csv')
RU = {'ref_p0': 'p0', 'A_flags': 'A', 'B_F': 'B', 'C_minchi2': 'C', 'C_1SE': 'C1SE', 'D_flags_then_F': 'D', 'D_ess50': 'D|ESS≥50', 'ref_oracle': 'oracle'}
o.append('\n## T3. Rule picks per τ (σ floored; 6 folds; mean ± SD) `[l4]`\n')
o.append('Cell: recovery ± SD (min) · ESS % · PUF2 % · #folds ESS ≥ 50 · #folds picking p0.\n')
o.append('| ensemble | L | p0 | γ (abs) | τ | ' + ' | '.join(RU.values()) + ' |\n|' + '---|' * (5 + len(RU)))
for (ens, part, lap, req, gamma), g0 in s.groupby(['ensemble', 'part', 'lap', 'p0req', 'gamma'], sort=False):
    for tau, g in g0.groupby('tau'):
        cells = []
        for rk in RU:
            r = g[g.rule == rk]
            if not len(r): cells.append('–'); continue
            r = r.iloc[0]
            cells.append(f'{r.rec_mean:.1f} ± {r.rec_std:.1f} ({r.rec_min:.1f}) · {100*r.ESS_mean:.0f} · {100*r.pop_PUF2_mean:.2g} · {r.n_ESS_ge50} · {r.n_picks_p0}')
        o.append(f'| {ens} | {lap} | {g.p0solver.iloc[0]}{" (S=1)" if req == "soft1" else ""} | {gamma:g} ({g.gamma_abs.iloc[0]:.3g}) | {tau:g} | ' + ' | '.join(cells) + ' |')
open(OUT + 'l6_tables.md', 'w').write('\n'.join(o) + '\n'); print('\n'.join(o))
