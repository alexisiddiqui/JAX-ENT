"""[q5] Markdown tables from q3_selected.csv (formatting only) -> q5_tables.md"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import q0_common as Q

S = pd.read_csv(Q.OUT + 'q3_selected.csv', dtype={'sigma': str})
RULES = ['A_flags', 'B_F', 'C_minchi2', 'C_1SE', 'D_flags_then_F', 'ref_p0', 'ref_oracle']
out = []
ms = lambda x, f='{:.1f}': f'{f.format(np.mean(x))} ± {f.format(np.std(x, ddof=1) if len(x) > 1 else 0)}'


def fmtS(s):
    return '0 (p0)' if s == 0 else f'{s:.3g}'


out.append('## T1. MoPrP: rule summary over 6 folds (3 block splits × 2 folds), mean ± SD `[q3]`\n')
out.append('| ensemble | σ | rule | S median [min–max] | ESS % | Folded / PUF1 / PUF2 / PUF3 / U (PUF2-like) % | recovery % (min) | held-out χ²/cell | flags | F − F(p0) | rec ≥ 95 | ESS > 50% | picks p0 |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for ens in Q.D.ENSS:
    for sv in ('floor', 'unfloored'):
        for rule in RULES:
            g = S[(S.system == 'MoPrP') & (S.ensemble == ens) & (S.sigma == sv) & (S.rule == rule)]
            if not len(g): continue
            pops = ' / '.join(f'{100 * g[f"pop_{s}"].mean():.1f}' for s in ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded'])
            if ens == 'AF2_filtered': pops += f' ({100 * g["pop_PUF2-like"].mean():.2f})'
            out.append(f'| {ens} | {sv} | {rule} | {fmtS(g.S.median())} [{fmtS(g.S.min())}–{fmtS(g.S.max())}] | {ms(100 * g.ESS, "{:.0f}")} | {pops} | '
                       f'{ms(g.rec)} ({g.rec.min():.1f}) | {ms(g.ho_chi2_cell, "{:.2f}")} | {ms(g.ho_n_flags)} | {ms(g.F_rel)} | '
                       f'{(g.rec >= 95).sum()}/{len(g)} | {(g.ESS > 0.5).sum()}/{len(g)} | {(g.S == 0).sum()}/{len(g)} |')
out.append('')

for sv in ('floor', 'unfloored'):
    out.append(f'## T2{"a" if sv == "floor" else "b"}. MoPrP per fold, σ {sv} `[q3]`\n')
    out.append('Cells: S · ESS% · rec% · χ²/cell · flags. Held-out block is named; training = the other block.\n')
    out.append('| ensemble | split | held-out | n cells (t_n) | min flags | ' + ' | '.join(RULES) + ' |')
    out.append('|---|---|---|---|---|' + '---|' * len(RULES))
    for ens in Q.D.ENSS:
        g0 = S[(S.system == 'MoPrP') & (S.ensemble == ens) & (S.sigma == sv)]
        for (sp, ho), g in g0.groupby(['split', 'heldout'], sort=False):
            g = g.set_index('rule')
            cells = [f'{fmtS(g.loc[r, "S"])} · {100 * g.loc[r, "ESS"]:.0f} · {g.loc[r, "rec"]:.1f} · {g.loc[r, "ho_chi2_cell"]:.2f} · {g.loc[r, "ho_n_flags"]}' for r in RULES]
            out.append(f'| {ens} | {sp} | {ho} | {g.ho_n_cells.iloc[0]} ({g.ho_t_n.iloc[0]:.2f}) | {g.min_flags.iloc[0]} | ' + ' | '.join(cells) + ' |')
    out.append('')

out.append('## T3. MoPrP flagged peptides (σ floor) at p0 and at each rule\'s pick `[q3]`\n')
out.append('Format: peptide(number of flagged timepoints).\n')
out.append('| ensemble | held-out | p0 | A | B | C | C1SE | D |')
out.append('|---|---|---|---|---|---|---|---|')
for ens in Q.D.ENSS:
    g0 = S[(S.system == 'MoPrP') & (S.ensemble == ens) & (S.sigma == 'floor')]
    for (sp, ho), g in g0.groupby(['split', 'heldout'], sort=False):
        g = g.set_index('rule')
        out.append(f'| {ens} | {sp} {ho} | ' + ' | '.join(str(g.loc[r, 'ho_flagged_units']) if isinstance(g.loc[r, 'ho_flagged_units'], str) else '–'
                                                         for r in ['ref_p0', 'A_flags', 'B_F', 'C_minchi2', 'C_1SE', 'D_flags_then_F']) + ' |')
out.append('')

iso = S[S.system.str.startswith('ISO')]
if len(iso):
    out.append('## T4. ISO control: rule summary over 3 noise seeds × 3 block splits × 2 folds = 18 replicates, mean ± SD `[q3]`\n')
    out.append('| system | noise σ | rule | S median [min–max] | ESS % | open / closed / unassigned % | recovery % (min) | rec ≥ 95 | held-out χ²/cell | flags | at ESS-10% edge |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for sysn in ['ISO_BI', 'ISO_TRI']:
        for sg in ('0.01', '0.02'):
            for rule in RULES:
                g = iso[(iso.system == sysn) & (iso.sigma == sg) & (iso.rule == rule)]
                if not len(g): continue
                pops = f'{100 * g.pop_open.mean():.1f} / {100 * g.pop_closed.mean():.1f} / {100 * g.pop_unassigned.mean():.1f}'
                out.append(f'| {sysn} | {sg} | {rule} | {fmtS(g.S.median())} [{fmtS(g.S.min())}–{fmtS(g.S.max())}] | {ms(100 * g.ESS, "{:.0f}")} | {pops} | '
                           f'{ms(g.rec)} ({g.rec.min():.1f}) | {(g.rec >= 95).sum()}/{len(g)} | {ms(g.ho_chi2_cell, "{:.2f}")} | {ms(g.ho_n_flags)} | {g.at_grid_edge.mean():.2f} |')
    out.append('')
open(Q.OUT + 'q5_tables.md', 'w').write('\n'.join(out))
print('\n'.join(out))
