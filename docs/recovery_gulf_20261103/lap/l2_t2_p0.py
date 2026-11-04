"""[l2] T2: re-solve p0 on smoothed openness. For every ens x graph x Laplacian x O family x tau:
 - merge identical raw rows (geometric-mean target), drop rows whose target is outside (min, max) of the smoothed row (q0/d3 rule)
 - LP feasibility of the merged system: eps* = max min_i w_i s.t. O w = t, sum w = 1 (eps* > 0 <=> exact MaxEnt exists);
   l1rel = min over the simplex of sum_s |O_s w - t_s| / t_s (0 if feasible)
 - exact MaxEnt dual (dg V.maxent_exact, uniform base) -> p0_exact; d3 log-least-squares soft fit (S = 100) -> p0_soft
 - production p0 ('use') = exact if LP-feasible AND no constraint dropped beyond those dropped at tau = 0 (a row is dropped when
   its target lies below every frame's smoothed openness, i.e. it is unattainable, which is also infeasibility), else soft (S=100);
   tau = 0 keeps q0's exact solution for comparability with selgci. Soft S = 1 is reported as a sensitivity ('soft1').
 - nesting check: (I + tau L)^-1 is entrywise >= 0 (inverse of a nonsingular M-matrix, both L and L_sym), so a raw nesting
   o_a1 <= o_b2 survives smoothing; reported as min_i (o_b2 - o_a1) and min entry of the smoother.
Reports state populations, recovery (target 97.1/2.4/0.5), ESS, PUF2 weight, per-constraint achieved/target.
-> l2_t2.csv, l2_constraints.csv, l2_p0.npz (log p0 for every case and solver), l2.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import l0_common as L0
D, Q = L0.D, L0.Q

OS = dict(np.load(L0.OUT + 'l1_O.npz'))
rows, crows, P0 = [], [], {}
NDROP0 = {}
for key in OS:
    ens, gname, lname, fam, tau = key.split('|'); tau = float(tau)
    if lname == 'Lsym' and gname != 'primary': continue
    O0, t, names = L0.raw_O(ens, fam); Ot = OS[key]; lab = D.assign(ens)
    R = L0.solve_p0(Ot, t, names, Oraw=O0)
    if tau == 0: NDROP0[(ens, gname, lname, fam)] = len(R['dropped'])
    w1, _ = L0.V.soft_fit(Ot, t, 1)
    for solver, w in (('exact', np.exp(R['logp0'])), ('soft', R['w_soft']), ('soft1', w1)):
        P = D.state_pops(lab, w)
        r = dict(ensemble=ens, graph=gname, lap=lname, fam=fam, tau=tau, solver=solver, n_con=len(R['names']), dropped=';'.join(R['dropped']),
                 lp_eps=R['eps'] if R['eps'] is not None else np.nan, lp_feasible=R['feasible'], lp_l1rel=R['l1rel'],
                 exact_max_rel_err=R['max_rel_err'], exact_lam_max=R['lam_max'], ESS=D.ess(w), rec=D.recovery(lab, w),
                 **{f'pop_{s}': P[i] for i, s in enumerate(L0.STATES5)}, w_PUF2_min=w[lab == 2].min(), w_PUF2_max=w[lab == 2].max(),
                 min_logw=float(np.log(np.maximum(w, 1e-320)).min()))
        ach = R['Oe'] @ w
        r['max_rel_err'] = float(np.max(np.abs(ach / R['te'] - 1))) if len(ach) else np.nan
        rows.append(r)
        for n, a, tt in zip(R['names'], ach, R['te']):
            crows.append(dict(ensemble=ens, graph=gname, lap=lname, fam=fam, tau=tau, solver=solver, constraint=n, target=tt, achieved=a, ratio=a / tt))
        P0[f'{key}|{solver}'] = R['logp0'] if solver == 'exact' else np.log(np.maximum(w, 1e-320))
    extra_drop = len(R['dropped']) > NDROP0[(ens, gname, lname, fam)]
    use = 'exact' if ((R['feasible'] and not extra_drop) or tau == 0) else 'soft'
    for r in rows[-3:]: r['use'] = (r['solver'] == use); r['extra_drop'] = extra_drop
df = pd.DataFrame(rows)
# nesting check (A rows: a1 subset of b2 in MSAss)
print('nesting check, MSAss A: min_i (o_b2 - o_a1) and min entry of (I + tau L)^-1')
O0, _, nm = L0.raw_O('AF2_MSAss', 'A'); ia, ib = nm.index('133-148'), nm.index('154-167')
print(f'  raw: a1 open {int(O0[ia].sum())}, b2 open {int(O0[ib].sum())}, a1 subset of b2: {bool(np.all(O0[ia] <= O0[ib]))}')
LL = dict(zip(('L', 'Lsym'), L0.laplacians(L0.graph('AF2_MSAss'))))
for lname in ('L', 'Lsym'):
    for tau in L0.TAUS:
        Ot = OS[f'AF2_MSAss|primary|{lname}|A|{tau:g}']; Sm = np.linalg.inv(np.eye(500) + tau * LL[lname])
        print(f'  {lname} tau={tau:g}: min(o_b2 - o_a1) = {(Ot[ib] - Ot[ia]).min():.2e}; min entry of smoother = {Sm.min():.2e}')
# tau=0 exact must equal q0's p0 (selgci) for A
for ens in D.ENSS:
    lp, info = Q.moprp_p0(ens)
    d = np.abs(np.exp(P0[f'{ens}|primary|L|A|0|exact']) - np.exp(lp)).max()
    print(f'{ens}: tau=0 exact p0 vs selgci q0 max |dp| = {d:.1e}; q0 max rel err of achieved vs target '
          f'{np.max(np.abs(np.array(info["achieved"]) / np.array(info["targets"]) - 1)):.3g}')
df.to_csv(L0.OUT + 'l2_t2.csv', index=False); pd.DataFrame(crows).to_csv(L0.OUT + 'l2_constraints.csv', index=False)
np.savez_compressed(L0.OUT + 'l2_p0.npz', **P0)
pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 2000)
cols = ['ensemble', 'graph', 'lap', 'fam', 'tau', 'solver', 'use', 'extra_drop', 'n_con', 'dropped', 'lp_eps', 'lp_l1rel', 'max_rel_err', 'ESS', 'rec',
        'pop_Folded', 'pop_PUF1', 'pop_PUF2', 'pop_PUF3', 'pop_unfolded', 'pop_PUF2-like', 'w_PUF2_min']
print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.4g}'))
c = pd.DataFrame(crows)
print('\n== per-constraint achieved/target (A, primary) ==')
print(c[(c.fam == 'A') & (c.graph == 'primary')].pivot_table(index=['ensemble', 'lap', 'tau', 'solver'], columns='constraint', values='ratio', sort=False)
      .to_string(float_format=lambda v: f'{v:.3f}'))
