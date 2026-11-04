"""[l1] T1: Laplacian-smoothed openness o_tau = (I + tau L)^-1 o.
Graphs: MSAss = stored sidecar k=20 L2-logPF symmetric-union graph (unweighted, as stored; 7215 edges);
        Filtered = rebuilt (same recipe, BV lnPF of _featurise_a101_k109, Gaussian weights at median-scaled bandwidth);
        MSAss 'rebuilt' (Gaussian) as a check. Laplacians: L = D - A and L_sym = I - D^-1/2 A D^-1/2.
O families: A (rule features; 4 distinct region rows b1/a1/b2/core) and B70 (native-contact Q_seg < 0.7).
-> l1_t1_summary.csv (per ens x graph x L x fam x tau x row), l1_t1_states.csv (per-state mean o_tau), l1_spectrum.csv,
   l1_O.npz (all smoothed O), l1.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import l0_common as L0
D = L0.D

GRAPHS = {'AF2_MSAss': ['primary', 'rebuilt'], 'AF2_filtered': ['primary']}
rows, srows, spec, OS = [], [], [], {}
for ens, glist in GRAPHS.items():
    lab = D.assign(ens)
    for gname in glist:
        A = L0.graph(ens, gname); L, Ls = L0.laplacians(A)
        for lname, LL in (('L', L), ('Lsym', Ls)):
            ev = np.linalg.eigvalsh(LL)
            sp = dict(ensemble=ens, graph=gname, lap=lname, lam2=ev[1], lam_med=np.median(ev), lam_max=ev[-1],
                      mean_degree=A.sum(1).mean() if lname == 'L' else np.nan)
            for tau in L0.TAUS:
                f = 1 / (1 + tau * ev)
                sp.update({f'filt_lam2_t{tau:g}': f[1], f'filt_lammed_t{tau:g}': np.median(f), f'filt_lammax_t{tau:g}': f[-1],
                           f'dof_t{tau:g}': f.sum()})        # trace of the smoother = effective number of modes kept
            spec.append(sp)
            for fam in ('A', 'B70'):
                O, t, names = L0.raw_O(ens, fam)
                # distinct rows (A: regions)
                _, first = np.unique(O, axis=0, return_index=True); first = sorted(first)
                for tau in [0.0] + L0.TAUS:
                    Ot = L0.smooth(O, LL, tau)
                    OS[f'{ens}|{gname}|{lname}|{fam}|{tau:g}'] = Ot
                    for i in first:
                        o, ot = O[i], Ot[i]
                        grp = '+'.join(n for n, oo in zip(names, O) if np.array_equal(oo, o))
                        frac = (ot > 1e-3) & (ot < 1 - 1e-3)
                        r = dict(ensemble=ens, graph=gname, lap=lname, fam=fam, tau=tau, row=grp, n_open_raw=int(o.sum()),
                                 mean_raw=o.mean(), mean_tau=ot.mean(), colsum_rel_change=(ot.sum() - o.sum()) / max(o.sum(), 1e-300),
                                 n_fractional=int(frac.sum()), n_gt_0=int((ot > 1e-3).sum()), min=ot.min(), max=ot.max(),
                                 max_on_open=ot[o > 0].max() if o.any() else np.nan, min_on_open=ot[o > 0].min() if o.any() else np.nan,
                                 max_on_closed=ot[o == 0].max() if (o == 0).any() else np.nan, mean_abs_change=np.abs(ot - o).mean(),
                                 target_geo=float(np.exp(np.mean(np.log([t[j] for j, oo in enumerate(O) if np.array_equal(oo, o)])))))
                        rows.append(r)
                        for s in range(6):
                            m = lab == s
                            if m.sum(): srows.append(dict(ensemble=ens, graph=gname, lap=lname, fam=fam, tau=tau, row=grp,
                                                          state=L0.STATES5[s], n=int(m.sum()), mean_o=ot[m].mean(), max_o=ot[m].max()))
df = pd.DataFrame(rows); ds = pd.DataFrame(srows); dsp = pd.DataFrame(spec)
df.to_csv(L0.OUT + 'l1_t1_summary.csv', index=False); ds.to_csv(L0.OUT + 'l1_t1_states.csv', index=False)
dsp.to_csv(L0.OUT + 'l1_spectrum.csv', index=False); np.savez_compressed(L0.OUT + 'l1_O.npz', **OS)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 2000)
fmt = lambda v: f'{v:.4g}'
print('== spectrum =='); print(dsp[['ensemble', 'graph', 'lap', 'mean_degree', 'lam2', 'lam_med', 'lam_max'] +
                              [f'dof_t{t:g}' for t in L0.TAUS]].to_string(index=False, float_format=fmt))
print('\n== A rows (primary graph) ==')
print(df[(df.fam == 'A') & (df.graph == 'primary')][['ensemble', 'lap', 'tau', 'row', 'n_open_raw', 'mean_raw', 'mean_tau', 'colsum_rel_change',
      'n_fractional', 'n_gt_0', 'min_on_open', 'max_on_closed', 'mean_abs_change', 'target_geo']].to_string(index=False, float_format=fmt))
print('\n== per-state mean o_tau, A, primary graph, L ==')
x = ds[(ds.fam == 'A') & (ds.graph == 'primary') & (ds.lap == 'L')]
print(x.pivot_table(index=['ensemble', 'row', 'tau'], columns='state', values='mean_o', sort=False).to_string(float_format=lambda v: f'{v:.3f}'))
print('\n== per-state mean o_tau, A, primary graph, Lsym ==')
x = ds[(ds.fam == 'A') & (ds.graph == 'primary') & (ds.lap == 'Lsym')]
print(x.pivot_table(index=['ensemble', 'row', 'tau'], columns='state', values='mean_o', sort=False).to_string(float_format=lambda v: f'{v:.3f}'))
print('\n== B70 rows (primary, L) ==')
print(df[(df.fam == 'B70') & (df.graph == 'primary') & (df.lap == 'L') & df.tau.isin([0, 1, 10])][['ensemble', 'tau', 'row', 'n_open_raw', 'mean_raw',
      'mean_tau', 'colsum_rel_change', 'n_fractional', 'mean_abs_change']].to_string(index=False, float_format=fmt))
