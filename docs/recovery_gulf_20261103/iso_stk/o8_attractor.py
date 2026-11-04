"""[o8] what the OMC term pulls toward, independent of data: the null vector of each penalty (L = D - K: uniform; L_sym:
w ∝ sqrt(degree); L_st: w ∝ sqrt(d)) -> recovery / populations / ESS on ISO_BI / ISO_TRI; and how the sidecar kernels' degree
distributes over the open / closed / unassigned (intermediate) frames. Also the cluster-label definition check: the ISO labels
come from Calpha RMSD to the open / closed references (cluster_assignments_*.csv columns rmsd_open / rmsd_closed), i.e. the
same coordinate as the RMSD kernel.
-> o8_attractor.csv, o8.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import i0_common as C
import o3_omc_iso as O3
Q = C.Q
rows = []
for which in ('bi', 'tri'):
    m = Q.ISO(which); lab = m.labels; N = m.nF
    G = np.load(C.OUT + 'o1_graphs.npz')
    cands = {'Lst': None, 'knnLsym': None, 'complete': None}
    for met in ('work_scale', 'rmsd', 'pyrosetta'):
        for h in (0.01, 0.1, 1.0, 10.0):
            if met == 'rmsd' and h == 0.01: continue
            cands[f'{met}|h{h:g}'] = None
    dist = C.iso_distances(which)
    for name in cands:
        if name in ('Lst', 'knnLsym'):
            d = G[f'{which}|d'] if name == 'Lst' else None
            if name == 'knnLsym':
                z = np.load(C.ISO_KNN); A = np.zeros((2225, 2225)); A[z['edge_sources'], z['edge_targets']] = 1; A = np.maximum(A, A.T)
                np.fill_diagonal(A, 0)
                if which == 'bi': A = A[np.ix_(C.iso_bi_index(), C.iso_bi_index())]
                w = np.sqrt(A.sum(1)); w /= w.sum(); forms = {'mat': w}       # null vector of L_sym = sqrt(degree)
            else:
                w = np.sqrt(d); w /= w.sum(); forms = {'mat': w}
        else:
            if name == 'complete':
                K = np.ones((N, N)); np.fill_diagonal(K, 0)
            else:
                met, h = name.split('|'); K = C.sidecar_kernel(dist[met][0], float(h[1:]))
            deg = K.sum(1)
            ws = np.sqrt(deg / deg.max()) if deg.max() > 0 else np.ones(N); ws = ws / ws.sum()
            forms = {'lap': np.full(N, 1 / N), 'sym': ws}
        for form, w in forms.items():
            P = m.pops(w)
            r = dict(system=f'ISO_{which.upper()}', kernel=name, form=form, rec=m.recovery(w), ESS=Q.ess(w), open=P[0], closed=P[1], unassigned=P[2])
            rows.append(r)
            print(f'{r["system"]:7s} {name:18s} {form:4s} null-vector attractor: rec {r["rec"]:6.2f} ESS {100 * r["ESS"]:5.1f}% '
                  f'open/closed/unassigned {100 * P[0]:5.1f}/{100 * P[1]:5.1f}/{100 * P[2]:5.1f}', flush=True)
    # RMSD h=0.1 degree by state (median-scaled RMSD)
    K = C.sidecar_kernel(dist['rmsd'][0], 0.1); deg = K.sum(1)
    for st, s in ((0, 'open'), (1, 'closed'), (-1, 'unassigned')):
        mm = lab == st
        if mm.any():
            print(f'   ISO_{which.upper()} RMSD h=0.1 degree, {s:10s}: median {np.median(deg[mm]):.3g}, log10 median {np.median(np.log10(deg[mm] + 1e-300)):.1f} '
                  f'(n={mm.sum()})', flush=True)
lb = pd.read_csv('jaxent/examples/1_IsoValidation_OMass/data/_clustering_results/cluster_assignments_ISO_TRI.csv')
lb = lb[lb.ensemble == 'ISO_TRI']
print('[labels] ISO_TRI cluster_assignment vs Calpha RMSD to references (min rmsd per state):')
print(lb.groupby('cluster_assignment')[['rmsd_open', 'rmsd_closed', 'min_rmsd']].describe().loc[:, (slice(None), ['min', '50%', 'max'])].round(2).to_string())
pd.DataFrame(rows).to_csv(C.OUT + 'o8_attractor.csv', index=False)
