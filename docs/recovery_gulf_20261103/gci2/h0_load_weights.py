"""T5 prep: load MaxEnt frame weights for ex2_fresh candidates (P2 pool: no running_best, step>=100) from HDF5 histories,
replicating analyze_within_trajectory_selection.candidates() ordering for candidate_index; verify recovery. -> h0_weights.npz, h0_pool.csv"""
import sys, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import g0_common as G
from jaxent.src.utils.hdf import load_optimization_history_from_file
from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
OUT = '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2/'
c = pd.read_csv('jaxent/examples/2_CrossValidation/analysis/_moprp_policy_sidecars_fresh_20261023/candidates.csv')
c = c[c.method == 'maxent']
pool = c[(c.candidate_kind != 'running_best') & (c.step >= 100)].copy()


def cands(h):
    seen = set(); e = [('trajectory', s) for s in h.states] + [('convergence', s) for s in h.convergence_states]
    if h.best_state is not None: e.append(('running_best', h.best_state))
    for kind, st in e:
        w = np.asarray(validated_frame_weight_simplex(st.params.frame_weight_simplex), float); w /= w.sum()
        key = (int(st.step), hashlib.sha256(w.tobytes()).hexdigest())
        if key in seen: continue
        seen.add(key); yield kind, st, w

W = {}; maxd = 0
asg = {e: G.assign(e) for e in G.ENSS}
for rid, g in pool.groupby('run_id'):
    h = load_optimization_history_from_file(g.history_path.iloc[0])
    lst = list(cands(h)); want = set(g.candidate_index)
    for i, (kind, st, w) in enumerate(lst):
        if i in want:
            row = g[g.candidate_index == i].iloc[0]
            assert row.step == int(st.step) and row.candidate_kind == kind, (rid, i)
            d = abs(G.recovery(asg[row.ensemble], w) - row.recovery_percent); maxd = max(maxd, d)
            W[f'{rid}|{i}'] = w
    print(rid, len(g), f'maxd={maxd:.2e}', flush=True)
np.savez_compressed(OUT + 'h0_weights.npz', **W)
pool.to_csv(OUT + 'h0_pool.csv', index=False)
print('n', len(W), 'max |drec|', maxd)
