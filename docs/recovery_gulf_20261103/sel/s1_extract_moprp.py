"""[s1] Extract, for every P2 candidate of a MoPrP arm, the frame weights and peptide-level predictions:
  P_old (n,12,15): native jaxent predict on the fit's own features (manifest features_dir = _featurise) and the
                   native full-dataset sparse map (12 peptides), exactly as rescore_moprp_all_peptide_selection.py;
  P_new (n,14,15): NumPy forward on _featurise_a101_k109 + row-normalised map (gci2/c0_common), same averaging mode.
Replays val_mse / val_weighted_mse / recovery / ESS from candidates.csv (asserts). Read-only.
usage: s1_extract_moprp.py ARM SHARD NSHARDS"""
import sys, time, contextlib, io
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel')
import s0_common as S
EXA = S.REPO / 'jaxent/examples/2_CrossValidation'
sys.path.insert(0, str(EXA / 'analysis'))
import rescore_moprp_all_peptide_selection as R  # imports run_moprp_policy_sidecars as fixed
import jax, jax.numpy as jnp
fixed = R.fixed

arm, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cand = S.p2(pd.read_csv(S.ARMS[arm] / 'candidates.csv'))
import json
man = json.load(open(S.CAMPAIGN[arm] / 'manifest.json'))
specs = {s.run_id: s for s in (fixed.RunSpec(**d) for d in man['specs'])}
runs = sorted(cand.run_id.unique())
# shard by context key to reuse contexts
keyof = lambda r: (specs[r].ensemble, specs[r].split_type, specs[r].split_idx)
keys = sorted({keyof(r) for r in runs})
mykeys = set(keys[shard::nsh])
runs = [r for r in runs if keyof(r) in mykeys]
contexts, datasets = {}, {}
out = dict(run_id=[], candidate_index=[], W=[], P_old=[], vrep=[])
t0 = time.time(); maxerr = 0.0
vpred = {}
for n, rid in enumerate(runs):
    spec = specs[rid]
    g = cand[cand.run_id == rid].set_index('candidate_index')
    with contextlib.redirect_stdout(io.StringIO()):
        context, (mapping, observed, surface, val_idx, ids) = R.full_dataset(spec, fixed, contexts, datasets)
    native, predict, assign = context[2], context[-1], context[6]
    key = (spec.ensemble, spec.split_type, spec.split_idx, spec.frame_averaging_mode)
    if key not in vpred:
        vpred[key] = jax.jit(jax.vmap(lambda w, p: predict(w, p), in_axes=(0, None)))
    h = fixed.load_optimization_history_from_file(str(spec.saved_history))
    Ws, idx = [], []
    for i, (kind, st, w) in enumerate(fixed.saved_candidates(h)):
        if i not in g.index:
            continue
        row = g.loc[i]
        assert row.step == int(st.step) and row.candidate_kind == kind, (rid, i)
        fixed.validate_protocol(st.params.model_parameters[0], native.params) if spec.frame_averaging_mode == 'linear_uptake' else None
        for a_, b_ in zip(jax.tree_util.tree_leaves(st.params.model_parameters[0]), jax.tree_util.tree_leaves(native.params)):
            np.testing.assert_allclose(a_, b_, rtol=1e-7, atol=1e-8)
        Ws.append(np.asarray(w, float)); idx.append(i)
    assert set(idx) == set(g.index), rid
    Wm = np.stack(Ws)
    U = np.asarray(vpred[key](jnp.asarray(Wm), native.params))  # (n,T,R)
    P = np.einsum('pr,ntr->npt', mapping, U)
    # replay
    vm = np.mean((P[:, val_idx] - observed[val_idx]) ** 2, axis=(1, 2))
    vw = np.mean(surface[val_idx] * (P[:, val_idx] - observed[val_idx]) ** 2, axis=(1, 2))
    ref = g.loc[idx]
    err = max(np.max(np.abs(vm - ref.val_mse.values) / np.maximum(1e-12, ref.val_mse.values)),
              np.max(np.abs(vw - ref.val_weighted_mse.values) / ref.val_weighted_mse.values))
    rec = np.array([S.C.recovery(assign, w) for w in Wm[:3]])
    assert np.allclose(rec, ref.recovery_percent.values[:3], atol=1e-5), (rid, rec)
    maxerr = max(maxerr, err)
    out['run_id'] += [rid] * len(idx); out['candidate_index'] += idx
    out['W'].append(Wm.astype(np.float64)); out['P_old'].append(P.astype(np.float32))
    if n % 10 == 0:
        print(f'{arm} shard{shard} {n+1}/{len(runs)} {rid} n={len(idx)} rel_replay_err={maxerr:.2e} t={time.time()-t0:.0f}s', flush=True)
W = np.concatenate(out['W']); Pold = np.concatenate(out['P_old'])
meta = pd.DataFrame(dict(run_id=out['run_id'], candidate_index=out['candidate_index']))
ens_of = meta.run_id.map(lambda r: specs[r].ensemble).values
Pnew = np.zeros((len(W), 14, 15), np.float32)
for e in S.ENSS:
    m = ens_of == e
    if m.any():
        Pnew[m] = S.new_forward(e, W[m], S.MODE[arm])[0]
np.savez_compressed(S.SEL / f'work/s1_{arm}_{shard:02d}.npz', run_id=meta.run_id.values, candidate_index=meta.candidate_index.values,
                    W=W.astype(np.float32), P_old=Pold, P_new=Pnew, observed_old=observed, surface_old=surface)
print(f'DONE {arm} shard{shard} runs={len(runs)} cands={len(W)} max rel replay err={maxerr:.2e} t={time.time()-t0:.0f}s', flush=True)
