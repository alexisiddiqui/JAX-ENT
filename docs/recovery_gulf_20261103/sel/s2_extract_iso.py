"""[s2] ISO P2 candidates: frame weights + residue-level predictions on train+val cells, using the exact native context of
run_iso_policy_sidecars.analyze() (self-consistent fit_features, frame_uptake, identity map). Replays val_mse (assert),
recovery (assert). Also per-frame mean lnPF (for work scale) and open/closed/unassigned populations. Read-only.
usage: s2_extract_iso.py SHARD NSHARDS"""
import sys, time, contextlib, io, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel')
import s0_common as S
ISO = S.REPO / 'jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT'
sys.path.insert(0, str(ISO))
import run_iso_policy_sidecars as RI
import jax, jax.numpy as jnp
base, original = RI.base, RI.original
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
man = json.load(open(S.REPO / 'artifacts/iso_policy_sidecars/manifest.json'))
s0 = man['specs'][0]
FD, CD, DD = s0['features_dir'], s0['clustering_dir'], s0['datasplit_dir']
cand = S.p2(pd.read_csv(S.REPO / 'artifacts/iso_policy_sidecars/candidates.csv'))
keys = sorted(cand.groupby(['ensemble', 'split_type', 'split_idx']).groups)
mykeys = keys[shard::nsh]
for key in mykeys:
    t0 = time.time()
    ens, st, si = key
    with contextlib.redirect_stdout(io.StringIO()):
        features, top = base.shrinkage.load_features(FD, ens)
        labels = base.shrinkage.load_cluster_assignments(CD, ens)
        model = base.shrinkage.configure_model("uptake", labels)
        train, val = base.shrinkage.load_split(DD, st, si)
        loader = base.source.create_data_loaders(train + val, train, val, features, top)
    forward = model.forward[base.shrinkage.m_key("HDX_peptide")]
    predict = RI.make_predict(forward, features)
    vp = jax.jit(jax.vmap(lambda w: predict(w, model.params)))
    vi = np.array([p.top.fragment_index for p in val]); ti = np.array([p.top.fragment_index for p in train])
    yv = np.asarray(loader.val.y_true)[..., 0]; yt = np.asarray(loader.train.y_true)[..., 0]
    Mv = np.asarray(loader.val.residue_feature_ouput_mapping.todense()); np.testing.assert_array_equal(Mv, np.eye(features.features_shape[0])[vi])
    Mt = np.asarray(loader.train.residue_feature_ouput_mapping.todense()); np.testing.assert_array_equal(Mt, np.eye(features.features_shape[0])[ti])
    bc, bh = float(np.asarray(model.params.bv_bc)), float(np.asarray(model.params.bv_bh))
    lnpf = bc * np.asarray(features.heavy_contacts, float) + bh * np.asarray(features.acceptor_contacts, float)
    lbar = lnpf.mean(0)
    G = cand[(cand.ensemble == ens) & (cand.split_type == st) & (cand.split_idx == si)]
    out_W, out_P, rid_l, ci_l = [], [], [], []
    maxerr = 0
    for rid, g in G.groupby('run_id'):
        g = g.set_index('candidate_index')
        h = base.source.load_optimization_history_from_file(str(g.history_path.iloc[0]))
        Ws, idx = [], []
        for i, (kind, state, w) in enumerate(original.candidates(h)):
            if i in g.index:
                assert g.loc[i, 'step'] == int(state.step) and g.loc[i, 'candidate_kind'] == kind
                for a_, b_ in zip(jax.tree_util.tree_leaves(state.params.model_parameters[0]), jax.tree_util.tree_leaves(model.params)):
                    np.testing.assert_allclose(a_, b_, rtol=1e-7, atol=1e-9)
                Ws.append(np.asarray(w, float)); idx.append(i)
        assert set(idx) == set(g.index)
        Wm = np.stack(Ws)
        U = np.asarray(vp(jnp.asarray(Wm)))  # (n,T,R)
        Pv = U[:, :, vi].transpose(0, 2, 1); Pt = U[:, :, ti].transpose(0, 2, 1)
        vm = np.mean((Pv - yv) ** 2, axis=(1, 2))
        maxerr = max(maxerr, np.max(np.abs(vm - g.loc[idx].val_mse.values) / g.loc[idx].val_mse.values))
        rec = base.source.calculate_recovery_percentage(labels, Wm[0], base.shrinkage.GROUND_TRUTH, base.shrinkage.STATE_MAPPING)
        assert abs(rec - g.loc[idx[0], 'recovery_percent']) < 1e-4, (rid, rec)
        out_W.append(Wm.astype(np.float32)); out_P.append(np.concatenate([Pv, Pt], 1).astype(np.float32))
        rid_l += [rid] * len(idx); ci_l += idx
    labels = np.asarray(labels)
    np.savez_compressed(S.SEL / f'work/s2_iso_{ens}_{st}_{si}.npz', run_id=np.array(rid_l), candidate_index=np.array(ci_l),
                        W=np.concatenate(out_W), P=np.concatenate(out_P), y=np.concatenate([yv, yt]), n_val=len(vi),
                        lbar=lbar, labels=labels)
    print(f'DONE {key} cands={len(rid_l)} frames={len(labels)} val={len(vi)} train={len(ti)} T={yv.shape[1]} '
          f'max rel val_mse replay err={maxerr:.2e} labels={np.unique(labels)} GT={base.shrinkage.GROUND_TRUTH} SM={base.shrinkage.STATE_MAPPING} '
          f't={time.time()-t0:.0f}s', flush=True)
