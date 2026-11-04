"""[s3] Per-candidate scores for all selection functions on every evaluation scope.
MoPrP scopes: old_val (fit features _featurise, 12-peptide native map, split's validation peptides) = the fit-faithful primary;
              old_all (same, all 12 peptides); new_val (_featurise_a101_k109 + row-normalised map, same validation peptides);
              new_all (new features, all 14 peptides).
ISO scopes:   val (split validation residues), all (train+val residues).  -> work/s3_scores_{arm}.parquet
Residual r = pred - obs. sigma_cell = 1/moprp.weights; sigma_pooled_p = RMS_t(1/w) (no floor). ISO: w=1, sigma=0.0586."""
import sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, '/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel')
import s0_common as S
EPS_LOG = 0.01


def scores(P, D, w, sp, ess_frac, ws):
    """P (n,p,t) predictions; D (p,t) observed; w (p,t) weights; sp (p,t) pooled sigma; returns dict of arrays (lower=better)."""
    P = P.astype(np.float64); r = P - D[None]
    n, p, t = r.shape; out = {}
    out['f1_wmse'] = np.mean(w[None] * r ** 2, (1, 2))
    out['f2_mse'] = np.mean(r ** 2, (1, 2))
    z = r / sp[None]
    out['f3_chi2_pooled'] = np.mean(z ** 2, (1, 2))
    out['f4_logerr'] = np.mean((np.log(np.clip(P, 0, None) + EPS_LOG) - np.log(np.clip(D, 0, None) + EPS_LOG)[None]) ** 2, (1, 2))
    X = P.reshape(n, -1); y = D.ravel()
    rx = rankdata(X, axis=1); ry = rankdata(y)
    def corr(A, b):
        A = A - A.mean(1, keepdims=True); b = b - b.mean()
        return (A @ b) / (np.sqrt((A ** 2).sum(1)) * np.sqrt((b ** 2).sum()) + 1e-300)
    out['f5a_neg_spearman'] = -corr(rx, ry)
    out['f5b_neg_pearson'] = -corr(X, y)
    out['f6_maxabsz'] = np.abs(z).reshape(n, -1).max(1)
    out['f7a_Qn_pooled'] = np.abs(np.mean(z ** 2, (1, 2)) - 1)
    out['f7b_Qn_homo'] = np.abs(np.mean((r / S.SIGMA_HOMO) ** 2, (1, 2)) - 1)
    var = np.maximum(D.var(1), 1e-4)
    out['f8_pep_norm_mse'] = np.mean(np.mean(r ** 2, 2) / var[None], 1)
    out['f9_meancentred_mse'] = np.mean((r - r.mean(2, keepdims=True)) ** 2, (1, 2))
    out['f10_offset_mse'] = np.mean((r - r.mean((1, 2), keepdims=True)) ** 2, (1, 2))
    for b in (0.01, 0.1):
        out[f'f11_wmse_ess_b{b}'] = out['f1_wmse'] + b * (1 - ess_frac)
        out[f'f11_mse_ess_b{b}'] = out['f2_mse'] + b * (1 - ess_frac)
    out['work_scale'] = ws
    return out


def old_lbar(ens):
    z = np.load(S.EX2 / f'fitting/jaxENT/_featurise/features_{ens}.npz')
    return (S.C.BC * z['heavy_contacts'] + S.C.BH * z['acceptor_contacts']).mean(0)


def moprp(arm):
    c = S.add_model(S.p2(pd.read_csv(S.ARMS[arm] / 'candidates.csv')))
    parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(S.SEL / f'work/s1_{arm}_*.npz')))]
    meta = pd.DataFrame(dict(run_id=np.concatenate([z['run_id'] for z in parts]), candidate_index=np.concatenate([z['candidate_index'] for z in parts])))
    W = np.concatenate([z['W'] for z in parts]).astype(np.float64); W /= W.sum(1, keepdims=True)
    Po = np.concatenate([z['P_old'] for z in parts]); Pn = np.concatenate([z['P_new'] for z in parts])
    meta['row'] = np.arange(len(meta))
    c = c.merge(meta, on=['run_id', 'candidate_index'], validate='one_to_one')
    assert len(c) == len(meta)
    D14, W14 = S.C.DEXP, S.C.WTS
    sp14 = np.sqrt(np.mean((1 / W14) ** 2, 1, keepdims=True)) * np.ones((1, 15))
    N = W.shape[1]; ess = 1 / np.sum(W ** 2, 1) / N
    lb = {('old', e): old_lbar(e) for e in S.ENSS}
    lb.update({('new', e): (S.C.BC * S.load_new(e)[1] + S.C.BH * S.load_new(e)[2]).mean(0) for e in S.ENSS})
    comp = np.zeros((len(W), 5))
    for e in S.ENSS:
        m = (c.ensemble == e).values
        comp[c.row.values[m]] = S.composition(S.assignments(e), W[c.row.values[m]])
    rows = []
    for (e, st, si), g in c.groupby(['ensemble', 'split_type', 'split_idx']):
        vi = [x['fragment_index'] for x in json.load(open(S.EX2 / f'fitting/jaxENT/_datasplits/{st}/split_{si:03d}/val_topology.json'))['topologies']]
        rr = g.row.values
        ess_g = ess[rr]
        for scope, P, idx, lbk in (('old_val', Po, vi, 'old'), ('old_all', Po, list(range(12)), 'old'),
                                   ('new_val', Pn, vi, 'new'), ('new_all', Pn, list(range(14)), 'new')):
            ws = S.RT * np.abs((W[rr] - 1 / N) @ lb[(lbk, e)])
            sc = scores(P[rr][:, idx], D14[idx], W14[idx], sp14[idx], ess_g, ws)
            if scope == 'old_val':  # replay check against candidates.csv
                assert np.allclose(sc['f2_mse'], g.val_mse.values, rtol=2e-5), (e, st, si)
                assert np.allclose(sc['f1_wmse'], g.val_weighted_mse.values, rtol=2e-5), (e, st, si)
            d = g[['run_id', 'candidate_index', 'ensemble', 'model', 'method', 'edge_type', 'graph_k', 'split_type', 'split_idx',
                   'strength', 'bandwidth', 'step', 'candidate_kind', 'recovery_percent', 'ess_percent']].copy()
            d['scope'] = scope
            for k, v in sc.items():
                d[k] = v
            for j, s_ in enumerate(S.STATES):
                d[s_] = comp[rr, j]
            rows.append(d)
    out = pd.concat(rows, ignore_index=True); out.insert(0, 'arm', arm)
    return out


def iso():
    c = S.p2(pd.read_csv(S.REPO / 'artifacts/iso_policy_sidecars/candidates.csv'))
    c['model'] = np.where(c.method == 'maxent', 'maxent', 'omc_' + c.graph_metric.astype(str))
    rows = []
    for f in sorted(glob.glob(str(S.SEL / 'work/s2_iso_*.npz'))):
        z = np.load(f, allow_pickle=True)
        meta = pd.DataFrame(dict(run_id=z['run_id'], candidate_index=z['candidate_index'])); meta['row'] = np.arange(len(meta))
        g = c.merge(meta, on=['run_id', 'candidate_index'], validate='one_to_one'); assert len(g) == len(meta)
        W = z['W'].astype(np.float64); W /= W.sum(1, keepdims=True); N = W.shape[1]
        rr = g.row.values; ess = 1 / np.sum(W[rr] ** 2, 1) / N
        ws = S.RT * np.abs((W[rr] - 1 / N) @ z['lbar'])
        lab = z['labels']; pops = {nm: W[rr][:, lab == v].sum(1) for nm, v in (('open', 0), ('closed', 1), ('intermediate', -1))}
        nv = int(z['n_val']); P = z['P'][rr]; y = z['y']
        for scope, sl in (('val', slice(0, nv)), ('all', slice(None))):
            D = y[sl]; one = np.ones_like(D); sp = np.full_like(D, S.SIGMA_HOMO)
            sc = scores(P[:, sl], D, one, sp, ess, ws)
            if scope == 'val':
                assert np.allclose(sc['f2_mse'], g.val_mse.values, rtol=1e-4)
            d = g[['run_id', 'candidate_index', 'ensemble', 'model', 'method', 'graph_metric', 'split_type', 'split_idx',
                   'data_strength', 'bandwidth', 'step', 'candidate_kind', 'recovery_percent', 'ess_percent', 'val_closed_sigma_mse']].copy()
            d['scope'] = scope
            for k, v in sc.items():
                d[k] = v
            if scope == 'all':
                d['val_closed_sigma_mse'] = np.nan  # defined on validation residues only
            for k, v in pops.items():
                d[k] = v
            rows.append(d)
        print(f, len(g), flush=True)
    out = pd.concat(rows, ignore_index=True); out.insert(0, 'arm', 'iso')
    return out.rename(columns={'data_strength': 'strength'})


if __name__ == '__main__':
    for arm in ('full_uptake', 'rate'):
        d = moprp(arm); d.to_parquet(S.SEL / f'work/s3_scores_{arm}.parquet'); print(arm, d.shape, flush=True)
    d = iso(); d.to_parquet(S.SEL / 'work/s3_scores_iso.parquet'); print('iso', d.shape)
