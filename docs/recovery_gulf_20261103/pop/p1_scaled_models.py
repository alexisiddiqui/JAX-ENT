"""T1: median-matched BV forward models (uniform weights), both ensembles, both averaging modes.
-> p1_models.json, p1_t1.csv, p1_region.csv, p1.log"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import p0_common as P

models, rows, reg_rows = {}, [], []
for ens in P.ENSS:
    res, h, a, k = P.load(ens); M = P.pmap(res); nF = h.shape[1]; u = np.ones(nF) / nF
    pf = P.pfact_vec(res); ok = pf > 0
    pf_all = np.array([v for v in P.C.PFACT.values()]); n_pos_all = int((pf_all > 0).sum()); n_zero = int((pf_all == 0).sum())
    L0 = P.BC0 * h + P.BH0 * a; Lm = L0.mean(1)
    med_pf = np.median(pf[ok]); med_pf_allseq = np.median(pf_all[pf_all > 0])
    for rset, mask in [('matched', ok), ('all97', np.ones(len(res), bool))]:
        med_L = np.median(Lm[mask])
        s = med_pf / med_L                         # lnPF linear in (bc,bh): median scales exactly by s
        d = med_pf - med_L
        q = lambda x: np.subtract(*np.percentile(x, [75, 25]))
        aff = q(pf[ok]) / q(Lm[mask])
        models[f'{ens}|scale|{rset}'] = dict(kind='scale', s=s, bc=s * P.BC0, bh=s * P.BH0)
        models[f'{ens}|shift|{rset}'] = dict(kind='shift', delta=d)
        models[f'{ens}|affine_iqr|{rset}'] = dict(kind='affine', a=aff, medL=med_L, medpf=med_pf)
        # check: scale gives matched median exactly
        assert abs(np.median((s * Lm)[mask]) - med_pf) < 1e-9
    models[f'{ens}|none|-'] = dict(kind='none')
    print(f'{ens}: featurised residues {len(res)}, with pfact>0 {ok.sum()} (seq-wide pfact>0 {n_pos_all}, zeros {n_zero}); '
          f'median pfact>0 matched {med_pf:.3f} / seq-wide {med_pf_allseq:.3f}; median BV(0.35,2) mean-lnPF matched {np.median(Lm[ok]):.3f}, all97 {np.median(Lm):.3f}')


def lnpf_model(ens, key, h, a):
    m = models[key]; L0 = P.BC0 * h + P.BH0 * a
    if m['kind'] == 'none': return L0
    if m['kind'] == 'scale': return m['bc'] * h + m['bh'] * a
    if m['kind'] == 'shift': return L0 + m['delta']
    if m['kind'] == 'affine': return m['medpf'] + m['a'] * (L0 - m['medL'])


for key in models:
    ens = key.split('|')[0]
    res, h, a, k = P.load(ens); M = P.pmap(res); u = np.ones(h.shape[1]) / h.shape[1]
    pf = P.pfact_vec(res); ok = pf > 0
    L = lnpf_model(ens, key, h, a); Lm = L.mean(1)
    row = dict(ensemble=ens, model=key.split('|')[1], median_set=key.split('|')[2],
               param=json.dumps({kk: round(v, 4) for kk, v in models[key].items() if kk != 'kind'}),
               median_lnPF_matched=np.median(Lm[ok]), mean_lnPF_res=Lm.mean())
    for (lo, hi) in P.REGIONS:
        rm = ok & (res >= lo) & (res <= hi)
        row[f'medres_{lo}-{hi}'] = np.median(pf[rm] - Lm[rm]); row[f'n_{lo}-{hi}'] = int(rm.sum())
    row['medres_all'] = np.median(pf[ok] - Lm[ok])
    for mode, f in P.PRED.items():
        sc = P.scores(f(L, k, M, u))
        row[f'{mode}_meanres'] = float(sc['mean_res']); row[f'{mode}_frac_exp>pred'] = float(sc['frac_exp_gt_pred'])
        row[f'{mode}_mse'] = float(sc['mse']); row[f'{mode}_chi2pool'] = float(sc['chi2_pool'])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(P.OUT + 'p1_t1.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.drop(columns=[c for c in df if c.startswith('n_')]).round(3).to_string())
json.dump(models, open(P.OUT + 'p1_models.json', 'w'), indent=1)

# ---- framework check of rate mode (and frame_uptake) on the scaled model ---------------
from types import SimpleNamespace
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jaxent.src.models.HDX.forward import BV_uptake_ForwardPass
for ens in P.ENSS:
    res, h, a, k = P.load(ens); M = P.pmap(res); nF = h.shape[1]
    rng = np.random.default_rng(0); w = rng.dirichlet(np.ones(nF) * 0.3)
    m = models[f'{ens}|scale|matched']
    feat = SimpleNamespace(heavy_contacts=h, acceptor_contacts=a, k_ints=k)
    par = SimpleNamespace(bv_bc=m['bc'], bv_bh=m['bh'], timepoints=P.TIMES)
    for mode in ['rate', 'frame_uptake']:
        fw = np.asarray(BV_uptake_ForwardPass(frame_averaging_mode=mode).average_frames(feat, par, jnp.asarray(w)).uptake)  # (T,res)
        fw_pep = M @ fw.T
        mine = P.PRED[mode](m['bc'] * h + m['bh'] * a, k, M, w)
        print(f'framework check {ens} {mode}: max|diff| = {np.abs(fw_pep - mine).max():.2e}')
