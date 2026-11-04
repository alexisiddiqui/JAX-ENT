"""T1 add-on (b'): additive shift with physical floor lnPF >= 0 (PF >= 1, k <= k_int); delta found by 1-D root find so that
median over matched residues of the uniform-mean clamped lnPF equals median(pfact>0). Appends to p1_models.json; -> p1c.log"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import brentq
import p0_common as P
models = json.load(open(P.OUT + 'p1_models.json'))
for ens in P.ENSS:
    res, h, a, k = P.load(ens); M = P.pmap(res); u = np.ones(h.shape[1]) / h.shape[1]
    pf = P.pfact_vec(res); ok = pf > 0; target = np.median(pf[ok]); L0 = P.BC0 * h + P.BH0 * a
    f = lambda d: np.median(np.maximum(L0 + d, 0).mean(1)[ok]) - target
    d = brentq(f, -10, 5, xtol=1e-12)
    models[f'{ens}|shift_clamp0|matched'] = dict(kind='shift_clamp0', delta=d)
    L = np.maximum(L0 + d, 0); Lm = L.mean(1)
    reg = {f'{lo}-{hi}': round(float(np.median((pf - Lm)[ok & (res >= lo) & (res <= hi)])), 3) for lo, hi in P.REGIONS}
    out = {m: P.scores(fn(L, k, M, u)) for m, fn in P.PRED.items()}
    print(f"{ens}: delta_clamp0 = {d:.4f} (unclamped {models[f'{ens}|shift|matched']['delta']:.4f}); frac clamped {np.mean(L0 + d < 0):.3f}; "
          f"region median residuals {reg}; all {np.median((pf - Lm)[ok]):.3f}; "
          + '; '.join(f"{m}: mean res {float(s['mean_res']):+.3f}, frac exp>pred {float(s['frac_exp_gt_pred']):.3f}, chi2/cell {float(s['chi2_pool']):.2f}" for m, s in out.items()))
json.dump(models, open(P.OUT + 'p1_models.json', 'w'), indent=1)
