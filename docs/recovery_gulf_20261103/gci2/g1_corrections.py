"""Re-creates corrections.npz (global, smooth5, bcbh_wr2 'bcbh_all') exactly as gci/g1_corrections.py of the previous pass.
-> corrections.npz, g1_constants.csv"""
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import numpy as np, pandas as pd
from scipy.optimize import minimize
import g0_common as G
tab, const = {}, []
for ens in G.ENSS:
    res, h, a, k = G.load(ens)
    L = G.BC0 * h + G.BH0 * a
    pf = np.array([G.PF.get(int(r), -1.0) for r in res]); ok = pf > 0
    d = pf - L.mean(1)
    tab[f'{ens}|global'] = np.full(len(res), np.median(d[ok]))
    K = np.exp(-0.5 * ((res[:, None] - res[None, ok]) / 5) ** 2)
    tab[f'{ens}|smooth5'] = K @ d[ok] / K.sum(1)
    M = G.pmap(res)
    for name, W in [('bcbh_all', G.WTS), ('bcbh_all_mse', np.ones_like(G.WTS))]:
        f = lambda p: np.mean(W * (G.DEXP - G.frame_uptake(p[0] * h + p[1] * a, k, M).mean(0)) ** 2)
        best = None
        for x0 in [(0.35, 2.0), (0.2, 1.0), (0.1, 0.5), (0.3, 0.0), (0.0, 1.5)]:
            r = minimize(f, x0, method='L-BFGS-B', bounds=[(0, 2), (0, 10)])
            if best is None or r.fun < best.fun: best = r
        tab[f'{ens}|{name}'] = best.x
        const.append(dict(ensemble=ens, corr=name, bc=best.x[0], bh=best.x[1], obj=best.fun))
    const.append(dict(ensemble=ens, corr='global', delta=np.median(d[ok])))
    tab[f'{ens}|res'] = res
np.savez(G.OUT + 'corrections.npz', **tab)
print(pd.DataFrame(const).to_string())
