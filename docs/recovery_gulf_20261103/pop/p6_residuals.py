"""Per-peptide residual (exp - pred, mean over 15 timepoints) and max|z| cell at the target populations, rate vs frame_uptake. -> p6_residuals.csv, p6.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import p0_common as P, p2_landscape as L2
TL = ['5s', '20s', '40s', '1m', '5m', '10m', '20m', '30m', '45m', '1h', '2.7h', '4h', '6.5h', '12.5h', '24h']
rows = []
for ens in P.ENSS:
    for key in ['scale|matched', 'shift_clamp0|matched', 'shift|matched']:
        tab = L2.state_tables(ens, f'{ens}|{key}')
        t = np.array([P.C.TARGET[n] for n in tab['names']]); t /= t.sum()
        for mode in P.PRED:
            pr = L2.predict(tab, t[None], mode)[0]; r = P.DEXP - pr; z = r / P.SIG_POOL
            i = np.unravel_index(np.abs(z).argmax(), z.shape)
            row = dict(ensemble=ens, model=key, mode=mode, maxz_cell=f'{P.LAB[i[0]]}@{TL[i[1]]}', maxz=round(float(z[i]), 1))
            row.update({P.LAB[p]: round(float(r[p].mean()), 3) for p in range(P.NP)})
            rows.append(row)
D = pd.DataFrame(rows); D.to_csv(P.OUT + 'p6_residuals.csv', index=False)
pd.set_option('display.width', 320); pd.set_option('display.max_columns', 30); print(D.to_string())
