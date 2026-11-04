"""check state-level tables (p2) against frame-level forward passes + repo recovery function. -> p2_check.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p0_common as P, p2_landscape as L
rng = np.random.default_rng(1)
for ens in P.ENSS:
    res, h, a, k = P.load(ens); M = P.pmap(res)
    for key in L.USE:
        tab = L.state_tables(ens, f'{ens}|{key}'); asg = tab['asg']
        Lf = L.lnpf(f'{ens}|{key}', h, a)
        Pm = rng.dirichlet(np.ones(len(tab['present'])), 5)
        for mode in P.PRED:
            st = L.predict(tab, Pm, mode); er, rr = 0, 0
            for i, p in enumerate(Pm):
                w = np.zeros(len(asg))
                for s, ps in zip(tab['present'], p): w[asg == s] = ps / (asg == s).sum()
                er = max(er, np.abs(P.PRED[mode](Lf, k, M, w) - st[i]).max())
                rr = max(rr, abs(P.recovery(asg, w) - L.P.recovery_p6(L.to6(tab, p[None]))[0]))
            print(f'{ens} {key} {mode}: max|state-frame pred diff| {er:.1e}, max|recovery diff| {rr:.1e}')
