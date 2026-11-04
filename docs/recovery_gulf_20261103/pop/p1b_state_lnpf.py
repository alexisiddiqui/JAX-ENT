"""T1 add-on: per-state mean lnPF and fraction of residue-frames with lnPF<0 (PF<1, k>k_int) under each model. -> p1b.csv"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import p0_common as P, p2_landscape as L2
rows = []
for ens in P.ENSS:
    res, h, a, k = P.load(ens); asg = P.assign(ens)
    for key in L2.USE:
        Lf = L2.lnpf(f'{ens}|{key}', h, a)
        for s in np.unique(asg):
            m = asg == s; Ls = Lf[:, m]
            kbar = (k[:, None] * np.exp(-Ls)).mean(1)
            rows.append(dict(ensemble=ens, model=key, state=P.STATES[s], n=int(m.sum()), mean_lnPF=Ls.mean(), median_res_lnPF=np.median(Ls.mean(1)),
                             frac_lnPF_lt0=(Ls < 0).mean(), median_res_ln_kbar_over_kint=np.median(np.log(kbar / k))))
D = pd.DataFrame(rows); D.to_csv(P.OUT + 'p1b.csv', index=False)
pd.set_option('display.width', 250); print(D.round(3).to_string())
