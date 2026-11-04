"""T2 figures. -> F1_face_{ens}.png (PUF1 x PUF2 face, decoys 0, F = 1-PUF1-PUF2; dashed line = the requested PUF2 = 0.5% slice),
F2_profile_F.png (F 0.5->1 at target PUF1:PUF2 ratio, decoys 0), F3_slices.png (PUF1 at PUF2=0.5%; decoy mass added to target),
p2_profiles.csv"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import p0_common as P, p2_landscape as L

MOD = {'scale|matched': ('(a)=(c) scale', '#2a78d6'), 'shift|matched': ('(b) shift', '#eb6834'), 'shift_clamp0|matched': ("(b') shift, lnPF>=0", '#eda100'),
       'affine_iqr|matched': ("(c') affine-IQR", '#1baf7a'), 'none|-': ('unscaled', '#52514e')}
LS = {'rate': '-', 'frame_uptake': '--'}
PAN = ['recovery', 'mse', 'wr2', 'chi2_pool', 'maxz', 'spearman']
TIT = dict(recovery='recovery %', mse='MSE', wr2='mean(w r²)', chi2_pool='χ²/cell (pooled σ)', maxz='max|z|', spearman='Spearman ρ (higher better)')
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.facecolor': '#fcfcfb', 'figure.facecolor': '#fcfcfb'})

tabs = {(e, k): L.state_tables(e, f'{e}|{k}') for e in P.ENSS for k in MOD}


def full(tab, core):  # core (n,3) F,PUF1,PUF2 -> (n,d) with decoys 0
    d = len(tab['names']); out = np.zeros((len(core), d)); out[:, :3] = core; return out


tg = np.array([P.C.TARGET[n] for n in L.CORE]); r12 = tg[1] / (tg[1] + tg[2])
# ---- F1 face ----
x1 = np.arange(0, 0.3001, 0.005); x2 = np.arange(0, 0.0601, 0.001)
X1, X2 = np.meshgrid(x1, x2); core = np.c_[1 - X1.ravel() - X2.ravel(), X1.ravel(), X2.ravel()]
for ens in P.ENSS:
    rows = [(k, m) for k in ['scale|matched', 'shift|matched', 'shift_clamp0|matched', 'affine_iqr|matched'] for m in ['rate', 'frame_uptake']]
    fig, ax = plt.subplots(len(rows), 6, figsize=(17, 2.3 * len(rows)), constrained_layout=True)
    for i, (k, m) in enumerate(rows):
        tab = tabs[(ens, k)]; ev = L.evaluate(tab, full(tab, core), m)
        for j, s in enumerate(PAN):
            v = ev[s].reshape(X1.shape); better_high = s in ('recovery', 'spearman')
            im = ax[i, j].pcolormesh(x1 * 100, x2 * 100, v, cmap='Blues' if better_high else 'Blues_r', shading='nearest')
            fig.colorbar(im, ax=ax[i, j], shrink=0.9)
            b = np.argmax(v) if better_high else np.argmin(v)
            ax[i, j].plot(X1.ravel()[b] * 100, X2.ravel()[b] * 100, 'x', color='#e34948', ms=8, mew=2, label='best on face')
            ax[i, j].plot(tg[1] * 100, tg[2] * 100, '*', color='#0b0b0b', ms=10, label='target')
            ax[i, j].axhline(0.5, color='#0b0b0b', lw=0.6, ls=':')
            ax[i, j].set_title(f'{TIT[s]}' + (f'\n{MOD[k][0]}, {m}' if j == 0 else ''), fontsize=8)
            if j == 0: ax[i, j].set_ylabel('PUF2 %')
            if i == len(rows) - 1: ax[i, j].set_xlabel('PUF1 %')
    ax[0, 0].legend(fontsize=6, loc='upper right')
    fig.suptitle(f'{ens}: (F, PUF1, PUF2) face, decoy states 0. ★ target, × best on face, dotted = PUF2 0.5% slice  [p2_fig]')
    fig.savefig(P.OUT + f'F1_face_{ens}.png', dpi=110); plt.close(fig)

# ---- profiles ----
prof = []
Fs = np.linspace(0.5, 1.0, 101)
coreF = np.c_[Fs, (1 - Fs) * r12, (1 - Fs) * (1 - r12)]
p1 = np.linspace(0, 0.3, 121); coreP1 = np.c_[0.995 - p1, p1, np.full_like(p1, 0.005)]
dm = np.linspace(0, 0.3, 121)


def run_profiles(name, xs, getP):
    for ens in P.ENSS:
        for k in MOD:
            tab = tabs[(ens, k)]; Pm = getP(tab)
            for m in P.PRED:
                ev = L.evaluate(tab, Pm, m)
                for i, x in enumerate(xs):
                    prof.append(dict(profile=name, ensemble=ens, model=k, mode=m, x=x, **{s: ev[s][i] for s in PAN}))


def decoyP(tab):
    asg = tab['asg']; cnt = np.array([(asg == s).sum() for s in tab['present']], float)
    dec = np.zeros(len(cnt)); dec[3:] = cnt[3:] / cnt[3:].sum()
    t = np.array([P.C.TARGET[n] for n in tab['names']]); t /= t.sum()
    return (1 - dm)[:, None] * t[None] + dm[:, None] * dec[None]


run_profiles('F', Fs, lambda t: full(t, coreF))
run_profiles('PUF1@PUF2=0.5%', p1, lambda t: full(t, coreP1))
run_profiles('decoy_mass', dm, decoyP)
D = pd.DataFrame(prof); D.to_csv(P.OUT + 'p2_profiles.csv', index=False)


def plot_prof(names, fname, xl):
    fig, ax = plt.subplots(len(names) * 2, 6, figsize=(17, 2.6 * 2 * len(names)), constrained_layout=True)
    for a, (pname, xlab) in enumerate(zip(names, xl)):
        for e, ens in enumerate(P.ENSS):
            r = 2 * a + e
            for j, s in enumerate(PAN):
                for k, (lab, col) in MOD.items():
                    for m in P.PRED:
                        d = D[(D.profile == pname) & (D.ensemble == ens) & (D.model == k) & (D['mode'] == m)]
                        if s == 'recovery' and (k != 'scale|matched' or m != 'rate'): continue
                        ax[r, j].plot(d.x * 100, d[s], LS[m], color=col if s != 'recovery' else '#0b0b0b', lw=1.5, label=f'{lab}, {m}')
                if pname == 'F': ax[r, j].axvline(tg[0] * 100, color='#52514e', lw=0.6, ls=':')
                if pname.startswith('PUF1'): ax[r, j].axvline(tg[1] * 100, color='#52514e', lw=0.6, ls=':')
                if s in ('mse', 'wr2', 'chi2_pool'): ax[r, j].set_yscale('log')
                ax[r, j].set_title(f'{TIT[s]}' + (f'\n{ens}' if j == 0 else ''), fontsize=8)
                ax[r, j].set_xlabel(xlab)
            if a == 0 and e == 0: ax[r, 1].legend(fontsize=6)
    fig.suptitle(f'profiles (dotted = target; recovery depends only on populations)  [p2_fig]')
    fig.savefig(P.OUT + fname, dpi=110); plt.close(fig)


plot_prof(['F'], 'F2_profile_F.png', ['Folded % (PUF1:PUF2 at target ratio, decoys 0)'])
plot_prof(['PUF1@PUF2=0.5%', 'decoy_mass'], 'F3_slices.png', ['PUF1 % (PUF2 = 0.5%, F = 99.5 − PUF1)', 'decoy mass % mixed into target'])

# argmin of each profile
out = []
for (pr, ens, k, m), d in D.groupby(['profile', 'ensemble', 'model', 'mode']):
    row = dict(profile=pr, ensemble=ens, model=k, mode=m)
    for s in PAN[1:]:
        i = d[s].idxmax() if s == 'spearman' else d[s].idxmin()
        row[f'x*_{s}'] = round(d.x[i] * 100, 2); row[f'rec*_{s}'] = round(d.recovery[i], 1)
    out.append(row)
O = pd.DataFrame(out); O.to_csv(P.OUT + 'p2_profile_optima.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 30)
print(O.to_string())
