"""Auditable tables and standalone plots for the ISO OMC experiment."""
from __future__ import annotations

import html
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

e = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_control')
COLORS = dict(zip(e.FAMILIES, ('black', '#a05a2c', '#0072b2', '#009e73')))


def nearest_ess(table):
    rows = []
    valid = table.loc[table.converged.astype(bool)]
    for (stage, ensemble), group in valid.groupby(['stage', 'ensemble']):
        controls = group.loc[group.family == 'maxent']
        for _, row in group.loc[group.family.isin(e.FAMILIES[2:])].iterrows():
            if controls.empty:
                continue
            other = controls.loc[(controls.ess_fraction-row.ess_fraction).abs().idxmin()]
            gap = abs(other.ess_fraction-row.ess_fraction)
            rows.append(dict(stage=stage, ensemble=ensemble, family=row.family, arm=row.arm,
                maxent_arm=other.arm, ess_gap=gap, matched=gap <= .02,
                delta_tv=row.tv-other.tv if gap <= .02 else np.nan,
                delta_mse=row.mse-other.mse if gap <= .02 else np.nan))
    return pd.DataFrame(rows)


def build(output):
    output = Path(output)
    e.load_manifest(output)
    rows = e.collect(output)
    if not rows:
        raise ValueError('No completed fits to report')
    table = pd.DataFrame(rows)
    selected = e.select(rows)
    residue_keys = e.layout(e.ROOT / 'data/_self_consistent_target_features/topology_open_closed.json')
    residuals = []
    population_rows = []
    for row in rows:
        if row.get('error'):
            continue
        folder = output / row['stage'] / row['ensemble'] / f"arm_{row['arm']:02d}"
        result = np.load(folder / 'fit.npz')
        data = np.load(output / row['ensemble'] / 'input.npz')
        weights = result['weights']
        np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
        prediction = e.predict(data['rates'], weights, data['groups'], row['stage'])
        np.testing.assert_allclose(prediction, result['prediction'], atol=1e-12, rtol=1e-10)
        mse = float(np.mean((prediction-data['target'])**2))
        np.testing.assert_allclose(mse, row['mse'], rtol=1e-10, atol=1e-14)
        penalty = 0.
        if row['family'] == 'maxent':
            penalty = float(np.mean(-np.log(len(weights)*weights)))
        elif row['family'] in e.FAMILIES[2:]:
            i = int(np.flatnonzero(e.QUANTILES == row['quantile'])[0])
            kernel = np.load(output / row['ensemble'] / f"{row['family']}_{i}.npy", mmap_mode='r')
            centered = weights - 1/len(weights)
            penalty = max(0., len(weights)**2 * (np.dot(weights*centered**2, kernel@weights)-
                      np.dot(weights*centered, kernel@(weights*centered))))
        objective = mse / float(data['scale']) + row['strength']*penalty
        np.testing.assert_allclose(objective, row['objective'], rtol=1e-7, atol=1e-10)
        for t, time in enumerate(e.TIMES):
            for r in range(prediction.shape[1]):
                residuals.append(dict(stage=row['stage'], ensemble=row['ensemble'], family=row['family'],
                    arm=row['arm'], residue_index=r, chain=residue_keys[r][0], residue=residue_keys[r][1][0], time_min=time, target=data['target'][t,r],
                    prediction=prediction[t,r], residual=prediction[t,r]-data['target'][t,r]))
        for start, w in enumerate(result['initial_weights']):
            population_rows.append(dict(stage=row['stage'], ensemble=row['ensemble'], family=row['family'],
                arm=row['arm'], start=start, **e.metrics(w, data['groups'])))
    table.to_csv(output / 'fits.csv', index=False)
    selected.to_csv(output / 'selected.csv', index=False)
    pd.DataFrame(residuals).to_csv(output / 'residuals.csv', index=False)
    pd.DataFrame(population_rows).to_csv(output / 'initialisation_populations.csv', index=False)
    nearest_ess(table).to_csv(output / 'nearest_ess.csv', index=False)
    graphs = pd.concat([pd.DataFrame(json.loads((output / ensemble / 'graphs.json').read_text())).assign(ensemble=ensemble)
                        for ensemble in ('ISO_BI', 'ISO_TRI')])
    graphs.to_csv(output / 'graphs.csv', index=False)
    if set(table.stage) == {'rate', 'uptake'}:
        paired = table.loc[table.stage == 'uptake'].merge(table.loc[table.stage == 'rate'],
            on=['ensemble','family','arm'], suffixes=('_grouped','_rate'))
        paired['valid_pair'] = paired.converged_grouped & paired.converged_rate
        for metric in ('recovery', 'tv', 'mse', 'ess_fraction'):
            paired['delta_'+metric] = (paired[metric+'_rate']-paired[metric+'_grouped']).where(paired.valid_pair)
        paired.to_csv(output / 'paired_stages.csv', index=False)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    figure_names = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(figures / f'{name}.{suffix}', dpi=160)
        plt.close(fig)
        figure_names.append(name)
    for (stage, ensemble), group in table.groupby(['stage','ensemble']):
        data = np.load(output / ensemble / 'input.npz')
        chosen = selected.loc[(selected.stage == stage) & (selected.ensemble == ensemble)]
        fig, axes = plt.subplots(1, 3, figsize=(16,4.5))
        for family, values in group.groupby('family', sort=False):
            finite = values.loc[np.isfinite(values.mse)]
            for ax, metric, label in zip(axes[:2], ('tv','mse'), ('Population TV error', 'All-residue MSE')):
                ax.plot(finite.ess_fraction*100, finite[metric], '.-', color=COLORS[family], label=family)
                invalid = finite.loc[~finite.converged.astype(bool)]
                ax.scatter(invalid.ess_fraction*100, invalid[metric], marker='x', s=70, color='red')
                ax.set(xlabel='ESS fraction (%)', ylabel=label)
        labels = ['Target', 'Initial'] + chosen.family.tolist()
        bars = [e.TRUTH, e.populations(np.ones(len(data['groups']))/len(data['groups']),data['groups'])]
        bars += [np.array([r.open,r.closed,r.intermediate]) for r in chosen.itertuples()]
        bars = np.array(bars)
        bottom = np.zeros(len(bars))
        for k, name in enumerate(('Open','Closed','Intermediate')):
            axes[2].bar(np.arange(len(bars)), bars[:,k], bottom=bottom, label=name)
            bottom += bars[:,k]
        axes[2].set_xticks(np.arange(len(bars)), labels, rotation=35, ha='right')
        axes[2].set(ylabel='Population fraction', ylim=(0,1))
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=8)
        fig.suptitle(f'{ensemble}: {stage} fitting; red crosses are unresolved')
        save(fig, f'{stage}_{ensemble}_comparison')
        fig, axes = plt.subplots(2, 3, figsize=(15,8))
        for column, (state, name) in enumerate(zip(e.STATES, ('Open','Closed','Intermediate'))):
            mask = data['groups'] == state
            if not mask.any():
                for ax in axes[:,column]:
                    ax.set_visible(False)
                continue
            for axis, coordinate in enumerate(('Open-reference RMSD (Å)', 'Closed-reference RMSD (Å)')):
                ax = axes[axis,column]
                x = data['rmsd'][mask, axis]
                order = np.argsort(x)
                ax.plot(x[order], np.arange(1,len(x)+1)/len(x), '--', color='grey', label='Initial within-state')
                for row in chosen.itertuples():
                    w = np.load(output / stage / ensemble / f'arm_{row.arm:02d}/fit.npz')['weights'][mask]
                    if w.sum() > 0:
                        ax.plot(x[order], np.cumsum(w[order])/w.sum(), color=COLORS[row.family], label=row.family)
                ax.set(xlabel=coordinate, ylabel='Conditional cumulative weight', title=name, ylim=(0,1))
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{ensemble}: {stage}, within-state structural coverage')
        save(fig, f'{stage}_{ensemble}_coverage')
        fig, axes = plt.subplots(max(1,len(chosen)), 1, figsize=(13,2.3*max(1,len(chosen))), squeeze=False)
        for ax, row in zip(axes[:,0], chosen.itertuples()):
            prediction = np.load(output / stage / ensemble / f'arm_{row.arm:02d}/fit.npz')['prediction']
            im = ax.imshow(prediction-data['target'], aspect='auto', cmap='coolwarm')
            ax.set(title=row.family, xlabel='Residue number', ylabel='Time (min)')
            ticks = np.arange(0, len(residue_keys), 40)
            ax.set_xticks(ticks, [residue_keys[i][1][0] for i in ticks])
            ax.set_yticks(np.arange(len(e.TIMES)), [f'{t:g}' for t in e.TIMES])
            fig.colorbar(im, ax=ax, label='Predicted − target uptake')
        save(fig, f'{stage}_{ensemble}_residuals')
    decision = json.loads((output / 'gate.json').read_text()) if (output / 'gate.json').exists() else {'passed':False, 'status':'not evaluated'}
    preflight = pd.read_csv(output / 'preflight.csv')
    columns = ['stage','ensemble','family','quantile','strength','recovery','tv','mse','ess_fraction','open','closed','intermediate',
               'open_conditional_ess','closed_conditional_ess','intermediate_conditional_ess']
    body = ['<!doctype html><html><head><meta charset="utf-8"><title>ISO OMC comparison</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}.scroll{overflow:auto}pre{white-space:pre-wrap}</style></head><body>',
        '<h1>ISO population recovery and distribution control</h1>',
        '<p>Target: frame-wise uptake, open 40%, closed 60%, intermediate 0%. Existing candidates retained; hard contacts, matching JAX-ENT intrinsic rates. Grouped fitting averages exchange rates within fixed known conformational groups; mean-rate fitting averages rates across the whole candidate. All-residue MSE selects models; population truth does not select parameters.</p>',
        '<p>Higher ESS alone is not evidence of recovery. Recovery = 100 × (1 − √base-2 JSD), including intermediate population. The 50% gate is an absolute score, not improvement over initial weights. Grouped fitting benefits from known state assignments. Nearest-ESS comparisons allow a two-percentage-point gap and use no interpolation.</p>',
        f'<p>Completed configurations: {len(table)}; converged: {int(table.converged.sum())}. Mean-rate gate: <strong>{decision["passed"]}</strong>.</p>',
        '<h2>Automatic progression decision</h2><pre>'+html.escape(json.dumps(decision,indent=2))+'</pre>',
        '<h2>MSE-selected converged fits</h2><div class="scroll">'+selected.reindex(columns=columns).to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<h2>Forward approximation and starting populations</h2><div class="scroll">'+preflight.to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<h2>Downloads</h2><p>']
    for path in sorted(output.glob('*.csv')):
        body.append(f'<a href="{path.name}">{path.name}</a> · ')
    body += ['<a href="manifest.json">manifest</a></p>']
    if (output / 'reference_contact_audit.json').exists():
        body.append('<p><a href="reference_contact_audit.json">Reference contact parity audit</a></p>')
    body.append('<h2>All configurations and convergence</h2><div class="scroll">' + table.to_html(index=False, float_format=lambda x:f'{x:.6g}') + '</div>')
    for name in figure_names:
        body.append(f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">')
    body.append('</body></html>')
    (output / 'index.html').write_text('\n'.join(body))
    print(output / 'index.html', flush=True)
