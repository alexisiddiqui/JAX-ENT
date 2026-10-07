"""Population, coverage and connectivity diagnostics for hybrid ISO OMC."""
from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

h = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_control')
e = h.e
COLORS = dict(zip((*e.FAMILIES, h.FAMILY), ('black', '#a05a2c', '#0072b2', '#009e73', '#cc79a7')))


def pairwise_penalty(weights, kernel):
    """Explicit pairwise definition, evaluated in blocks to bound memory."""
    n = len(weights)
    total = 0.
    for start in range(0, n, 128):
        w = weights[start:start+128, None]
        total += np.sum(w*weights[None, :]*kernel[start:start+128]*(w-weights[None, :])**2)
    return float(.5*n*n*total)


def comparisons(table):
    hybrids = table.loc[table.family == h.FAMILY]
    pairs, matched = [], []
    for _, row in hybrids.iterrows():
        for family in ('scalar_logpf', 'profile_logpf', 'maxent'):
            controls = table.loc[(table.ensemble == row.ensemble) & (table.family == family)]
            if family != 'maxent':
                control = controls.loc[controls['quantile'] == row['quantile']].iloc[0]
                valid = bool(row.converged and control.converged)
                pairs.append(dict(ensemble=row.ensemble, quantile=row['quantile'], control=family,
                    valid_pair=valid, **{'delta_'+metric: row[metric]-control[metric] if valid else np.nan
                    for metric in ('recovery', 'tv', 'mse', 'ess_fraction', 'intermediate')}))
            valid_controls = controls.loc[controls.converged.astype(bool)]
            base = dict(ensemble=row.ensemble, arm=row.arm, quantile=row['quantile'], control=family)
            if not row.converged or valid_controls.empty:
                matched.append(dict(**base, matched=False, reason='unconverged or no converged control'))
                continue
            control = valid_controls.loc[(valid_controls.ess_fraction-row.ess_fraction).abs().idxmin()]
            gap = abs(control.ess_fraction-row.ess_fraction)
            valid = bool(gap <= .02)
            matched.append(dict(**base, matched=valid, reason='within tolerance' if valid else 'outside ESS tolerance',
                control_arm=control.arm, ess_gap=gap,
                **{'delta_'+metric: row[metric]-control[metric] if valid else np.nan
                   for metric in ('recovery', 'tv', 'mse', 'intermediate')}))
    return pd.DataFrame(pairs), pd.DataFrame(matched)


def build(output):
    output = Path(output)
    h.validate(output)
    rows = h.collect(output)
    table = pd.DataFrame(rows)
    selected = e.select(rows)
    initial_rows, residual_rows, component_rows = [], [], []
    residue_keys = e.layout(output / 'source/topology.json')
    data_by_ensemble = {name: dict(np.load(output / name / 'input.npz')) for name in ('ISO_BI', 'ISO_TRI')}
    for row in rows:
        if row.get('error'):
            continue
        data = data_by_ensemble[row['ensemble']]
        result = np.load(h.archive(output, row))
        weights = result['weights']
        components = np.load(output / row['ensemble'] / 'neighbourhood.npz')['components']
        for component in np.unique(components):
            mask = components == component
            component_rows.append(dict(ensemble=row['ensemble'], family=row['family'], arm=row['arm'],
                component=int(component), population=float(weights[mask].sum()),
                uniform_population=float(mask.mean()),
                true_population_diagnostic=float(data['truth_weights'][mask].sum())))
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1), 1., atol=1e-12)
        if not np.isfinite(result['initial_weights']).all() or (result['initial_weights'] < 0).any():
            raise ValueError('Invalid archived weights')
        prediction = e.predict(data['rates'], weights, data['groups'], 'uptake')
        np.testing.assert_allclose(prediction, result['prediction'], rtol=1e-10, atol=1e-12)
        mse = float(np.mean((prediction-data['target'])**2))
        penalty = 0.
        if row['family'] == 'maxent':
            penalty = float(np.mean(-np.log(len(weights)*weights)))
        elif row['family'] != 'unregularised':
            i = int(np.flatnonzero(e.QUANTILES == row['quantile'])[0])
            kernel = np.load(output / row['ensemble'] / f"{row['family']}_{i}.npy", mmap_mode='r')
            penalty = pairwise_penalty(weights, kernel)
        np.testing.assert_allclose(mse, row['mse'], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(mse/float(data['scale'])+row['strength']*penalty,
                                   row['objective'], rtol=1e-7, atol=1e-10)
        for start, w in enumerate(result['initial_weights']):
            initial_rows.append(dict(ensemble=row['ensemble'], family=row['family'], arm=row['arm'],
                start=start, **e.metrics(w, data['groups'])))
        for t, time in enumerate(e.TIMES):
            for r, (chain, residues) in enumerate(residue_keys):
                residual_rows.append(dict(ensemble=row['ensemble'], family=row['family'], arm=row['arm'],
                    chain=chain, residue=residues[0], time_min=time, target=data['target'][t,r],
                    prediction=prediction[t,r], residual=prediction[t,r]-data['target'][t,r]))
    pairs, matched = comparisons(table)
    table.to_csv(output / 'fits.csv', index=False)
    selected.to_csv(output / 'selected.csv', index=False)
    pairs.to_csv(output / 'paired_bandwidth.csv', index=False)
    matched.to_csv(output / 'nearest_ess.csv', index=False)
    pd.DataFrame(initial_rows).to_csv(output / 'initialisation_populations.csv', index=False)
    pd.DataFrame(residual_rows).to_csv(output / 'residuals.csv', index=False)
    pd.DataFrame(component_rows).to_csv(output / 'component_populations.csv', index=False)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    names = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(figures / f'{name}.{suffix}', dpi=160)
        plt.close(fig)
        names.append(name)
    for ensemble, data in data_by_ensemble.items():
        group = table.loc[table.ensemble == ensemble]
        chosen = selected.loc[selected.ensemble == ensemble]
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        for family, values in group.groupby('family'):
            finite = values.loc[np.isfinite(values.mse)]
            for ax, metric, label in zip(axes[:3], ('recovery','tv','mse'), ('Recovery score (%)','Population TV error','All-residue MSE')):
                ax.plot(finite.ess_fraction*100, finite[metric], '.-', label=family, color=COLORS[family])
                invalid = finite.loc[~finite.converged.astype(bool)]
                ax.scatter(invalid.ess_fraction*100, invalid[metric], marker='x', s=65, color='red')
                ax.set(xlabel='ESS fraction (%)', ylabel=label)
        axes[0].legend(fontsize=7)
        uniform = np.ones(len(data['groups']))/len(data['groups'])
        bars = np.array([e.TRUTH, e.populations(uniform, data['groups'])] +
                        [np.array([r.open,r.closed,r.intermediate]) for r in chosen.itertuples()])
        bottom = np.zeros(len(bars))
        for i, name in enumerate(('Open','Closed','Intermediate')):
            axes[3].bar(np.arange(len(bars)), bars[:,i], bottom=bottom, label=name)
            bottom += bars[:,i]
        axes[3].set_xticks(np.arange(len(bars)), ['Target','Initial']+chosen.family.tolist(), rotation=40, ha='right')
        axes[3].set(ylabel='Population', ylim=(0,1))
        axes[3].legend(fontsize=7)
        fig.suptitle(f'{ensemble}: grouped uptake; red crosses denote unresolved fits')
        save(fig, ensemble+'_comparison')
        fig, axes = plt.subplots(2, 3, figsize=(15,8))
        for column, (state, name) in enumerate(zip(e.STATES, ('Open','Closed','Intermediate'))):
            mask = data['groups'] == state
            if not mask.any():
                for ax in axes[:,column]:
                    ax.set_visible(False)
                continue
            for coordinate, ref in enumerate(('Open','Closed')):
                ax = axes[coordinate,column]
                x = data['rmsd'][mask,coordinate]
                order = np.argsort(x)
                ax.plot(x[order], np.arange(1,len(x)+1)/len(x), '--', color='grey', label='Initial')
                for row in chosen.to_dict('records'):
                    weights = np.load(h.archive(output,row))['weights'][mask]
                    if weights.sum() > 0:
                        ax.plot(x[order], np.cumsum(weights[order])/weights.sum(), label=row['family'], color=COLORS[row['family']])
                ax.set(title=name, xlabel=f'{ref}-reference RMSD (Å)', ylabel='Conditional cumulative weight', ylim=(0,1))
        axes[0,0].legend(fontsize=7)
        fig.suptitle(ensemble+': within-state structural coverage')
        save(fig, ensemble+'_coverage')
        neighbourhood = np.load(output / ensemble / 'neighbourhood.npz')
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        axes[0].hist(neighbourhood['degree'], bins=30)
        axes[0].set(xlabel='Symmetric neighbour count',ylabel='Frames')
        axes[1].scatter(data['rmsd'][:,0],data['rmsd'][:,1],c=neighbourhood['components'],s=8,cmap='tab10')
        axes[1].set(xlabel='Open-reference RMSD (Å)',ylabel='Closed-reference RMSD (Å)',title='Colour denotes graph component')
        fig.suptitle(ensemble+': profile-neighbourhood geometry')
        save(fig, ensemble+'_geometry')
        fig, axes = plt.subplots(len(chosen),1,figsize=(13,2.2*len(chosen)),squeeze=False)
        residual_arrays = [np.load(h.archive(output,row))['prediction']-data['target'] for row in chosen.to_dict('records')]
        limit = max(float(abs(a).max()) for a in residual_arrays)
        for ax, row, residual in zip(axes[:,0],chosen.to_dict('records'),residual_arrays):
            im = ax.imshow(residual,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
            ticks = np.arange(0,len(residue_keys),40)
            ax.set_xticks(ticks,[residue_keys[i][1][0] for i in ticks])
            ax.set_yticks(np.arange(len(e.TIMES)),[f'{t:g}' for t in e.TIMES])
            ax.set(title=row['family'],xlabel='Residue number',ylabel='Time (min)')
            fig.colorbar(im,ax=ax,label='Predicted − target uptake')
        save(fig,ensemble+'_residuals')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    columns = ['ensemble','family','quantile','strength','open','closed','intermediate','recovery','ess_fraction','mse',
               'open_conditional_ess','closed_conditional_ess','intermediate_conditional_ess']
    body = ['<!doctype html><html><head><meta charset="utf-8"><title>Hybrid ISO OMC</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}.scroll{overflow:auto}img{width:100%}</style></head><body>',
        '<h1>Hybrid OMC: profile neighbours, scalar coupling</h1>',
        '<p>Frame-wise target uptake; 40% open, 60% closed, 0% intermediate. Grouped fitting uses fixed conformational assignments and averages exchange rates within groups. Existing ISO candidates are retained.</p>',
        '<p>The hybrid retains the union of each frame’s 20 nearest full-profile neighbours, then weights edges with the original scalar Gaussian kernels. Total off-diagonal coupling matches scalar OMC at every bandwidth; strength is 0.1. Disconnected components are retained, so their populations can change without direct coupling between components. State labels do not construct the graph.</p>',
        f'<p>{len(table)} configurations: {sum(table.family == h.FAMILY)} new hybrid fits and {sum(table.family != h.FAMILY)} reused controls. {int(table.converged.sum())} converged.</p>',
        '<p>Models are selected by all-residue MSE. Recovery = 100 × (1 − √base-2 JSD), including intermediate mass; it is not improvement from initial weights. Higher ESS alone does not establish recovery. Nearest-ESS comparisons require a gap no larger than two percentage points and use no interpolation. Unresolved pairs are excluded.</p>',
        '<h2>MSE-selected converged models</h2>',html_table(selected.reindex(columns=columns)),
        '<h2>Graph components and state composition</h2>',html_table(pd.read_csv(output/'components.csv')),
        '<h2>Coupling and neighbourhood diagnostics</h2>',html_table(pd.read_csv(output/'graphs.csv')),
        '<h2>Same-bandwidth differences: hybrid minus control</h2>',html_table(pairs),
        '<h2>Nearest-ESS comparisons</h2>',html_table(matched),
        '<h2>Forward approximation and starting populations</h2>',html_table(pd.read_csv(output/'source/preflight.csv')),
        '<h2>Downloads</h2><p>']
    body += [f'<a href="{p.name}">{p.name}</a> · ' for p in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">manifest</a></p>']
    for name in names:
        body.append(f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">')
    body += ['<h2>All configurations</h2>',html_table(table),'</body></html>']
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html',flush=True)
