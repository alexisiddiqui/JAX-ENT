"""Validated fit and graph diagnostics for the ISO_TRI hierarchy experiment."""
from __future__ import annotations

import importlib
from pathlib import Path

t = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_control')
r = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')
e = t.e
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(output):
    output = Path(output)
    manifest = t.validate(output)
    rows = t.s.collect(output)
    table = pd.DataFrame(rows)
    graphs = pd.read_csv(output/'graphs.csv')
    table['distance'] = table.get('distance', pd.Series(index=table.index, dtype=object))
    table = table.merge(graphs[['distance','quantile','sigma']], on=['distance','quantile'], how='left', validate='many_to_one')
    selected = t.select(table.to_dict('records'))
    data = dict(np.load(output/'input.npz'))
    initial = []
    for row in rows:
        if 'error' in row:
            continue
        result = np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')
        w = result['weights']
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1), 1., atol=1e-12)
        if not np.isfinite(result['initial_weights']).all() or (result['initial_weights'] < 0).any():
            raise ValueError('Invalid archived weights')
        prediction = e.predict(data['rates'], w, data['groups'], 'uptake')
        np.testing.assert_allclose(prediction, result['prediction'], rtol=1e-10, atol=1e-12)
        mse = float(np.mean((prediction-data['target'])**2))
        objective = mse/float(data['scale'])
        if row['family'] == 'maxent':
            objective += row['strength']*float(np.mean(-np.log(len(w)*w)))
        elif row['family'] != 'unregularised':
            kernel = np.load(t.kernel_path(output, row), mmap_mode='r')
            objective += row['strength']*r.pairwise_penalty(w, kernel)
        np.testing.assert_allclose(objective, row['objective'], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(mse, row['mse'], rtol=1e-10, atol=1e-14)
        for metric, value in e.metrics(w, data['groups']).items():
            np.testing.assert_allclose(value, row[metric], rtol=1e-10, atol=1e-12)
        for start, weights in enumerate(result['initial_weights']):
            initial.append(dict(arm=row['arm'], family=row['family'], start=start, **e.metrics(weights, data['groups'])))
    table.to_csv(output/'fits.csv', index=False)
    selected.to_csv(output/'selected.csv', index=False)
    pd.DataFrame(initial).to_csv(output/'initialisations.csv', index=False)
    selection_status = [dict(family=family, selected=bool((selected.family==family).any()),
                             reason='lowest converged all-residue MSE' if (selected.family==family).any() else 'no converged fit')
                        for family in table.family.unique()]
    pd.DataFrame(selection_status).to_csv(output/'selection_status.csv', index=False)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    names = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png','svg'):
            fig.savefig(figures/f'{name}.{suffix}', dpi=150)
        plt.close(fig)
        names.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for family, group in table.groupby('family'):
        if family in ('maxent','unregularised'):
            continue
        curve = group.sort_values('quantile')
        for ax, metric, label in zip(axes.flat, ('mse','recovery','intermediate','ess_fraction'),
                                   ('All-residue MSE','Recovery score (%)','Intermediate population','ESS fraction')):
            ax.plot(curve['quantile'], curve[metric], '.-', label=family, alpha=.85)
            bad = curve.loc[~curve.converged.astype(bool)]
            ax.scatter(bad['quantile'], bad[metric], marker='x', color='red', s=60)
            ax.set(xscale='log', xlabel='Existing scalar bandwidth quantile', ylabel=label)
            ax.set_xticks(e.QUANTILES, [f'{q:g}' for q in e.QUANTILES])
    axes[0,0].legend(fontsize=7)
    fig.suptitle('ISO_TRI at strength 0.1; red crosses mark unresolved configurations')
    save(fig, 'bandwidth_curves')
    fig, ax = plt.subplots(figsize=(8, 4))
    for family, group in table.loc[table.arm>=25].groupby('family'):
        curve = group.sort_values('quantile')
        ax.plot(curve['quantile'], 100*curve.ess_fraction, '.-', label=family)
        bad = curve.loc[~curve.converged.astype(bool)]
        ax.scatter(bad['quantile'], 100*bad.ess_fraction, marker='x', color='red', s=60)
    ax.set(xscale='log', xlabel='Scalar bandwidth quantile', ylabel='ESS fraction (%)',
           title='Hierarchical graphs: ESS at an expanded scale')
    ax.set_xticks(e.QUANTILES, [f'{q:g}' for q in e.QUANTILES])
    ax.legend(fontsize=8)
    save(fig, 'hierarchical_ess')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for family, group in table.groupby('family'):
        valid = group.loc[group.converged.astype(bool)]
        for ax, x, label in zip(axes, ('mse','ess_fraction'), ('All-residue MSE','ESS fraction')):
            ax.scatter(valid[x], valid.recovery, label=family, s=30)
            bad = group.loc[~group.converged.astype(bool)]
            ax.scatter(bad[x], bad.recovery, marker='x', alpha=.4)
            ax.set(xlabel=label, ylabel='Recovery score (%)')
    axes[0].legend(fontsize=7)
    save(fig, 'recovery_tradeoffs')
    fig, ax = plt.subplots(figsize=(12, 5))
    populations = np.vstack([e.TRUTH, selected[['open','closed','intermediate']].to_numpy()])
    bottom = np.zeros(len(populations))
    for i, name in enumerate(('Open','Closed','Intermediate')):
        ax.bar(np.arange(len(populations)), populations[:,i], bottom=bottom, label=name)
        bottom += populations[:,i]
    ax.set_xticks(np.arange(len(populations)), ['Target']+selected.family.tolist(), rotation=30, ha='right')
    ax.set(ylabel='Population', ylim=(0,1), title='Lowest-MSE converged fit per family')
    ax.legend()
    save(fig, 'selected_populations')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in t.DISTANCES:
        graph = np.load(output/f'graph_{name}.npz')
        degrees, counts = np.unique(graph['degree'], return_counts=True)
        axes[0].plot(degrees, counts, '.-', label=name)
        heights = np.sort(graph['linkage'][:,2])
        axes[1].plot(np.arange(1,len(heights)+1)/len(heights), heights, label=name)
        curve = graphs.loc[graphs.distance==name]
        axes[2].plot(curve['quantile'], curve.effective_edges, '.-', label=name)
    axes[0].set(xlabel='Frame degree', ylabel='Number of frames', yscale='log')
    axes[1].set(xlabel='Fraction of merges', ylabel='Merge height (metric-specific units)')
    axes[2].set(xlabel='Scalar bandwidth quantile', ylabel='Effective number of weighted edges', xscale='log')
    axes[0].legend()
    save(fig, 'graph_diagnostics')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False, float_format=lambda v:f'{v:.6g}')+'</div>'
    columns = ['family','quantile','sigma','strength','open','closed','intermediate','recovery','ess_fraction','mse','converged','objective_gap','steps']
    new = table.loc[table.arm>=25]
    body = ['<!doctype html><html><head><meta charset="utf-8"><title>ISO_TRI hierarchical graphs</title>',
            '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}</style></head><body>',
            '<h1>ISO_TRI: hierarchical neighbourhoods with mean-log-PF weights</h1>',
            '<p>Three average-linkage hierarchies: exact pairwise Cα RMSD, 256-quantile W1, and full-residue log-PF RMS distance. Each retains the closest cross-child frame pair at every merge, yielding 2,224 undirected edges on 2,225 frames. All edges use the existing mean-log-PF Gaussian weights and six original scalar bandwidths. Strength is 0.1 and total coupling matches the original scalar graph separately at each bandwidth.</p>',
            '<p>The frozen hard-contact candidate inputs, frame-wise uptake target and grouped-uptake fitter are unchanged. Target populations are 40% open, 60% closed and 0% intermediate. Population labels are used only after graph construction for diagnostics. All-residue MSE selects among converged fits; recovery is the existing base-2 JSD score, not percentage improvement.</p>',
            f'<p>{len(new)}/18 new configurations available; {int(new.converged.sum())} converged. {len(table)-len(new)} existing controls included. Unresolved configurations remain visible and are excluded from selection.</p>',
            '<h2>All new configurations</h2>', html_table(new.reindex(columns=columns)),
            '<h2>Selected per family</h2>', html_table(selected.reindex(columns=columns)), html_table(pd.DataFrame(selection_status)),
            '<p>Comparisons between the three new graphs isolate distance choice under this construction. Comparisons with old graphs also change sparsity/topology. Total coupling varies across the original bandwidth sweep; matching it controls comparisons between graphs at each bandwidth. A connected tree need not have strong effective coupling across every state boundary.</p>',
            '<h2>Graph diagnostics</h2>', html_table(graphs), html_table(pd.read_csv(output/'graph_overlap.csv')),
            f'<h2>W1 approximation audit</h2><p>Exact empirical W1 checks: 1,024 pairs (half random, half neighbours), pair Spearman {manifest["w1_pair_spearman"]:.4f}; exact 20-neighbour overlap averaged over 16 anchors {manifest["w1_neighbour_overlap"]:.1%}. These checks measure approximation error, not physical validity.</p>',
            '<h2>Downloads</h2><p>']
    body += [f'<a href="{path.name}">{path.name}</a> · ' for path in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">Source and artifact manifest</a></p>']
    for name in names:
        body += [f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">']
    body += ['<h2>All controls and new fits</h2>', html_table(table.reindex(columns=columns)), '</body></html>']
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html', flush=True)
