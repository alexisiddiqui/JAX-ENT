"""Tables and figures for the matched-frame ATLAS representation audit."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import graph_representation_audit as audit

LABELS = {'rmsd': 'Cα RMSD', 'w1': 'W1 (256 quantiles)',
          'logpf_profile': 'Full log-PF profile', 'logpf_mean': 'Mean log-PF'}


def build(output):
    output = Path(output)
    catalog = pd.DataFrame(audit.load_systems())
    folders = sorted((output / 'systems').glob('*/complete.json'))
    if not folders:
        raise ValueError('No completed systems')
    records = [audit.validate(path.parent) for path in folders]
    summary = pd.DataFrame([record['summary'] for record in records])
    expected = set(catalog.system_id)
    missing = sorted(expected - set(summary.system_id))
    tables = {}
    for name in ('pairs', 'graphs', 'edge_geometry', 'clusters', 'cluster_agreement'):
        tables[name] = pd.concat([pd.read_csv(path.parent / f'{name}.csv') for path in folders], ignore_index=True)
        tables[name].to_csv(output / f'{name}.csv', index=False)
    summary.to_csv(output / 'systems.csv', index=False)
    pairs = tables['pairs']
    values = ['spearman', 'neighbour_overlap', 'cross_replica_overlap', 'edge_jaccard']
    medians = pairs.groupby(['first', 'second'])[values].median().reset_index()
    medians.to_csv(output / 'pair_medians.csv', index=False)
    quartiles = pairs.groupby(['first', 'second'])[values].quantile([.25, .5, .75])
    quartiles.to_csv(output / 'pair_quartiles.csv')
    cluster = tables['cluster_agreement']
    cluster.groupby(['first', 'second', 'k'])[['ari', 'nmi']].median().to_csv(output / 'cluster_medians.csv')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, field, title in zip(axes, values[:3], ['Pair-distance Spearman', '20-neighbour overlap', 'Cross-replica overlap']):
        matrix = np.eye(4)
        for row in medians.itertuples():
            i, j = audit.METRICS.index(row.first), audit.METRICS.index(row.second)
            matrix[i, j] = matrix[j, i] = getattr(row, field)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(4), [LABELS[x] for x in audit.METRICS], rotation=45, ha='right')
        ax.set_yticks(range(4), [LABELS[x] for x in audit.METRICS])
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='white' if matrix[i,j] < .65 else 'black')
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.savefig(output / 'agreement.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, structural in zip(axes, ('rmsd', 'w1')):
        for representation in ('logpf_profile', 'logpf_mean'):
            data = cluster[(cluster['first'] == structural) & (cluster.second == representation)]
            grouped = data.groupby('k').ari
            x = np.array(audit.COUNTS)
            ax.plot(x, grouped.median().reindex(x), marker='o', label=LABELS[representation])
            ax.fill_between(x, grouped.quantile(.25).reindex(x), grouped.quantile(.75).reindex(x), alpha=.15)
        ax.set(title=LABELS[structural], xlabel='Number of clusters', ylabel='Adjusted Rand index (median, IQR)')
        ax.set_xticks(audit.COUNTS)
        ax.legend()
    fig.savefig(output / 'clusters.png')
    plt.close(fig)
    per_system = catalog[catalog.system_id.isin(summary.system_id)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, structural in zip(axes, ('rmsd', 'w1')):
        data = pairs[pairs['first'] == structural].pivot(index='system_id', columns='second', values='neighbour_overlap')
        per_system = per_system.merge(data[['logpf_profile', 'logpf_mean']].add_prefix(structural + '_'), on='system_id')
        ax.scatter(data.logpf_mean, data.logpf_profile, alpha=.7, s=22)
        ax.plot([0, 1], [0, 1], '--', color='grey')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean log-PF neighbour overlap', ylabel='Full-profile neighbour overlap', title=LABELS[structural])
    fig.savefig(output / 'profile_vs_mean.png')
    plt.close(fig)
    per_system.to_csv(output / 'system_comparison.csv', index=False)
    metadata = per_system.groupby('rmsf_tercile')[[c for c in per_system if c.endswith(('logpf_profile', 'logpf_mean'))]].agg(['count', 'median'])
    metadata.to_csv(output / 'rmsf_groups.csv')
    edges = tables['edge_geometry'].groupby(['metric', 'structural'])[['ratio_to_structural_neighbours', 'distant_edge_fraction']].median()
    edges.to_csv(output / 'edge_medians.csv')
    coverage = dict(completed=len(records), expected=len(expected), missing=missing,
                    complete_hashes={p.parent.name: audit.digest(p) for p in folders})
    audit.write_json(output / 'coverage.json', coverage)
    def table(frame):
        return frame.to_html(index=False, float_format=lambda x: f'{x:.3f}', escape=True)
    display = medians.replace(LABELS)
    worst = per_system.sort_values('rmsd_logpf_profile')[['system_id', 'length', 'rmsf_tercile', 'rmsd_logpf_profile', 'rmsd_logpf_mean', 'w1_logpf_profile']]
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>ATLAS graph representation audit</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;padding:20px;color:#172638}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd}}img{{width:100%}}a{{color:#155da0}}</style></head><body>
<h1>ATLAS: structural geometry versus log-PF</h1>
<p>{len(records)} / {len(expected)} systems complete. Missing: {html.escape(', '.join(missing) or 'none')}.</p>
<p>Fresh Bradshaw switched contacts: heavy-contact centre 6.5 Å, acceptor centre 2.4 Å, both switch scales 0.1 Å in the MDAnalysis coordinate convention, sequence exclusion ±2, BV coefficients 0.35 and 2. The implemented switch is 1/(1+((distance-centre)/scale)^6). Each system uses the same 768 frames (256 per replica, after 10 ns) for all four representations. Historical feature banks are not used.</p>
<p>Graphs use 20 nearest neighbours, with a symmetric union for edge comparisons. Cross-replica overlap excludes neighbours from the same replica. Random directed overlap is {pairs.random_neighbour_overlap.median():.2%}, or {pairs.random_cross_replica_overlap.median():.2%} across replicas. Clusters use average linkage with fixed counts 2, 3, 5, 10 and 20. Every system contributes equally to medians; shaded cluster ranges show the interquartile range across systems.</p>
<h2>Graph agreement</h2>{table(display)}<img src="agreement.png">
<p>Agreement measures whether these representations connect the same structures. It does not establish which graph gives the correct physical weights. RMSD preserves aligned atomic geometry; W1 compares internal-distance distributions and discards residue-pair identity; full log-PF preserves residue-specific contact protection, while its mean discards that identity and can cancel opposing changes.</p>
<h2>Cluster agreement</h2><img src="clusters.png"><p>ARI compares assignments at the same cluster count and adjusts for chance. No cluster count was selected to maximise agreement. Unequal or replica-specific clusters can affect these scores; all sizes, silhouettes and replica NMI are in <a href="clusters.csv">clusters.csv</a>.</p>
<h2>Does retaining residue identity help?</h2><img src="profile_vs_mean.png"><p>Each point is one system; points above the diagonal favour full profiles.</p>
<h2>Structural length of graph edges</h2>{table(edges.reset_index().replace(LABELS))}
<p>The distance ratio compares a graph's mean structural neighbour distance to the structural graph's own mean distance. Distant edges lie beyond the 95th percentile of structural distance from their source frame.</p>
<h2>W1 approximation audit</h2><p>W1 uses 256 quantiles, matching the existing ATLAS graph convention. Exact empirical W1 is checked on 1,024 pairs and on all neighbours of 16 sampled anchors per system. Median exact/approximate pair Spearman: {summary.w1_audit_spearman.median():.3f}. Median exact 20-neighbour overlap: {summary.w1_exact_neighbour_overlap.median():.1%}; range {summary.w1_exact_neighbour_overlap.min():.1%}–{summary.w1_exact_neighbour_overlap.max():.1%}. This bounds interpretation of W1 graph agreement; RMSD is computed with exact pairwise optimal rotation.</p>
<h2>Systems ordered by full-profile/RMSD neighbour agreement</h2>{table(worst)}
<h2>Mobility groups</h2>{metadata.to_html()}<p>These are descriptive associations with catalogued RMSF, not evidence of a causal mechanism.</p>
<h2>Download and provenance</h2><p>Aggregated tables: '''
    for name in ('pairs', 'pair_medians', 'pair_quartiles', 'graphs', 'edge_geometry', 'edge_medians', 'clusters', 'cluster_agreement', 'cluster_medians', 'systems', 'system_comparison', 'rmsf_groups'):
        body += f'<a href="{name}.csv">{name}</a> · '
    body += '<a href="coverage.json">coverage and hashes</a>.</p><p>Per-system folders contain regenerated features, frame indices, distance matrices, graph masks, cluster labels, exact-W1 audits and hashed source/artifact manifests. This diagnostic does not test fitted population recovery or establish a preferred regularisation strength.</p></body></html>'
    (output / 'index.html').write_text(body)
    print(json.dumps(coverage, indent=2))
