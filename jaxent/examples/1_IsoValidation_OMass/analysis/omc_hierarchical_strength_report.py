"""Matched-kernel strength comparisons for ISO_TRI hierarchical OMC."""
from __future__ import annotations

import importlib
from pathlib import Path

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_strength')
r=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')
e=s.e
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(output):
    output=Path(output)
    s.validate(output)
    rows=s.s.collect(output)
    table=pd.DataFrame(rows)
    data=dict(np.load(output/'input.npz'))
    starts=[]
    for row in rows:
        if 'error' in row:
            continue
        result=np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')
        weights=result['weights']
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1),1.,atol=1e-12)
        if not np.isfinite(result['initial_weights']).all() or (result['initial_weights']<0).any():
            raise ValueError('Invalid saved weights')
        prediction=e.predict(data['rates'],weights,data['groups'],'uptake')
        np.testing.assert_allclose(prediction,result['prediction'],rtol=1e-10,atol=1e-12)
        mse=float(np.mean((prediction-data['target'])**2))
        objective=mse/float(data['scale'])
        if row['family']=='maxent':
            objective+=row['strength']*float(np.mean(-np.log(len(weights)*weights)))
        elif row['family']!='unregularised':
            kernel=np.load(s.kernel_path(output,row),mmap_mode='r')
            objective+=row['strength']*r.pairwise_penalty(weights,kernel)
        np.testing.assert_allclose(objective,row['objective'],rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(mse,row['mse'],rtol=1e-10,atol=1e-14)
        for key,value in e.metrics(weights,data['groups']).items():
            np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-12)
        for start,w in enumerate(result['initial_weights']):
            starts.append(dict(arm=row['arm'],family=row['family'],strength=row['strength'],start=start,**e.metrics(w,data['groups'])))
    graphs=pd.read_csv(output/'graphs.csv')
    table=table.merge(graphs[['distance','quantile','sigma']],on=['distance','quantile'],how='left',validate='many_to_one')
    hierarchical=table.loc[table.family.str.startswith('hierarchical_')]
    wider=hierarchical.loc[hierarchical['quantile'].isin(s.WIDTHS)]
    selected=s.select(hierarchical.to_dict('records'))
    selected_wide=s.select(wider.to_dict('records'))
    pairs=s.comparisons(table)
    table.to_csv(output/'fits.csv',index=False)
    wider.to_csv(output/'wider_bandwidths.csv',index=False)
    selected.to_csv(output/'selected_by_strength.csv',index=False)
    selected_wide.to_csv(output/'selected_wider_only.csv',index=False)
    pairs.to_csv(output/'paired_strength.csv',index=False)
    pd.DataFrame(starts).to_csv(output/'initialisations.csv',index=False)
    status=[]
    for name in s.t.DISTANCES:
        for strength in (*s.STRENGTHS,.1):
            exists=((selected.family=='hierarchical_'+name)&(selected.strength==strength)).any()
            status.append(dict(distance=name,strength=strength,selected=bool(exists),reason='lowest converged MSE' if exists else 'no converged fit'))
    pd.DataFrame(status).to_csv(output/'selection_status.csv',index=False)
    figures=output/'figures'
    figures.mkdir(exist_ok=True)
    names=[]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','svg'):
            fig.savefig(figures/f'{name}.{suffix}',dpi=150)
        plt.close(fig)
        names.append(name)
    fig,axes=plt.subplots(3,4,figsize=(17,11))
    for row_index,name in enumerate(s.t.DISTANCES):
        data_rows=wider.loc[wider.distance==name]
        for q,group in data_rows.groupby('quantile'):
            curve=group.sort_values('strength')
            for ax,metric,label in zip(axes[row_index],('mse','recovery','intermediate','ess_fraction'),('MSE','Recovery score (%)','Intermediate (%)','ESS (%)')):
                factor=100 if metric in ('intermediate','ess_fraction') else 1
                ax.plot(curve.strength,curve[metric]*factor,'.-',label=f'q={q:g}')
                bad=curve.loc[~curve.converged.astype(bool)]
                ax.scatter(bad.strength,bad[metric]*factor,marker='x',color='red',s=65)
                ax.set(xscale='log',xlabel='Strength',ylabel=label,title=name)
                ax.set_xticks([.01,.03,.1],['0.01','0.03','0.1'])
        axes[row_index,0].legend(fontsize=8)
    fig.suptitle('Identical kernels at each bandwidth; red crosses mark unresolved fits')
    save(fig,'strength_curves')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,name in zip(axes,s.t.DISTANCES):
        for q,group in wider.loc[wider.distance==name].groupby('quantile'):
            curve=group.sort_values('strength')
            ax.plot(curve.ess_fraction*100,curve.recovery,'.-',label=f'q={q:g}')
            for row in curve.to_dict('records'):
                ax.annotate(f"{row['strength']:g}",(row['ess_fraction']*100,row['recovery']),fontsize=7,xytext=(3,3),textcoords='offset points')
            bad=curve.loc[~curve.converged.astype(bool)]
            ax.scatter(bad.ess_fraction*100,bad.recovery,marker='x',color='red')
        ax.set(title=name,xlabel='ESS fraction (%)',ylabel='Recovery score (%)')
        ax.legend(fontsize=8)
    save(fig,'recovery_ess')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    columns=['distance','quantile','sigma','strength','open','closed','intermediate','recovery','ess_fraction','mse','converged','objective_gap','steps']
    new=table.loc[table.arm>=43]
    body=['<!doctype html><html><head><meta charset="utf-8"><title>ISO_TRI lower hierarchical strengths</title>',
          '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}</style></head><body>',
          '<h1>ISO_TRI: higher bandwidth, lower hierarchical strength</h1>',
          '<p>RMSD, W1 and full-profile log-PF hierarchies; bandwidth quantiles 0.16, 0.32 and 0.64; new strengths 0.01 and 0.03. All kernels are byte-identical to the corresponding strength-0.1 graph. No coupling rescaling offsets the strength reduction. Frozen hard-contact inputs, frame-wise uptake target and grouped-uptake fitter are unchanged.</p>',
          f'<p>{len(new)}/18 new outcomes; {int(new.converged.sum())} converged. Red crosses show unresolved fits, which are excluded from selection. Existing controls include the full strength-0.1 bandwidth sweep.</p>',
          '<p>Target populations: 40% open, 60% closed, 0% intermediate. Population and ESS table columns are fractions. Recovery is the existing JSD-based score. All-residue MSE selects among converged configurations; higher recovery alone does not determine selection.</p>',
          '<h2>All wider-bandwidth results, including strength 0.1</h2>',html_table(wider.sort_values(['distance','quantile','strength']).reindex(columns=columns)),
          '<h2>Selected within each strength, using available bandwidths</h2>',html_table(selected.reindex(columns=columns)),
          '<p>Strength 0.1 has six available bandwidths, while lower strengths have only the three wider bandwidths. The following restricted selection uses the same three bandwidths for every strength.</p>',
          '<h2>Selected within the three wider bandwidths only</h2>',html_table(selected_wide.reindex(columns=columns)),
          html_table(pd.DataFrame(status)),
          '<h2>Differences from strength 0.1 at identical bandwidth</h2>',html_table(pairs),
          '<p>Paired deltas require convergence on both sides. Invalid pairs have blank deltas; no interpolation or substituted control is used.</p>',
          '<h2>Downloads</h2><p>']
    body += [f'<a href="{path.name}">{path.name}</a> · ' for path in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">Manifest</a></p>']
    for name in names:
        body += [f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">']
    body += ['<h2>All archived controls and new fits</h2>',html_table(table.reindex(columns=['family',*columns[1:]])),'</body></html>']
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html',flush=True)
