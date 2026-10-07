"""Mean-rate hierarchy results and paired grouped-uptake comparisons."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_rate')
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
    rows=s.t.s.collect(output)
    grouped=[]
    for marker in sorted((output/'grouped').glob('arm_*/complete.json')):
        record=json.loads(marker.read_text())
        if e.digest(marker.with_name('fit.npz'))!=record['sha256']:
            raise ValueError('Corrupt grouped archive')
        grouped.append(record['row'])
    data=dict(np.load(output/'input.npz'))
    initial=[]
    for row in rows+grouped:
        if 'error' in row:
            continue
        folder=output/('fits' if row['stage']=='rate' else 'grouped')/f"arm_{row['arm']:02d}"
        result=np.load(folder/'fit.npz')
        w=result['weights']
        if not np.isfinite(result['initial_weights']).all() or (result['initial_weights']<0).any():
            raise ValueError('Invalid weights')
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1),1.,atol=1e-12)
        prediction=e.predict(data['rates'],w,data['groups'],row['stage'])
        np.testing.assert_allclose(prediction,result['prediction'],rtol=1e-10,atol=1e-12)
        mse=float(np.mean((prediction-data['target'])**2))
        objective=mse/float(data['scale'])
        if row['family']=='maxent':
            objective+=row['strength']*float(np.mean(-np.log(len(w)*w)))
        elif row['family']!='unregularised':
            objective+=row['strength']*r.pairwise_penalty(w,np.load(s.kernel_path(output,row),mmap_mode='r'))
        np.testing.assert_allclose(objective,row['objective'],rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(mse,row['mse'],rtol=1e-10,atol=1e-14)
        for key,value in e.metrics(w,data['groups']).items():
            np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-12)
        for start,weights in enumerate(result['initial_weights']):
            initial.append(dict(arm=row['arm'],stage=row['stage'],family=row['family'],start=start,**e.metrics(weights,data['groups'])))
    table=pd.DataFrame(rows)
    table['distance']=table.get('distance',pd.Series(index=table.index,dtype=object))
    graphs=pd.read_csv(output/'graphs.csv')
    table=table.merge(graphs[['distance','quantile','sigma']],on=['distance','quantile'],how='left',validate='many_to_one')
    selected=s.t.select(table.to_dict('records'))
    pairs=[]
    for row in rows:
        if row['arm']<25:
            continue
        other=next(r for r in grouped if r['arm']==row['arm'])
        valid=bool(row['converged'] and other['converged'])
        pairs.append(dict(family=row['family'],quantile=row['quantile'],valid_pair=valid,
                          **{'delta_'+key:row[key]-other[key] if valid else np.nan
                             for key in ('mse','recovery','intermediate','ess_fraction')}))
    table.to_csv(output/'fits.csv',index=False)
    selected.to_csv(output/'selected.csv',index=False)
    pd.DataFrame(grouped).to_csv(output/'grouped_controls.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'paired_forward_models.csv',index=False)
    pd.DataFrame(initial).to_csv(output/'initialisations.csv',index=False)
    status=pd.DataFrame([dict(family=family,selected=bool((selected.family==family).any())) for family in table.family.unique()])
    status.to_csv(output/'selection_status.csv',index=False)
    figures=output/'figures'
    figures.mkdir(exist_ok=True)
    names=[]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','svg'):
            fig.savefig(figures/f'{name}.{suffix}',dpi=150)
        plt.close(fig)
        names.append(name)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for family,group in table.groupby('family'):
        if family in ('maxent','unregularised'):
            continue
        curve=group.sort_values('quantile')
        for ax,metric,label in zip(axes.flat,('mse','recovery','intermediate','ess_fraction'),('All-residue MSE','Recovery score (%)','Intermediate (%)','ESS (%)')):
            factor=100 if metric in ('intermediate','ess_fraction') else 1
            ax.plot(curve['quantile'],curve[metric]*factor,'.-',label=family)
            bad=curve.loc[~curve.converged.astype(bool)]
            ax.scatter(bad['quantile'],bad[metric]*factor,marker='x',color='red',s=65)
            ax.set(xscale='log',xlabel='Scalar bandwidth quantile',ylabel=label)
            ax.set_xticks(e.QUANTILES,[f'{q:g}' for q in e.QUANTILES])
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Mean-rate forward model, strength 0.1; crosses mark unresolved fits')
    save(fig,'mean_rate_sweep')
    fig,axes=plt.subplots(3,3,figsize=(14,11))
    both=pd.DataFrame(rows+grouped)
    for i,name in enumerate(s.t.DISTANCES):
        for mode,group in both.loc[both.family=='hierarchical_'+name].groupby('stage'):
            curve=group.sort_values('quantile')
            for ax,key,label in zip(axes[i],('mse','recovery','ess_fraction'),('MSE','Recovery score (%)','ESS (%)')):
                factor=100 if key=='ess_fraction' else 1
                ax.plot(curve['quantile'],curve[key]*factor,'.-',label='Mean rate' if mode=='rate' else 'Grouped uptake')
                bad=curve.loc[~curve.converged.astype(bool)]
                ax.scatter(bad['quantile'],bad[key]*factor,marker='x',color='red')
                ax.set(xscale='log',xlabel='Scalar bandwidth quantile',ylabel=label,title=name)
        axes[i,0].legend(fontsize=8)
    save(fig,'forward_model_comparison')
    fig,ax=plt.subplots(figsize=(12,5))
    masses=np.vstack([e.TRUTH,selected[['open','closed','intermediate']].to_numpy()])
    bottom=np.zeros(len(masses))
    for i,label in enumerate(('Open','Closed','Intermediate')):
        ax.bar(np.arange(len(masses)),masses[:,i],bottom=bottom,label=label)
        bottom+=masses[:,i]
    ax.set_xticks(np.arange(len(masses)),['Target']+selected.family.tolist(),rotation=25,ha='right')
    ax.set(ylabel='Population',ylim=(0,1),title='Lowest-MSE converged mean-rate fit per family')
    ax.legend()
    save(fig,'selected_populations')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    columns=['family','quantile','sigma','strength','open','closed','intermediate','recovery','ess_fraction','mse','converged','objective_gap','steps']
    new=table.loc[table.arm>=25]
    body=['<!doctype html><html><head><meta charset="utf-8"><title>ISO_TRI hierarchical mean rate</title>',
          '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}</style></head><body>',
          '<h1>ISO_TRI hierarchical graphs: mean-rate forward model</h1>',
          '<p>The prediction is 1 − exp(−t × Σᵢ wᵢ kᵣᵢ): average exchange rates across all candidate frames before computing uptake. This is neither average log-PF nor grouped uptake. The frame-wise uptake target is unchanged, as are all 2,225 candidate frames, hard-contact features, intrinsic rates, kernels, total coupling, initialisations and convergence rules.</p>',
          '<p>Three existing hierarchical graphs, six original scalar bandwidths, strength 0.1. No lower-strength fits are included. Target populations: 40% open, 60% closed, 0% intermediate. Table populations and ESS are fractions; recovery is the existing JSD score.</p>',
          f'<p>{len(new)}/18 new outcomes; {int(new.converged.sum())} converged. Nineteen original mean-rate controls are included, with 18 grouped hierarchy fits as paired forward-model controls.</p>',
          '<h2>All new mean-rate fits</h2>',html_table(new.reindex(columns=columns)),
          '<h2>Selected by all-residue MSE among converged fits</h2>',html_table(selected.reindex(columns=columns)),html_table(status),
          '<h2>Mean-rate minus grouped-uptake results at the same graph and bandwidth</h2>',html_table(pd.DataFrame(pairs)),
          '<p>Paired differences require both fits to converge; unresolved configurations remain visible but are excluded from selection and valid comparisons. Population labels are used for evaluation, not by the global mean-rate prediction.</p>',
          '<h2>Downloads</h2><p>']
    body += [f'<a href="{path.name}">{path.name}</a> · ' for path in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">Manifest</a></p>']
    for name in names:
        body += [f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">']
    body += ['<h2>All mean-rate controls and fits</h2>',html_table(table.reindex(columns=columns)),'</body></html>']
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html',flush=True)
