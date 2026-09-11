"""Grouped-uptake fitting to a mean-rate reference target."""
from __future__ import annotations

import importlib
from pathlib import Path

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_reverse')
r=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')
e=s.e
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(output, experiment=s, mode='uptake'):
    if mode not in ('uptake','rate'):
        raise ValueError('Unsupported prediction mode')
    s=experiment
    prediction_name='grouped uptake' if mode=='uptake' else 'global mean rate'
    output=Path(output)
    s.validate(output)
    rows=s.t.s.collect(output)
    data=dict(np.load(output/'input.npz'))
    reference=np.load(output/'target.npz')
    expected=-np.expm1(-e.TIMES[:,None]*reference['reference_mean_rates'][None,:])
    np.testing.assert_allclose(data['target'],expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(data['scale'],np.var(expected)+1e-8)
    initial=[]
    for row in rows:
        if 'error' in row:
            continue
        result=np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')
        weights=result['weights']
        if not np.isfinite(result['initial_weights']).all() or (result['initial_weights']<0).any():
            raise ValueError('Invalid saved weights')
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1),1.,atol=1e-12)
        if row['stage']!=mode:
            raise ValueError('Archived forward-model mode mismatch')
        prediction=e.predict(data['rates'],weights,data['groups'],mode)
        np.testing.assert_allclose(prediction,result['prediction'],rtol=1e-10,atol=1e-12)
        mse=float(np.mean((prediction-data['target'])**2))
        objective=mse/float(data['scale'])
        if row['family']=='maxent':
            objective+=row['strength']*float(np.mean(-np.log(len(weights)*weights)))
        elif row['family']!='unregularised':
            objective+=row['strength']*r.pairwise_penalty(weights,np.load(s.kernel_path(output,row),mmap_mode='r'))
        np.testing.assert_allclose(objective,row['objective'],rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(mse,row['mse'],rtol=1e-10,atol=1e-14)
        for key,value in e.metrics(weights,data['groups']).items():
            np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-12)
        for start,w in enumerate(result['initial_weights']):
            initial.append(dict(arm=row['arm'],family=row['family'],start=start,**e.metrics(w,data['groups'])))
    table=pd.DataFrame(rows)
    table['distance']=table.get('distance',pd.Series(index=table.index,dtype=object))
    graphs=pd.read_csv(output/'graphs.csv')
    table=table.merge(graphs[['distance','quantile','sigma']],on=['distance','quantile'],how='left',validate='many_to_one')
    selected=s.t.select(table.to_dict('records'))
    table.to_csv(output/'fits.csv',index=False)
    selected.to_csv(output/'selected.csv',index=False)
    pd.DataFrame(initial).to_csv(output/'initialisations.csv',index=False)
    status=pd.DataFrame([dict(family=name,selected=bool((selected.family==name).any())) for name in table.family.unique()])
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
        for ax,metric,label in zip(axes.flat,('mse','recovery','intermediate','ess_fraction'),('MSE','Recovery score (%)','Intermediate (%)','ESS (%)')):
            factor=100 if metric in ('intermediate','ess_fraction') else 1
            ax.plot(curve['quantile'],factor*curve[metric],'.-',label=family)
            bad=curve.loc[~curve.converged.astype(bool)]
            ax.scatter(bad['quantile'],factor*bad[metric],marker='x',color='red',s=65)
            ax.set(xscale='log',xlabel='Scalar bandwidth quantile',ylabel=label)
            ax.set_xticks(e.QUANTILES,[f'{q:g}' for q in e.QUANTILES])
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Mean-rate target → {prediction_name} fit; OMC strength 0.1; crosses unresolved')
    save(fig,'bandwidth_sweep')
    fig,ax=plt.subplots(figsize=(12,5))
    masses=np.vstack([e.TRUTH,selected[['open','closed','intermediate']].to_numpy()])
    bottom=np.zeros(len(masses))
    for i,label in enumerate(('Open','Closed','Intermediate')):
        ax.bar(np.arange(len(masses)),masses[:,i],bottom=bottom,label=label)
        bottom+=masses[:,i]
    ax.set_xticks(np.arange(len(masses)),['Reference masses']+selected.family.tolist(),rotation=25,ha='right')
    ax.set(ylabel='Population',ylim=(0,1),title='Lowest-MSE converged fit per family')
    ax.legend()
    save(fig,'selected_populations')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for family,group in table.groupby('family'):
        valid=group.loc[group.converged.astype(bool)]
        for ax,key,label in zip(axes,('mse','ess_fraction'),('MSE','ESS fraction')):
            ax.scatter(valid[key],valid.recovery,label=family)
            ax.set(xlabel=label,ylabel='Recovery score (%)')
    axes[0].legend(fontsize=8)
    save(fig,'recovery_tradeoffs')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    columns=['family','quantile','sigma','strength','open','closed','intermediate','recovery','ess_fraction','mse','converged','objective_gap','steps']
    hierarchy=table.loc[table.arm>=25]
    body=[f'<!doctype html><html><head><meta charset="utf-8"><title>Mean-rate target, {prediction_name} fit</title>',
          '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}</style></head><body>',
          f'<h1>ISO_TRI: mean-rate target → {prediction_name} prediction</h1>',
          '<p>Target: 1 − exp(−t × Σᵢ wᵢ kᵣᵢ), using the full reference ensemble at 40% open and 60% closed. '+
          ('Prediction: average rates within each candidate state group, compute each group’s uptake, then mix by fitted group mass. ' if mode=='uptake' else
           'Prediction: the same global mean-rate equation over candidate frames; state groups do not enter the prediction. ')+
          'All 2,225 candidate frames, hard-contact rates, hierarchy kernels and bandwidths are unchanged. Data loss uses the mean-rate target variance + 1e−8.</p>',
          '<p>Eighteen hierarchy configurations at strength 0.1 and six bandwidths, plus 19 newly fitted baseline configurations: unregularised, six MaxEnt strengths, six scalar OMC and six profile OMC settings. Earlier fits to a different target are not reused as controls.</p>',
          f'<p>{len(table)}/37 outcomes; {int(table.converged.sum())} converged. Hierarchies: {int(hierarchy.converged.sum())}/{len(hierarchy)} converged. Unresolved results are shown but excluded from all-residue-MSE selection.</p>',
          '<p>Population and ESS table columns are fractions. Recovery is the existing JSD score against the reference’s 40:60:0 masses. Raw MSE across experiments with different targets is not a paired performance comparison.</p>',
          ('<p>For fixed rates and weights, uptake is concave in rate: global mean-rate uptake is at least as large as a mixture of group-wise uptakes. Thus swapping the two approximations does not undo their mismatch. A fit can compensate by changing selected structures or state populations. The preflight table quantifies the discrepancy before fitting; it does not by itself identify which candidate frames provide the compensation.</p>' if mode=='uptake' else
           '<p>The target and prediction now use the same approximation, but reference and candidate ensembles still differ. Consistency removes the forward-model mismatch without guaranteeing population identifiability or complete candidate coverage. The preflight table shows the candidate mean-rate error at true state masses with uniform weights within each state.</p>'),
          '<h2>Target and forward-model checks</h2>',html_table(pd.read_csv(output/'preflight.csv')),
          '<h2>All hierarchical fits</h2>',html_table(hierarchy.reindex(columns=columns)),
          '<h2>Selected per family</h2>',html_table(selected.reindex(columns=columns)),html_table(status),
          '<h2>Downloads</h2><p>']
    body += [f'<a href="{path.name}">{path.name}</a> · ' for path in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">Manifest</a></p>']
    for name in names:
        body += [f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">']
    body += ['<h2>All newly fitted configurations</h2>',html_table(table.reindex(columns=columns)),'</body></html>']
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html',flush=True)
