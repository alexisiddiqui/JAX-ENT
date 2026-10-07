"""Fixed-graph lower-strength ISO_TRI recovery and distribution diagnostics."""
from __future__ import annotations

import importlib
from pathlib import Path

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_strength')
r=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')
e=s.e
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def objective_at_weights(data,weights,kernel,strength):
    prediction=e.predict(data['rates'],weights,data['groups'],'uptake')
    mse=float(np.mean((prediction-data['target'])**2))
    return mse/float(data['scale'])+strength*r.pairwise_penalty(weights,kernel)


def comparisons(table):
    pairs,matched=[],[]
    for row in table.loc[table.arm>=25].to_dict('records'):
        baseline=table.loc[(table.family==s.h.FAMILY)&(table.strength==.1)&(table['quantile']==row['quantile'])].iloc[0]
        valid=bool(row['converged'] and baseline.converged)
        pairs.append(dict(arm=row['arm'],quantile=row['quantile'],strength=row['strength'],valid_pair=valid,
            **{'delta_'+metric:row[metric]-baseline[metric] if valid else np.nan
               for metric in ('recovery','tv','intermediate','ess_fraction','mse')}))
        for family in (s.h.FAMILY,'maxent'):
            controls=table.loc[(table.arm<25)&(table.family==family)&table.converged.astype(bool)]
            base=dict(arm=row['arm'],quantile=row['quantile'],strength=row['strength'],control=family)
            if not row['converged'] or controls.empty:
                matched.append(dict(**base,matched=False,reason='unconverged or no converged control'))
                continue
            other=controls.loc[(controls.ess_fraction-row['ess_fraction']).abs().idxmin()]
            gap=abs(other.ess_fraction-row['ess_fraction'])
            valid=bool(gap<=.02)
            matched.append(dict(**base,matched=valid,reason='within tolerance' if valid else 'outside ESS tolerance',
                control_arm=other.arm,ess_gap=gap,**{'delta_'+metric:row[metric]-other[metric] if valid else np.nan
                for metric in ('recovery','tv','intermediate','mse')}))
    return pd.DataFrame(pairs),pd.DataFrame(matched)


def build(output):
    output=Path(output)
    s.validate(output)
    rows=s.collect(output)
    table=pd.DataFrame(rows)
    selected=s.select(rows)
    per_strength=s.select(rows,per_strength=True)
    data=dict(np.load(output/'input.npz'))
    residue_keys=e.layout(output/'topology.json')
    initial,residuals=[],[]
    for row in rows:
        if 'error' in row:
            continue
        result=np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')
        weights=result['weights']
        if not np.isfinite(result['initial_weights']).all():
            raise ValueError('Nonfinite archived weights')
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1),1.,atol=1e-12)
        prediction=e.predict(data['rates'],weights,data['groups'],'uptake')
        np.testing.assert_allclose(prediction,result['prediction'],rtol=1e-10,atol=1e-12)
        mse=float(np.mean((prediction-data['target'])**2))
        objective=mse/float(data['scale'])
        if row['family']==s.h.FAMILY:
            kernel=np.load(s.kernel_path(output,row['quantile']),mmap_mode='r')
            objective=objective_at_weights(data,weights,kernel,row['strength'])
        elif row['family']=='maxent':
            objective+=row['strength']*float(np.mean(-np.log(len(weights)*weights)))
        np.testing.assert_allclose(objective,row['objective'],rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(mse,row['mse'],rtol=1e-10,atol=1e-14)
        for start,w in enumerate(result['initial_weights']):
            initial.append(dict(arm=row['arm'],family=row['family'],start=start,**e.metrics(w,data['groups'])))
        for t,time in enumerate(e.TIMES):
            for j,(chain,residue) in enumerate(residue_keys):
                residuals.append(dict(arm=row['arm'],family=row['family'],chain=chain,residue=residue[0],time_min=time,
                    target=data['target'][t,j],prediction=prediction[t,j],residual=prediction[t,j]-data['target'][t,j]))
    pairs,matched=comparisons(table)
    table.to_csv(output/'fits.csv',index=False)
    selected.to_csv(output/'selected.csv',index=False)
    per_strength.to_csv(output/'selected_by_strength.csv',index=False)
    pairs.to_csv(output/'paired_strength.csv',index=False)
    matched.to_csv(output/'nearest_ess.csv',index=False)
    pd.DataFrame(initial).to_csv(output/'initialisation_populations.csv',index=False)
    pd.DataFrame(residuals).to_csv(output/'residuals.csv',index=False)
    figures=output/'figures'
    figures.mkdir(exist_ok=True)
    names=[]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','svg'):
            fig.savefig(figures/f'{name}.{suffix}',dpi=160)
        plt.close(fig)
        names.append(name)
    hybrids=table.loc[table.family==s.h.FAMILY]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for q in s.WIDTHS:
        curve=hybrids.loc[hybrids['quantile']==q].sort_values('strength')
        finite=curve.loc[np.isfinite(curve.mse)]
        for ax,metric,label in zip(axes,('recovery','intermediate','mse'),('Recovery score (%)','Intermediate mass (%)','All-residue MSE')):
            factor=100 if metric=='intermediate' else 1
            ax.plot(finite.ess_fraction*100,finite[metric]*factor,'.-',label=f'q={q:g}')
            for row in finite.to_dict('records'):
                ax.annotate(f"{row['strength']:g}",(row['ess_fraction']*100,row[metric]*factor),fontsize=7,xytext=(3,4),textcoords='offset points')
            invalid=finite.loc[~finite.converged.astype(bool)]
            ax.scatter(invalid.ess_fraction*100,invalid[metric]*factor,marker='x',color='red',s=65)
            ax.set(xlabel='ESS fraction (%)',ylabel=label)
    axes[0].legend(fontsize=8)
    fig.suptitle('ISO_TRI: labels show strength; lines connect strengths at fixed bandwidth; crosses are unresolved')
    save(fig,'strength_curves')
    baseline=table.loc[(table.family=='unregularised')&table.converged.astype(bool)]
    maxent=table.loc[(table.family=='maxent')&table.converged.astype(bool)].sort_values(['mse','arm']).head(1)
    chosen=pd.concat([per_strength,maxent,baseline]).drop_duplicates('arm')
    fig,ax=plt.subplots(figsize=(10,4.5))
    bars=np.vstack([e.TRUTH,chosen[['open','closed','intermediate']].to_numpy()])
    bottom=np.zeros(len(bars))
    for i,name in enumerate(('Open','Closed','Intermediate')):
        ax.bar(np.arange(len(bars)),bars[:,i],bottom=bottom,label=name)
        bottom+=bars[:,i]
    labels=['Target']+[f"hybrid λ={row['strength']:g}, q={row['quantile']:g}" if row['family']==s.h.FAMILY else row['family'] for row in chosen.to_dict('records')]
    ax.set_xticks(np.arange(len(bars)),labels,rotation=25,ha='right')
    ax.set(ylabel='Population',ylim=(0,1),title='MSE-selected fits within each strength and comparator family')
    ax.legend(fontsize=8)
    save(fig,'selected_populations')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,(state,name) in enumerate(zip(e.STATES,('Open','Closed','Intermediate'))):
        keep=data['groups']==state
        for coordinate,ref in enumerate(('Open','Closed')):
            ax=axes[coordinate,col]
            x=data['rmsd'][keep,coordinate]
            order=np.argsort(x)
            ax.plot(x[order],np.arange(1,len(x)+1)/len(x),'--',color='grey',label='Uniform within state')
            for row in chosen.to_dict('records'):
                w=np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')['weights'][keep]
                label=f"hybrid λ={row['strength']:g}" if row['family']==s.h.FAMILY else row['family']
                ax.plot(x[order],np.cumsum(w[order])/w.sum(),label=label)
            ax.set(title=name,xlabel=f'{ref}-reference RMSD (Å)',ylabel='Conditional cumulative weight',ylim=(0,1))
    axes[0,0].legend(fontsize=7)
    save(fig,'coverage')
    fig,axes=plt.subplots(len(chosen),1,figsize=(13,2.3*len(chosen)),squeeze=False)
    arrays=[np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')['prediction']-data['target'] for row in chosen.to_dict('records')]
    limit=max(float(abs(a).max()) for a in arrays)
    for ax,row,residual in zip(axes[:,0],chosen.to_dict('records'),arrays):
        im=ax.imshow(residual,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
        ticks=np.arange(0,len(residue_keys),40)
        ax.set_xticks(ticks,[residue_keys[i][1][0] for i in ticks])
        ax.set_yticks(np.arange(len(e.TIMES)),[f'{t:g}' for t in e.TIMES])
        ax.set(title=f"{row['family']}, strength={row['strength']:g}",xlabel='Residue number',ylabel='Time (min)')
        fig.colorbar(im,ax=ax,label='Predicted − target uptake')
    save(fig,'residuals')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    columns=['family','quantile','strength','open','closed','intermediate','recovery','ess_fraction','mse','converged',
             'objective_gap','open_conditional_ess','closed_conditional_ess','intermediate_conditional_ess']
    body=['<!doctype html><html><head><meta charset="utf-8"><title>ISO_TRI lower hybrid strengths</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}</style></head><body>',
        '<h1>ISO_TRI: lower strength on wider hybrid graphs</h1>',
        '<p>Six new fits: bandwidth quantiles 0.16, 0.32 and 0.64 at strengths 0.01 and 0.03. The kernels are byte-identical copies of the existing hybrid graphs; no rescaling compensates for lower strength. Target and candidate frames, grouped-uptake fitting, initialisations and convergence rules are fixed.</p>',
        '<p>Target: 40% open, 60% closed, 0% intermediate, generated by frame-wise uptake. All-residue MSE selects fits. Recovery is the base-2 JSD score, not percentage improvement. Higher ESS alone does not establish recovery. Valid paired differences require convergence on both sides; nearest-ESS comparisons use a two-percentage-point tolerance and no interpolation.</p>',
        f'<p>{len(table)} configurations reported; {int(table.converged.sum())} converged. Thirteen original controls are reused.</p>',
        '<h2>All hybrid results</h2>',html_table(hybrids.sort_values(['quantile','strength']).reindex(columns=columns)),
        '<h2>Best converged hybrid by MSE across available settings</h2>',html_table(selected.reindex(columns=columns)),
        '<h2>MSE-selected hybrid within each strength</h2>',html_table(per_strength.reindex(columns=columns)),
        '<h2>Differences from strength 0.1 at the same bandwidth</h2>',html_table(pairs),
        '<h2>Nearest-ESS comparisons</h2>',html_table(matched),
        '<h2>Downloads</h2><p>']
    body += [f'<a href="{p.name}">{p.name}</a> · ' for p in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">manifest</a></p>']
    for name in names:
        body.append(f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">')
    body += ['<h2>All configurations</h2>',html_table(table.reindex(columns=columns)),'</body></html>']
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html',flush=True)
