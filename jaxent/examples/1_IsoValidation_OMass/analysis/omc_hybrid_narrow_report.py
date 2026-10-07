"""Report the fixed-coupling ISO_TRI hybrid bandwidth extension."""
from __future__ import annotations

import importlib
from pathlib import Path

n=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_narrow')
r=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')
e=n.e

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(output):
    output=Path(output)
    n.validate(output)
    rows=n.collect(output)
    data=dict(np.load(output/'input.npz'))
    mask=np.load(output/'neighbourhood.npz')['mask']
    reference=np.load(output/'kernel_1.npy')
    off=~np.eye(len(mask),dtype=bool)
    initial,residuals=[],[]
    residue_keys=e.layout(output/'topology.json')
    for row in rows:
        if 'error' in row:
            continue
        multiplier=row['multiplier']
        result=np.load(output/'fits'/f'scale_{multiplier:g}'/'fit.npz')
        kernel=np.load(output/f'kernel_{multiplier:g}.npy',mmap_mode='r')
        np.testing.assert_allclose(kernel[off].mean(),reference[off].mean(),rtol=1e-12)
        np.testing.assert_array_equal(kernel[off & ~mask],0.)
        np.testing.assert_allclose(result['initial_weights'].sum(axis=1),1.,atol=1e-12)
        if not np.isfinite(result['initial_weights']).all():
            raise ValueError('Nonfinite saved weights')
        prediction=e.predict(data['rates'],result['weights'],data['groups'],'uptake')
        np.testing.assert_allclose(prediction,result['prediction'],rtol=1e-10,atol=1e-12)
        mse=float(np.mean((prediction-data['target'])**2))
        objective=mse/float(data['scale'])+.1*r.pairwise_penalty(result['weights'],kernel)
        np.testing.assert_allclose(objective,row['objective'],rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(mse,row['mse'],rtol=1e-10,atol=1e-14)
        for start,w in enumerate(result['initial_weights']):
            initial.append(dict(multiplier=multiplier,start=start,**e.metrics(w,data['groups'])))
        for t,time in enumerate(e.TIMES):
            for j,(chain,residue) in enumerate(residue_keys):
                residuals.append(dict(multiplier=multiplier,chain=chain,residue=residue[0],time_min=time,
                    target=data['target'][t,j],prediction=prediction[t,j],residual=prediction[t,j]-data['target'][t,j]))
    table=pd.DataFrame(rows).merge(pd.read_csv(output/'graphs.csv'),on='multiplier',validate='one_to_one').sort_values('multiplier')
    valid=table.loc[table.converged.astype(bool)&np.isfinite(table.mse)]
    selected=valid.sort_values(['mse','multiplier'],kind='stable').head(1)
    reference_row=table.loc[table.multiplier==1].iloc[0]
    for metric in ('mse','recovery','tv','ess_fraction','intermediate'):
        table['delta_'+metric]=(table[metric]-reference_row[metric]).where(table.converged.astype(bool))
    table.to_csv(output/'fits.csv',index=False)
    selected.to_csv(output/'selected.csv',index=False)
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
    fig,axes=plt.subplots(1,4,figsize=(19,4.5))
    finite=table.loc[np.isfinite(table.mse)]
    unresolved=finite.loc[~finite.converged.astype(bool)]
    for ax,metric,label in zip(axes[:3],('recovery','ess_fraction','mse'),('Recovery score (%)','ESS fraction (%)','All-residue MSE')):
        factor=100 if metric=='ess_fraction' else 1
        ax.plot(finite.multiplier,finite[metric]*factor,'.-')
        ax.scatter(unresolved.multiplier,unresolved[metric]*factor,marker='x',color='red',s=70)
        ax.set(xlabel='Bandwidth / reference bandwidth',ylabel=label)
        ax.set_xticks([.25,.5,.75,1.])
    bars=np.vstack([e.TRUTH,finite[['open','closed','intermediate']].to_numpy()])
    bottom=np.zeros(len(bars))
    for i,name in enumerate(('Open','Closed','Intermediate')):
        axes[3].bar(np.arange(len(bars)),bars[:,i],bottom=bottom,label=name)
        bottom+=bars[:,i]
    axes[3].set_xticks(np.arange(len(bars)),['Target']+[f'{v:g}×' for v in finite.multiplier])
    axes[3].set(ylabel='Population',ylim=(0,1))
    axes[3].legend(fontsize=8)
    fig.suptitle('ISO_TRI hybrid: fixed neighbour mask and total coupling; red crosses are unresolved')
    save(fig,'bandwidth_comparison')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,(state,name) in enumerate(zip(e.STATES,('Open','Closed','Intermediate'))):
        keep=data['groups']==state
        for coordinate,ref in enumerate(('Open','Closed')):
            ax=axes[coordinate,col]
            x=data['rmsd'][keep,coordinate]
            order=np.argsort(x)
            ax.plot(x[order],np.arange(1,len(x)+1)/len(x),'--',color='grey',label='Uniform within state')
            for row in valid.to_dict('records'):
                w=np.load(output/'fits'/f"scale_{row['multiplier']:g}"/'fit.npz')['weights'][keep]
                ax.plot(x[order],np.cumsum(w[order])/w.sum(),label=f"{row['multiplier']:g}×")
            ax.set(title=name,xlabel=f'{ref}-reference RMSD (Å)',ylabel='Conditional cumulative weight',ylim=(0,1))
    axes[0,0].legend(fontsize=8)
    save(fig,'coverage')
    fig,axes=plt.subplots(len(finite),1,figsize=(13,2.4*len(finite)),squeeze=False)
    arrays=[np.load(output/'fits'/f'scale_{v:g}'/'fit.npz')['prediction']-data['target'] for v in finite.multiplier]
    limit=max(float(abs(a).max()) for a in arrays)
    for ax,row,residual in zip(axes[:,0],finite.to_dict('records'),arrays):
        im=ax.imshow(residual,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
        ticks=np.arange(0,len(residue_keys),40)
        ax.set_xticks(ticks,[residue_keys[i][1][0] for i in ticks])
        ax.set_yticks(np.arange(len(e.TIMES)),[f'{t:g}' for t in e.TIMES])
        ax.set(title=f"{row['multiplier']:g}×; converged: {row['converged']}",xlabel='Residue number',ylabel='Time (min)')
        fig.colorbar(im,ax=ax,label='Predicted − target uptake')
    save(fig,'residuals')
    def html_table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</div>'
    columns=['multiplier','sigma','open','closed','intermediate','recovery','ess_fraction','mse','converged',
             'objective_gap','open_conditional_ess','closed_conditional_ess','intermediate_conditional_ess']
    body=['<!doctype html><html><head><meta charset="utf-8"><title>ISO_TRI narrower hybrid bandwidths</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{width:100%}</style></head><body>',
        '<h1>ISO_TRI: narrower hybrid bandwidths</h1>',
        '<p>Three new bandwidths are 0.25×, 0.5× and 0.75× the original q=0.02 scalar bandwidth. These multipliers are not new distance quantiles. The 1× fit is reused. The 20-neighbour profile mask, total off-diagonal coupling, OMC strength 0.1, frame-wise target and grouped-uptake fitting are fixed.</p>',
        '<p>Target populations: open 40%, closed 60%, intermediate 0%. Recovery is the existing base-2 JSD score, not percentage improvement. Selection uses all-residue MSE among converged fits; numerical agreement between starts does not establish unique populations. Unresolved fits are displayed but excluded from selection and valid reference differences.</p>',
        '<h2>Results</h2>',html_table(table.reindex(columns=columns)),
        '<h2>MSE-selected converged fit</h2>',html_table(selected.reindex(columns=columns)),
        '<h2>Differences from the 1× reference</h2>',html_table(table[['multiplier','converged','delta_mse','delta_recovery','delta_tv','delta_ess_fraction','delta_intermediate']]),
        '<h2>Graph diagnostics</h2>',html_table(pd.read_csv(output/'graphs.csv')),
        '<h2>Downloads</h2><p>']
    body += [f'<a href="{p.name}">{p.name}</a> · ' for p in sorted(output.glob('*.csv'))]
    body += ['<a href="manifest.json">manifest</a></p>']
    for name in names:
        body.append(f'<h2>{name.replace("_"," ")}</h2><a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png" alt="{name}">')
    body.append('</body></html>')
    (output/'index.html').write_text('\n'.join(body))
    print(output/'index.html',flush=True)
