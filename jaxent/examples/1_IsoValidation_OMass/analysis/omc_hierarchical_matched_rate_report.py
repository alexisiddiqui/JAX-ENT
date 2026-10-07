"""Consistent mean-rate fit report and same-target grouped-model comparisons."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_matched_rate')
base=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_reverse_report')
import numpy as np
import pandas as pd


def build(output):
    output=Path(output)
    base.build(output,experiment=s,mode='rate')
    grouped={}
    for marker in sorted((output/'grouped').glob('arm_*/complete.json')):
        saved=json.loads(marker.read_text())
        if s.e.digest(marker.with_name('fit.npz'))!=saved['sha256']:
            raise ValueError('Corrupt paired grouped archive')
        grouped[saved['row']['arm']]=saved['row']
    pairs=[]
    convergence=[]
    for row in s.t.s.collect(output):
        if 'error' not in row:
            saved=np.load(output/'fits'/f"arm_{row['arm']:02d}"/'fit.npz')
            plateau=bool(np.all(saved['initial_relative']<=.01))
            agreement=bool(row['objective_gap']<=.01)
            if bool(row['converged'])!=(plateau and agreement):
                raise ValueError('Convergence flags disagree with saved diagnostics')
            convergence.append(dict(arm=row['arm'],family=row['family'],quantile=row['quantile'],
                converged=row['converged'],plateau_pass=plateau,start_agreement_pass=agreement,
                start0_relative_change=float(saved['initial_relative'][0]),start1_relative_change=float(saved['initial_relative'][1]),
                absolute_objective_gap=float(np.ptp(saved['initial_objectives'])),relative_objective_gap=row['objective_gap'],
                initial_population_tv=row['initial_population_tv'],
                start0_steps=int(saved['initial_steps'][0]),start1_steps=int(saved['initial_steps'][1])))
        other=grouped[row['arm']]
        valid=bool(row['converged'] and other['converged'])
        pairs.append(dict(family=row['family'],arm=row['arm'],quantile=row['quantile'],strength=row['strength'],valid_pair=valid,
                          **{'delta_'+key:row[key]-other[key] if valid else np.nan
                             for key in ('mse','recovery','intermediate','ess_fraction')}))
    table=pd.DataFrame(pairs)
    table.to_csv(output/'paired_forward_models.csv',index=False)
    pd.DataFrame(grouped.values()).to_csv(output/'grouped_controls.csv',index=False)
    diagnostics=pd.DataFrame(convergence)
    diagnostics.to_csv(output/'convergence.csv',index=False)
    text=(output/'index.html').read_text()
    addition='<h2>Same mean-rate target: mean-rate minus grouped-uptake fitting</h2><p>Both experiments use byte-identical targets, inputs, graph kernels and loss scaling. Deltas are reported only when both fits converged. A missing delta denotes an unresolved pair, not no effect.</p>'
    addition+='<p><a href="paired_forward_models.csv">Paired comparisons CSV</a> · <a href="grouped_controls.csv">Grouped controls CSV</a></p>'
    addition+='<div class="scroll">'+table.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    addition+='<h2>Convergence diagnostics</h2><p>The unchanged criteria require at most 1% relative objective change over the last 250 steps for both starts and at most 1% relative objective disagreement. Very small objective values can still fail these relative criteria. No absolute-tolerance override is applied.</p><p><a href="convergence.csv">Convergence CSV</a></p>'
    addition+='<div class="scroll">'+diagnostics.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
    (output/'index.html').write_text(text.replace('</body>',addition+'</body>'))
