"""Global mean-rate reference target fitted with grouped uptake on ISO_TRI."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil

t=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_control')
e=t.e
import numpy as np
import pandas as pd

OUTPUT=t.OUTPUT.with_name('omc_hierarchical_reverse')


def specs():
    return [dict(row,ensemble='ISO_TRI',stage='uptake') for row in e.specs()]+t.specs()


def identity():
    return dict(runner=e.digest(__file__),source=t.identity())


def validate(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['code']!=identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path,expected in manifest['hashes'].items():
        if e.digest(path)!=expected:
            raise ValueError(f'Changed source or artifact: {path}')
    return manifest


def reference_target(features,weights,groups):
    logpf=.35*np.asarray(features['heavy_contacts'],float)+2*np.asarray(features['acceptor_contacts'],float)
    rates=np.asarray(features['k_ints'],float)[:,None]*np.exp(-logpf)
    if rates.shape[1]!=len(weights) or not np.isfinite(rates).all():
        raise ValueError('Invalid reference frame/rate alignment')
    np.testing.assert_allclose(np.sum(weights),1.,atol=1e-12)
    target=e.predict(rates,weights,groups,'rate')
    if not np.isfinite(target).all() or np.var(target)<=1e-8:
        raise ValueError('Degenerate target')
    return target,rates


def prepare(output):
    if (output/'manifest.json').exists():
        return validate(output)
    t.validate(t.OUTPUT)
    e.load_manifest(e.OUTPUT)
    hashes={}
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(source.resolve())]=e.digest(source)
        hashes[str(destination.resolve())]=e.digest(destination)
    copy(t.OUTPUT/'input.npz',output/'original_input.npz')
    copy(t.OUTPUT/'graphs.csv',output/'graphs.csv')
    copy(t.OUTPUT/'manifest.json',output/'source_manifest.json')
    source=e.ROOT/'data/_self_consistent_target_features'
    for name in ('features_open_closed.npz','topology_open_closed.json','manifest.json'):
        copy(source/name,output/'reference'/name)
    copy(e.OUTPUT/'target.npz',output/'reference/original_target.npz')
    provenance=json.loads((source/'manifest.json').read_text())
    if provenance['contact_mode']!='hard' or provenance['kint_unit']!='min^-1':
        raise ValueError('Reference forward convention mismatch')
    old=np.load(output/'reference/original_target.npz')
    features=np.load(output/'reference/features_open_closed.npz')
    weights,groups=old['reference_weights'],old['reference_groups']
    expected_groups=np.repeat([0,1],[provenance['open_frames'],provenance['closed_frames']])
    np.testing.assert_array_equal(groups,expected_groups)
    np.testing.assert_allclose([weights[groups==0].sum(),weights[groups==1].sum()],[.4,.6],atol=1e-12)
    target,rates=reference_target(features,weights,groups)
    original=e.predict(rates,weights,groups,'frame_uptake')
    np.testing.assert_allclose(original,old['target'],rtol=1e-12,atol=1e-12)
    # Uptake is concave in rate, so global mean-rate uptake bounds mixtures above.
    if np.min(target-original)<-1e-12:
        raise AssertionError('Mean-rate target violates expected mixture inequality')
    data=dict(np.load(output/'original_input.npz'))
    data.update(target=target,scale=float(np.var(target)+1e-8))
    np.savez_compressed(output/'input.npz',**data)
    np.savez_compressed(output/'target.npz',target=target,reference_weights=weights,reference_groups=groups,
                        reference_mean_rates=rates@weights,times=e.TIMES)
    diagnostics=[]
    for mode in ('frame_uptake','uptake','rate'):
        prediction=e.predict(rates,weights,groups,mode)
        diagnostics.append(dict(ensemble='reference',weights='40:60 reference',prediction=mode,mse=float(np.mean((prediction-target)**2))))
        prediction=e.predict(data['rates'],data['truth_weights'],data['groups'],mode)
        diagnostics.append(dict(ensemble='ISO_TRI',weights='true masses, uniform within state',prediction=mode,mse=float(np.mean((prediction-target)**2))))
    pd.DataFrame(diagnostics).to_csv(output/'preflight.csv',index=False)
    for distance in (*t.DISTANCES,'baseline'):
        copy(output/'input.npz',output/distance/'ISO_TRI/input.npz')
        for i in range(6):
            if distance=='baseline':
                for family in ('scalar_logpf','profile_logpf'):
                    copy(t.OUTPUT/f'{family}_{i}.npy',output/distance/'ISO_TRI'/f'{family}_{i}.npy')
            else:
                copy(t.OUTPUT/f'hierarchical_{distance}_{i}.npy',output/distance/'ISO_TRI'/f'scalar_logpf_{i}.npy')
    for path in (output/'target.npz',output/'preflight.csv'):
        hashes[str(path.resolve())]=e.digest(path)
    manifest=dict(code=identity(),hashes=hashes,specs=specs(),new_arms=37,hierarchical_arms=18,baseline_arms=19,
                  target='global mean-rate reference uptake',prediction='grouped uptake',target_populations=[.4,.6,0.],
                  omc_strength=.1,normalisation='new target variance + 1e-8',kernels='unchanged hierarchy and baseline kernels')
    e.write_json(output/'manifest.json',manifest)
    return manifest


def kernel_path(output,row):
    distance=row.get('distance','baseline')
    family='scalar_logpf' if row['family'].startswith('hierarchical_') else row['family']
    i=int(np.flatnonzero(e.QUANTILES==row['quantile'])[0])
    return output/distance/'ISO_TRI'/f'{family}_{i}.npy'


def run_arm(task):
    output,spec=task
    output=Path(output)
    distance=spec.get('distance','baseline')
    internal={key:value for key,value in spec.items() if key not in ('stage','ensemble')}
    if spec['family'].startswith('hierarchical_'):
        internal['family']='scalar_logpf'
    result=e.run_arm((str(output/distance),'uptake','ISO_TRI',internal))
    result['family']=spec['family']
    result['target_mode']='rate'
    source=output/distance/'uptake/ISO_TRI'/f"arm_{spec['arm']:02d}"
    destination=output/'fits'/source.name
    destination.mkdir(parents=True,exist_ok=True)
    if (source/'complete.json').exists():
        temporary=destination/'fit.npz.tmp'
        shutil.copy2(source/'fit.npz',temporary)
        temporary.replace(destination/'fit.npz')
        e.write_json(destination/'complete.json',dict(sha256=e.digest(destination/'fit.npz'),row=result))
    else:
        e.write_json(destination/'failure.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run','report','all'),default='all')
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if not 1<=args.workers<=10:
        parser.error('--workers must be 1..10')
    output=args.output.resolve()
    if args.phase in ('prepare','all'):
        prepare(output)
    else:
        validate(output)
    if args.phase in ('run','all'):
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=e.worker_init) as pool:
            for future in as_completed([pool.submit(run_arm,(str(output),spec)) for spec in specs()]):
                print(json.dumps(future.result()),flush=True)
        rows=t.s.collect(output)
        e.write_json(output/'run_status.json',dict(status='complete' if len(rows)==37 else 'incomplete',fits=len(rows),
                     converged=sum(row['converged'] for row in rows),numerical_failures=sum('error' in row for row in rows)))
    if args.phase in ('report','all'):
        importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_reverse_report').build(output)


if __name__=='__main__':
    main()
