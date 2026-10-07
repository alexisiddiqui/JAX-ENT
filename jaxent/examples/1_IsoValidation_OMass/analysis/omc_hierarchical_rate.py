"""Frozen hierarchical ISO_TRI graphs fitted with the global mean-rate model."""
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

OUTPUT=t.OUTPUT.with_name('omc_hierarchical_rate')


def specs():
    return [dict(row,stage='rate') for row in t.specs()]


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


def prepare(output):
    if (output/'manifest.json').exists():
        return validate(output)
    t.validate(t.OUTPUT)
    e.load_manifest(e.OUTPUT)
    controls=[row for row in e.collect(e.OUTPUT,'rate') if row['ensemble']=='ISO_TRI']
    grouped=[row for row in t.s.collect(t.OUTPUT) if row['arm']>=25]
    if len(controls)!=19 or len(grouped)!=18 or any('error' in row for row in controls+grouped):
        raise ValueError('19 mean-rate controls and 18 grouped hierarchy archives required')
    hashes={}
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(source.resolve())]=e.digest(source)
        hashes[str(destination.resolve())]=e.digest(destination)
    for name in ('input.npz','graphs.csv','manifest.json'):
        copy(t.OUTPUT/name,output/('source_manifest.json' if name=='manifest.json' else name))
    for row in controls:
        source=e.OUTPUT/'rate/ISO_TRI'/f"arm_{row['arm']:02d}"
        for name in ('fit.npz','complete.json'):
            copy(source/name,output/'fits'/source.name/name)
    for row in grouped:
        source=t.OUTPUT/'fits'/f"arm_{row['arm']:02d}"
        for name in ('fit.npz','complete.json'):
            copy(source/name,output/'grouped'/source.name/name)
    for distance in t.DISTANCES:
        copy(t.OUTPUT/'input.npz',output/distance/'ISO_TRI/input.npz')
        for i in range(6):
            copy(t.OUTPUT/f'hierarchical_{distance}_{i}.npy',output/distance/'ISO_TRI'/f'scalar_logpf_{i}.npy')
    for family in ('scalar_logpf','profile_logpf'):
        for i in range(6):
            copy(e.OUTPUT/'ISO_TRI'/f'{family}_{i}.npy',output/f'{family}_{i}.npy')
    manifest=dict(code=identity(),hashes=hashes,specs=specs(),new_arms=18,rate_controls=19,grouped_controls=18,
                  strength=.1,stage='rate',target='unchanged frame-wise uptake',kernels='byte-identical grouped hierarchy kernels')
    e.write_json(output/'manifest.json',manifest)
    return manifest


def kernel_path(output,row):
    if row['family'].startswith('hierarchical_'):
        name=row['family'].removeprefix('hierarchical_')
        i=int(np.flatnonzero(e.QUANTILES==row['quantile'])[0])
        return output/name/'ISO_TRI'/f'scalar_logpf_{i}.npy'
    return t.kernel_path(output,row)


def run_arm(task):
    output,spec=task
    output=Path(output)
    # The existing fitter recognises scalar_logpf as its general OMC kernel branch.
    internal={key:value for key,value in spec.items() if key not in ('stage','ensemble')}
    internal['family']='scalar_logpf'
    result=e.run_arm((str(output/spec['distance']),'rate','ISO_TRI',internal))
    result.update(family=spec['family'],distance=spec['distance'])
    source=output/spec['distance']/'rate/ISO_TRI'/f"arm_{spec['arm']:02d}"
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
        rows=[row for row in t.s.collect(output) if row['arm']>=25]
        e.write_json(output/'run_status.json',dict(status='complete' if len(rows)==18 else 'incomplete',new_fits=len(rows),
                     converged=sum(row['converged'] for row in rows),numerical_failures=sum('error' in row for row in rows)))
    if args.phase in ('report','all'):
        importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_rate_report').build(output)


if __name__=='__main__':
    main()
