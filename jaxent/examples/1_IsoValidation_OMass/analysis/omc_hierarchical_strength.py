"""Lower strengths on the immutable wider ISO_TRI hierarchical graphs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil

t = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_control')
s, e = t.s, t.e
import numpy as np
import pandas as pd

OUTPUT = t.OUTPUT.with_name('omc_hierarchical_strength')
WIDTHS = (.16, .32, .64)
STRENGTHS = (.01, .03)


def specs():
    return [dict(arm=43+i, family='hierarchical_'+name, distance=name, quantile=q,
                 strength=strength, ensemble='ISO_TRI', stage='uptake')
            for i, (name,q,strength) in enumerate((name,q,strength) for name in t.DISTANCES
                                                  for q in WIDTHS for strength in STRENGTHS)]


def identity():
    return dict(runner=e.digest(__file__), source=t.identity())


def validate(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['code'] != identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path, expected in manifest['hashes'].items():
        if e.digest(path) != expected:
            raise ValueError(f'Changed input or artifact: {path}')
    return manifest


def prepare(output):
    if (output/'manifest.json').exists():
        return validate(output)
    t.validate(t.OUTPUT)
    controls=s.collect(t.OUTPUT)
    if len(controls)!=43 or any('error' in row for row in controls):
        raise ValueError('All 43 source archives required')
    hashes={}
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(source.resolve())]=e.digest(source)
        hashes[str(destination.resolve())]=e.digest(destination)
    for name in ('input.npz','graphs.csv','manifest.json'):
        copy(t.OUTPUT/name,output/('source_manifest.json' if name=='manifest.json' else name))
    for row in controls:
        for name in ('fit.npz','complete.json'):
            copy(t.OUTPUT/'fits'/f"arm_{row['arm']:02d}"/name,output/'fits'/f"arm_{row['arm']:02d}"/name)
    for distance in t.DISTANCES:
        copy(t.OUTPUT/'input.npz',output/distance/'input.npz')
        for i in range(6):
            copy(t.OUTPUT/f'hierarchical_{distance}_{i}.npy',output/distance/f'kernel_{i}.npy')
    # Preserve all kernels needed to independently check original comparator fits.
    for family in ('scalar_logpf','profile_logpf','hybrid_logpf'):
        for i in range(6):
            copy(t.OUTPUT/f'{family}_{i}.npy',output/f'{family}_{i}.npy')
    manifest=dict(code=identity(),hashes=hashes,specs=specs(),new_arms=18,reused_arms=43,
                  strengths=STRENGTHS,bandwidths=WIDTHS,kernels='byte-identical to strength 0.1',
                  stage='uptake',ensemble='ISO_TRI')
    e.write_json(output/'manifest.json',manifest)
    return manifest


def kernel_path(output,row):
    if row['family'].startswith('hierarchical_'):
        distance=row['family'].removeprefix('hierarchical_')
        return s.kernel_path(output/distance,row['quantile'])
    return t.kernel_path(output,row)


def run_arm(task):
    output,row=task
    output=Path(output)
    result=s.run_arm((str(output/row['distance']),row))
    source=output/row['distance']/'fits'/f"arm_{row['arm']:02d}"
    destination=output/'fits'/source.name
    destination.mkdir(parents=True,exist_ok=True)
    # Publish the archive before its completion marker for resumable aggregation.
    for name in ('fit.npz','complete.json','failure.json'):
        if (source/name).exists():
            temporary=destination/(name+'.tmp')
            shutil.copy2(source/name,temporary)
            temporary.replace(destination/name)
    return result


def select(rows):
    table=pd.DataFrame(rows)
    valid=table.loc[table.converged.astype(bool)&np.isfinite(table.mse)]
    return valid.sort_values(['mse','arm'],kind='stable').groupby(['family','strength'],sort=False).head(1)


def comparisons(table):
    result=[]
    for row in table.loc[table.arm>=43].to_dict('records'):
        baseline=table.loc[(table.family==row['family'])&(table.strength==.1)&(table['quantile']==row['quantile'])].iloc[0]
        valid=bool(row['converged'] and baseline.converged)
        result.append(dict(arm=row['arm'],family=row['family'],quantile=row['quantile'],strength=row['strength'],valid_pair=valid,
                           **{'delta_'+metric:row[metric]-baseline[metric] if valid else np.nan
                              for metric in ('mse','recovery','intermediate','ess_fraction')}))
    return pd.DataFrame(result)


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
            for future in as_completed([pool.submit(run_arm,(str(output),row)) for row in specs()]):
                print(json.dumps(future.result()),flush=True)
        rows=[row for row in s.collect(output) if row['arm']>=43]
        e.write_json(output/'run_status.json',dict(status='complete' if len(rows)==18 else 'incomplete',new_fits=len(rows),
                     converged=sum(row['converged'] for row in rows),numerical_failures=sum('error' in row for row in rows)))
    if args.phase in ('report','all'):
        importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_strength_report').build(output)


if __name__=='__main__':
    main()
