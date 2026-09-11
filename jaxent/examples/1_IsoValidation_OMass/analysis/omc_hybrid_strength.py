"""Lower-strength ISO_TRI fits on immutable wider hybrid kernels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil

h = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_control')
e = h.e
import numpy as np
import pandas as pd

OUTPUT = h.OUTPUT.with_name('omc_hybrid_strength')
WIDTHS = (.16, .32, .64)
STRENGTHS = (.01, .03)


def specs():
    return [dict(arm=25+i, quantile=q, strength=strength, family=h.FAMILY,
                 ensemble='ISO_TRI', stage='uptake')
            for i, (q, strength) in enumerate((q, s) for q in WIDTHS for s in STRENGTHS)]


def identity():
    return dict(runner=e.digest(__file__), hybrid=h.identity())


def validate(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['code']!=identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path, expected in manifest['hashes'].items():
        if e.digest(path)!=expected:
            raise ValueError(f'Changed input or artifact: {path}')
    return manifest


def prepare(output):
    if (output/'manifest.json').exists():
        return validate(output)
    h.validate(h.OUTPUT)
    controls=[row for row in h.collect(h.OUTPUT) if row['ensemble']=='ISO_TRI'
              and row['family'] in (h.FAMILY,'maxent','unregularised')]
    if len(controls)!=13 or any('error' in row for row in controls):
        raise ValueError('Thirteen completed control archives required')
    output.mkdir(parents=True,exist_ok=True)
    hashes={}
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(source.resolve())]=e.digest(source)
        hashes[str(destination.resolve())]=e.digest(destination)
    for name in ('input.npz','neighbourhood.npz','original_graphs.json'):
        copy(h.OUTPUT/'ISO_TRI'/name,output/name)
    # Copy all six kernels for independent validation of reused hybrid controls.
    for i in range(6):
        copy(h.OUTPUT/'ISO_TRI'/f'hybrid_logpf_{i}.npy',output/f'kernel_{i}.npy')
    copy(h.OUTPUT/'source/topology.json',output/'topology.json')
    copy(h.OUTPUT/'manifest.json',output/'source_manifest.json')
    for row in controls:
        source=h.archive(h.OUTPUT,row)
        for name in ('fit.npz','complete.json'):
            copy(source.with_name(name),output/'fits'/f"arm_{row['arm']:02d}"/name)
    manifest=dict(code=identity(),hashes=hashes,specs=specs(),new_arms=6,reused_arms=13,
                  ensemble='ISO_TRI',stage='uptake',kernels='byte-identical original hybrid graphs')
    e.write_json(output/'manifest.json',manifest)
    return manifest


def fit_at_strength(data,kernel,strength,**kwargs):
    if not np.isfinite(strength) or strength<0:
        raise ValueError('Nonnegative finite strength required')
    # The hybrid convenience wrapper fixes strength at 0.1; call the general fitter.
    return e.fit(data,dict(family='scalar_logpf',strength=strength),'uptake',kernel,**kwargs)


def kernel_path(output,quantile):
    index=int(np.flatnonzero(e.QUANTILES==quantile)[0])
    return output/f'kernel_{index}.npy'


def run_arm(task):
    output,spec=task
    output=Path(output)
    folder=output/'fits'/f"arm_{spec['arm']:02d}"
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'complete.json').exists():
            saved=json.loads((folder/'complete.json').read_text())
            if e.digest(folder/'fit.npz')!=saved['sha256']:
                raise ValueError('Corrupt saved fit')
            return saved['row']
        row=dict(spec)
        try:
            data=dict(np.load(output/'input.npz'))
            kernel=np.load(kernel_path(output,spec['quantile']),mmap_mode='r')
            result=fit_at_strength(data,kernel,spec['strength'])
            np.testing.assert_allclose(result['initial_weights'].sum(axis=1),1.,atol=1e-12)
            row.update(converged=bool(result['converged']),mse=float(np.mean((result['prediction']-data['target'])**2)),
                objective=float(result['objective']),objective_gap=float(result['objective_gap']),
                steps=int(result['initial_steps'].max()),
                initial_population_tv=float(abs(e.populations(result['initial_weights'][0],data['groups'])-
                    e.populations(result['initial_weights'][1],data['groups'])).sum()/2),
                **e.metrics(result['weights'],data['groups']))
            with (folder/'fit.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream,**result)
            (folder/'fit.npz.tmp').replace(folder/'fit.npz')
            e.write_json(folder/'complete.json',dict(sha256=e.digest(folder/'fit.npz'),row=row))
        except (FloatingPointError,ValueError,AssertionError) as error:
            row.update(converged=False,mse=None,recovery=None,error=str(error))
            e.write_json(folder/'failure.json',row)
        return row


def collect(output):
    rows=[]
    for folder in sorted((output/'fits').glob('arm_*')):
        if (folder/'complete.json').exists():
            saved=json.loads((folder/'complete.json').read_text())
            if e.digest(folder/'fit.npz')!=saved['sha256']:
                raise ValueError('Fit checksum mismatch')
            rows.append(saved['row'])
        elif (folder/'failure.json').exists():
            rows.append(json.loads((folder/'failure.json').read_text()))
    return rows


def select(rows,per_strength=False):
    table=pd.DataFrame(rows)
    valid=table.loc[table.converged.astype(bool)&np.isfinite(table.mse)&(table.family==h.FAMILY)]
    ordered=valid.sort_values(['mse','quantile','strength'],kind='stable')
    return ordered.groupby('strength',sort=False).head(1) if per_strength else ordered.head(1)


def smoke(output):
    data=dict(np.load(output/'input.npz'))
    kernel=np.load(kernel_path(output,.16))
    result=fit_at_strength(data,kernel,.01,checkpoints=(20,),window=10)
    if not np.isfinite(result['objective']):
        raise FloatingPointError('Nonfinite smoke fit')
    return dict(finite=True,steps=20,strength=.01,quantile=.16)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run','report','all'),default='all')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=6:
        parser.error('--workers must be 1..6')
    output=args.output.resolve()
    if args.phase in ('prepare','all'):
        prepare(output)
    else:
        validate(output)
    if args.smoke or args.phase in ('run','all'):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=e.worker_init) as pool:
            e.write_json(output/'smoke.json',pool.submit(smoke,output).result())
        if args.smoke:
            return
    if args.phase in ('run','all'):
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=e.worker_init) as pool:
            for future in as_completed([pool.submit(run_arm,(str(output),spec)) for spec in specs()]):
                print(json.dumps(future.result()),flush=True)
        rows=[row for row in collect(output) if row['arm']>=25]
        e.write_json(output/'run_status.json',dict(status='complete',new_fits=len(rows),
            new_converged=sum(row['converged'] for row in rows),numerical_failures=sum('error' in row for row in rows)))
    if args.phase in ('report','all'):
        report=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_strength_report')
        report.build(output)


if __name__=='__main__':
    main()
