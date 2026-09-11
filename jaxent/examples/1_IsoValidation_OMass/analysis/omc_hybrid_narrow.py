"""Three narrower ISO_TRI hybrid bandwidths at fixed reference graph coupling."""
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

OUTPUT = h.OUTPUT.with_name('omc_hybrid_narrow')
MULTIPLIERS = (.25, .5, .75)


def kernel_at_scale(logpf, mask, sigma, reference):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Positive finite bandwidth required')
    values = np.asarray(logpf, float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Finite residue-by-frame profiles required')
    mean = values.mean(axis=0)
    distance = abs(mean[:, None]-mean[None, :])
    raw = np.exp(-np.minimum(.5*(distance/sigma)**2, 80.))
    # First validates the unchanged neighbourhood and scalar kernel.
    kernel, factor = h.match_kernel(mask, raw)
    target = float(reference.sum()-np.trace(reference))
    current = float(kernel.sum()-np.trace(kernel))
    if target <= 0 or not np.isfinite(target):
        raise ValueError('Positive finite reference coupling required')
    scale = target/current
    kernel *= scale
    np.fill_diagonal(kernel, 1.)
    np.testing.assert_allclose(kernel.sum()-np.trace(kernel), target, rtol=1e-12)
    return kernel, factor*scale


def identity():
    return dict(runner=e.digest(__file__), hybrid=h.identity())


def validate(output):
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['code'] != identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path, expected in manifest['hashes'].items():
        if e.digest(path) != expected:
            raise ValueError(f'Input or artifact changed: {path}')
    return manifest


def prepare(output):
    if (output/'manifest.json').exists():
        return validate(output)
    h.validate(h.OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(source.resolve())] = e.digest(source)
        hashes[str(destination.resolve())] = e.digest(destination)
    for name in ('input.npz','neighbourhood.npz','original_graphs.json'):
        copy(h.OUTPUT/'ISO_TRI'/name,output/name)
    copy(h.OUTPUT/'ISO_TRI/hybrid_logpf_0.npy',output/'kernel_1.npy')
    for name in ('fit.npz','complete.json'):
        copy(h.OUTPUT/'uptake/ISO_TRI/arm_19'/name,output/'fits/scale_1'/name)
    copy(h.OUTPUT/'manifest.json',output/'source_manifest.json')
    copy(h.OUTPUT/'source/topology.json',output/'topology.json')
    saved = json.loads((output/'fits/scale_1/complete.json').read_text())
    if e.digest(output/'fits/scale_1/fit.npz') != saved['sha256']:
        raise ValueError('Reference fit checksum mismatch')
    data = np.load(output/'input.npz')
    mask = np.load(output/'neighbourhood.npz')['mask']
    reference = np.load(output/'kernel_1.npy')
    original = json.loads((output/'original_graphs.json').read_text())
    sigma = next(row['sigma'] for row in original if row['family']=='scalar_logpf' and row['quantile']==.02)
    reconstructed,_ = kernel_at_scale(data['logpf'],mask,sigma,reference)
    np.testing.assert_allclose(reconstructed,reference,rtol=1e-12,atol=1e-14)
    graph_rows = []
    groups = data['groups']
    off = ~np.eye(len(mask),dtype=bool)
    cross = groups[:,None] != groups[None,:]
    for multiplier in (*MULTIPLIERS,1.):
        kernel, factor = kernel_at_scale(data['logpf'],mask,sigma*multiplier,reference)
        graph_rows.append(dict(multiplier=multiplier,sigma=sigma*multiplier,factor=factor,
            coupling=float(kernel[off].mean()),cross_state_coupling_fraction=float(kernel[cross].sum()/kernel[off].sum())))
        if multiplier != 1:
            path=output/f'kernel_{multiplier:g}.npy'
            np.save(path,kernel)
            hashes[str(path.resolve())] = e.digest(path)
    pd.DataFrame(graph_rows).to_csv(output/'graphs.csv',index=False)
    hashes[str((output/'graphs.csv').resolve())] = e.digest(output/'graphs.csv')
    manifest=dict(code=identity(),hashes=hashes,ensemble='ISO_TRI',stage='uptake',strength=.1,
        base_quantile=.02,base_sigma=sigma,multipliers=list(MULTIPLIERS),coupling='fixed to original hybrid q=0.02')
    e.write_json(output/'manifest.json',manifest)
    return manifest


def run_arm(task):
    output,multiplier=task
    output=Path(output)
    folder=output/'fits'/f'scale_{multiplier:g}'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'complete.json').exists():
            saved=json.loads((folder/'complete.json').read_text())
            if e.digest(folder/'fit.npz')!=saved['sha256']:
                raise ValueError('Corrupt saved fit')
            return saved['row']
        row=dict(multiplier=multiplier,ensemble='ISO_TRI',stage='uptake',family='hybrid_logpf',strength=.1)
        try:
            data=dict(np.load(output/'input.npz'))
            kernel=np.load(output/f'kernel_{multiplier:g}.npy',mmap_mode='r')
            result=h.checked_fit(data,kernel)
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
        except (FloatingPointError,AssertionError,ValueError) as error:
            row.update(converged=False,mse=None,recovery=None,error=str(error))
            e.write_json(folder/'failure.json',row)
        return row


def collect(output):
    rows=[]
    for multiplier in (*MULTIPLIERS,1.):
        folder=output/'fits'/f'scale_{multiplier:g}'
        if (folder/'complete.json').exists():
            saved=json.loads((folder/'complete.json').read_text())
            if e.digest(folder/'fit.npz')!=saved['sha256']:
                raise ValueError('Fit checksum mismatch')
            rows.append(dict(saved['row'],multiplier=multiplier))
        elif (folder/'failure.json').exists():
            rows.append(json.loads((folder/'failure.json').read_text()))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run','report','all'),default='all')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=3:
        parser.error('--workers must be 1..3')
    output=args.output.resolve()
    if args.phase in ('prepare','all'):
        prepare(output)
    else:
        validate(output)
    if args.phase in ('run','all'):
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=e.worker_init) as pool:
            tasks=[pool.submit(run_arm,(str(output),multiplier)) for multiplier in MULTIPLIERS]
            for future in as_completed(tasks):
                print(json.dumps(future.result()),flush=True)
        rows=collect(output)
        e.write_json(output/'run_status.json',dict(status='complete',new_fits=sum(r['multiplier']!=1 for r in rows),
            new_converged=sum(r['multiplier']!=1 and r['converged'] for r in rows)))
    if args.phase in ('report','all'):
        report=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_narrow_report')
        report.build(output)


if __name__=='__main__':
    main()
