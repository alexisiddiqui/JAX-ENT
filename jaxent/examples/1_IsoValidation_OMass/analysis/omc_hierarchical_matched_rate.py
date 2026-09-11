"""Mean-rate target and mean-rate prediction on frozen ISO_TRI graphs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil

reverse=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_reverse')
t,e=reverse.t,reverse.e
OUTPUT=t.OUTPUT.with_name('omc_hierarchical_matched_rate')
kernel_path=reverse.kernel_path


def specs():
    return [dict(row,stage='rate') for row in reverse.specs()]


def identity():
    return dict(runner=e.digest(__file__),source=reverse.identity())


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
    reverse.validate(reverse.OUTPUT)
    controls=t.s.collect(reverse.OUTPUT)
    if len(controls)!=37 or any('error' in row for row in controls):
        raise ValueError('37 reverse-experiment archives required')
    hashes={}
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(source.resolve())]=e.digest(source)
        hashes[str(destination.resolve())]=e.digest(destination)
    for name in ('input.npz','target.npz','graphs.csv','preflight.csv','manifest.json'):
        copy(reverse.OUTPUT/name,output/('source_manifest.json' if name=='manifest.json' else name))
    for distance in (*t.DISTANCES,'baseline'):
        for source in (reverse.OUTPUT/distance/'ISO_TRI').iterdir():
            if source.suffix in ('.npz','.npy'):
                copy(source,output/distance/'ISO_TRI'/source.name)
    for row in controls:
        for name in ('fit.npz','complete.json'):
            copy(reverse.OUTPUT/'fits'/f"arm_{row['arm']:02d}"/name,output/'grouped'/f"arm_{row['arm']:02d}"/name)
    manifest=dict(code=identity(),hashes=hashes,specs=specs(),new_arms=37,hierarchical_arms=18,baseline_arms=19,
                  target='unchanged global mean-rate reference target',prediction='global mean rate',
                  omc_strength=.1,normalisation='unchanged mean-rate target variance + 1e-8',
                  kernels='byte-identical reverse-experiment kernels',grouped_controls=37)
    e.write_json(output/'manifest.json',manifest)
    return manifest


def run_arm(task):
    output,spec=task
    output=Path(output)
    distance=spec.get('distance','baseline')
    internal={key:value for key,value in spec.items() if key not in ('stage','ensemble')}
    if spec['family'].startswith('hierarchical_'):
        internal['family']='scalar_logpf'
    result=e.run_arm((str(output/distance),'rate','ISO_TRI',internal))
    result.update(family=spec['family'],target_mode='rate')
    source=output/distance/'rate/ISO_TRI'/f"arm_{spec['arm']:02d}"
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
        importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_matched_rate_report').build(output)


if __name__=='__main__':
    main()
