"""Profile kNN connectivity with scalar-logPF edge weights on frozen ISO inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil

e = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_control')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

OUTPUT = e.OUTPUT.with_name('omc_hybrid_control')
FAMILY = 'hybrid_logpf'


def neighbourhood(logpf, k=20):
    values = np.asarray(logpf, float)
    if values.ndim != 2 or not np.isfinite(values).all() or not 0 < k < values.shape[1]:
        raise ValueError('Finite residue-by-frame profiles and 0 < k < N required')
    distances = squareform(pdist(values.T)) / np.sqrt(values.shape[0])
    np.fill_diagonal(distances, np.inf)
    # Stable ordering resolves equal distances by the original frame index.
    nearest = np.argsort(distances, axis=1, kind='stable')[:, :k]
    mask = np.zeros(distances.shape, bool)
    mask[np.arange(len(mask))[:, None], nearest] = True
    return mask | mask.T


def match_kernel(mask, scalar):
    mask, scalar = np.asarray(mask), np.asarray(scalar, float)
    if (mask.dtype != bool or mask.shape != scalar.shape or scalar.ndim != 2
            or scalar.shape[0] != scalar.shape[1] or not np.isfinite(scalar).all()
            or (scalar < 0).any() or not np.array_equal(mask, mask.T)
            or np.diag(mask).any() or not np.allclose(scalar, scalar.T)):
        raise ValueError('Symmetric nonnegative scalar kernel and loop-free boolean mask required')
    raw = np.where(mask, scalar, 0.)
    reference_sum = scalar.sum() - np.trace(scalar)
    if raw.sum() <= 0 or reference_sum <= 0:
        raise ValueError('Positive retained and reference coupling required')
    factor = float(reference_sum / raw.sum())
    kernel = raw * factor
    np.fill_diagonal(kernel, 1.)
    np.testing.assert_allclose(kernel.sum()-np.trace(kernel), reference_sum, rtol=1e-12)
    return kernel, factor


def identity():
    return dict(runner=e.digest(__file__), original=e.identity())


def validate(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['code'] != identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path, expected in manifest['hashes'].items():
        if e.digest(path) != expected:
            raise ValueError(f'Changed input or artifact: {path}')
    return manifest


def prepare(output, source=e.OUTPUT):
    if (output / 'manifest.json').exists():
        return validate(output)
    e.load_manifest(source)
    controls = e.collect(source, 'uptake')
    if len(controls) != 38 or any(row.get('error') for row in controls):
        raise ValueError('All 38 completed grouped-control archives required')
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def copy(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[str(src.resolve())] = e.digest(src)
        hashes[str(dst.resolve())] = e.digest(dst)
    for name in ('manifest.json', 'target.npz', 'preflight.csv'):
        copy(source / name, output / 'source' / name)
    copy(e.ROOT / 'data/_self_consistent_target_features/topology_open_closed.json',
         output / 'source/topology.json')
    for row in controls:
        folder = Path('uptake') / row['ensemble'] / f"arm_{row['arm']:02d}"
        for name in ('fit.npz', 'complete.json'):
            copy(source / folder / name, output / 'controls' / folder / name)
    graph_rows, component_rows = [], []
    for ensemble in ('ISO_BI', 'ISO_TRI'):
        folder = output / ensemble
        copy(source / ensemble / 'input.npz', folder / 'input.npz')
        copy(source / ensemble / 'graphs.json', folder / 'original_graphs.json')
        data = np.load(folder / 'input.npz')
        mask = neighbourhood(data['logpf'])
        count, components = connected_components(mask, directed=False)
        np.savez_compressed(folder / 'neighbourhood.npz', mask=mask, components=components,
                            degree=mask.sum(axis=1))
        for component in range(count):
            selected = components == component
            component_rows.append(dict(ensemble=ensemble, component=component, size=int(selected.sum()),
                **{name: int(np.sum(selected & (data['groups'] == label)))
                   for label, name in zip(e.STATES, ('open', 'closed', 'intermediate'))}))
        original = json.loads((folder / 'original_graphs.json').read_text())
        for i, q in enumerate(e.QUANTILES):
            for family in e.FAMILIES[2:]:
                copy(source / ensemble / f'{family}_{i}.npy', folder / f'{family}_{i}.npy')
            scalar = np.load(folder / f'scalar_logpf_{i}.npy')
            kernel, factor = match_kernel(mask, scalar)
            np.save(folder / f'{FAMILY}_{i}.npy', kernel)
            n = len(mask)
            total = float(kernel.sum()-np.trace(kernel))
            groups = data['groups']
            cross = groups[:, None] != groups[None, :]
            intermediate_edges = (groups[:, None] == -1) != (groups[None, :] == -1)
            sigma = next(x['sigma'] for x in original
                         if x['family'] == 'scalar_logpf' and x['quantile'] == q)
            graph_rows.append(dict(ensemble=ensemble, quantile=float(q), sigma=sigma,
                components=count, edge_fraction=float(mask.sum()/(n*(n-1))),
                degree_min=int(mask.sum(axis=1).min()), degree_median=float(np.median(mask.sum(axis=1))),
                degree_max=int(mask.sum(axis=1).max()), factor=factor,
                reference_coupling=float((scalar.sum()-np.trace(scalar))/(n*(n-1))),
                coupling=total/(n*(n-1)), cross_state_coupling_fraction=float(kernel[cross].sum()/total),
                intermediate_target_coupling_fraction=float(kernel[intermediate_edges].sum()/total)))
        for path in folder.iterdir():
            hashes[str(path.resolve())] = e.digest(path)
    pd.DataFrame(graph_rows).to_csv(output / 'graphs.csv', index=False)
    pd.DataFrame(component_rows).to_csv(output / 'components.csv', index=False)
    for name in ('graphs.csv', 'components.csv'):
        hashes[str((output / name).resolve())] = e.digest(output / name)
    manifest = dict(code=identity(), hashes=hashes, source=str(source), neighbours=20,
                    symmetrisation='union', tie_break='frame_index', strength=.1,
                    quantiles=e.QUANTILES.tolist(), stage='uptake', reused_arms=38, new_arms=12)
    e.write_json(output / 'manifest.json', manifest)
    return manifest


def checked_fit(data, kernel, **kwargs):
    # Preserve the original fitter and explicitly select its OMC penalty branch.
    return e.fit(data, dict(family='scalar_logpf', strength=.1), 'uptake', kernel, **kwargs)


def run_arm(task):
    output, ensemble, i = task
    output = Path(output)
    folder = output / 'uptake' / ensemble / f'arm_{19+i:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'complete.json').exists():
            saved = json.loads((folder / 'complete.json').read_text())
            if e.digest(folder / 'fit.npz') != saved['sha256']:
                raise ValueError('Corrupt saved fit')
            return saved['row']
        row = dict(stage='uptake', ensemble=ensemble, family=FAMILY, arm=19+i,
                   strength=.1, quantile=float(e.QUANTILES[i]))
        try:
            data = dict(np.load(output / ensemble / 'input.npz'))
            kernel = np.load(output / ensemble / f'{FAMILY}_{i}.npy', mmap_mode='r')
            result = checked_fit(data, kernel)
            np.testing.assert_allclose(result['initial_weights'].sum(axis=1), 1., atol=1e-12)
            row.update(converged=bool(result['converged']), objective=float(result['objective']),
                mse=float(np.mean((result['prediction']-data['target'])**2)),
                objective_gap=float(result['objective_gap']), steps=int(result['initial_steps'].max()),
                initial_population_tv=float(abs(e.populations(result['initial_weights'][0], data['groups'])-
                    e.populations(result['initial_weights'][1], data['groups'])).sum()/2),
                **e.metrics(result['weights'], data['groups']))
            with (folder / 'fit.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, **result)
            (folder / 'fit.npz.tmp').replace(folder / 'fit.npz')
            e.write_json(folder / 'complete.json', dict(sha256=e.digest(folder / 'fit.npz'), row=row))
        except (FloatingPointError, AssertionError, ValueError) as error:
            row.update(converged=False, mse=None, recovery=None, error=str(error))
            e.write_json(folder / 'failure.json', row)
        return row


def collect(output):
    return e.collect(output / 'controls', 'uptake') + e.collect(output, 'uptake')


def archive(output, row):
    base = output if row['family'] == FAMILY else output / 'controls'
    return base / 'uptake' / row['ensemble'] / f"arm_{row['arm']:02d}" / 'fit.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'report', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not 1 <= args.workers <= 10:
        parser.error('--workers must be 1..10')
    if args.phase in ('prepare', 'all'):
        prepare(output)
    else:
        validate(output)
    if args.smoke or args.phase in ('run', 'all'):
        # Run smoke in a spawned worker so parent affinity remains unchanged.
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=e.worker_init) as pool:
            smoke = pool.submit(smoke_check, output).result()
        e.write_json(output / 'smoke.json', smoke)
        if args.smoke:
            return
    if args.phase in ('run', 'all'):
        tasks = [(str(output), ensemble, i) for ensemble in ('ISO_BI', 'ISO_TRI') for i in range(6)]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=e.worker_init) as pool:
            for future in as_completed([pool.submit(run_arm, task) for task in tasks]):
                print(json.dumps(future.result()), flush=True)
        rows = e.collect(output, 'uptake')
        e.write_json(output / 'run_status.json', dict(status='complete', completed=len(rows),
            converged=sum(bool(r['converged']) for r in rows), numerical_failures=sum('error' in r for r in rows)))
    if args.phase in ('report', 'all'):
        report = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')
        report.build(output)


def smoke_check(output):
    data = dict(np.load(output / 'ISO_BI/input.npz'))
    kernel = np.load(output / 'ISO_BI/hybrid_logpf_0.npy')
    result = checked_fit(data, kernel, checkpoints=(20,), window=10)
    if not np.isfinite(result['objective']):
        raise FloatingPointError('Smoke objective is nonfinite')
    return dict(finite=True, steps=20)


if __name__ == '__main__':
    main()
