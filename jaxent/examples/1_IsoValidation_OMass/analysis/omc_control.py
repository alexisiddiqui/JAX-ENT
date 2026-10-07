"""Frozen ISO OMC comparison: frame-uptake target, grouped then gated mean rate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iso-omc-matplotlib')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OUTPUT = ROOT / 'analysis/omc_control'
TIMES = np.array([0.167, 1., 10., 60., 120.])
QUANTILES = np.array([.02, .04, .08, .16, .32, .64])
STATES = (0, 1, -1)
TRUTH = np.array([.4, .6, 0.])
FAMILIES = ('unregularised', 'maxent', 'scalar_logpf', 'profile_logpf')
SEED = 20260910


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity():
    paths = [Path(__file__), REPO / 'jaxent/src/models/HDX/forward.py',
             REPO / 'jaxent/src/models/config.py', REPO / 'jaxent/src/models/func/contacts.py',
             REPO / 'jaxent/src/models/HDX/BV/forwardmodel.py',
             ROOT / 'fitting/jaxENT/featurise_ISO_TRI_BI.py',
             REPO / 'jaxent/examples/common/analysis/clustering.py']
    return {str(p): digest(p) for p in paths}


def layout(path):
    items = json.loads(Path(path).read_text())['topologies']
    keys = [(x['chain'], tuple(x['residues'])) for x in items]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate residue identifiers')
    return keys


def alignment(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError('Target and candidate residue layouts differ')
    return np.array([candidate.index(k) for k in reference])


def predict(rates, weights, groups, mode, times=TIMES, xp=np):
    """Existing ISO semantics, with stable expm1 and safe zero-mass groups."""
    times = xp.asarray(times)
    if mode == 'frame_uptake':
        return xp.stack([-xp.expm1(-t * rates) @ weights for t in times])
    if mode == 'rate':
        return -xp.expm1(-times[:, None] * (rates @ weights)[None, :])
    if mode != 'uptake':
        raise ValueError(mode)
    result = xp.zeros((len(times), rates.shape[0]))
    for label in STATES:
        gw = xp.where(groups == label, weights, 0.)
        mass = xp.sum(gw)
        mean_rate = rates @ (gw / xp.where(mass > 0, mass, 1.))
        result = result + mass * -xp.expm1(-times[:, None] * mean_rate[None, :])
    return result


def specs():
    result = [dict(family='unregularised', strength=0., quantile=None)]
    result += [dict(family='maxent', strength=s, quantile=None)
               for s in (1e-5, 1e-4, 1e-3, 1e-2, .1, 1.)]
    result += [dict(family=f, strength=.1, quantile=float(q))
               for f in FAMILIES[2:] for q in QUANTILES]
    return [dict(arm=i, **row) for i, row in enumerate(result)]


def populations(weights, groups):
    return np.array([weights[groups == k].sum() for k in STATES])


def recovery(weights, groups):
    from jaxent.examples.common.analysis.clustering import calculate_recovery_JSD
    jsd, _ = calculate_recovery_JSD(groups, weights,
        {'open': .4, 'closed': .6, 'intermediate': 0.},
        {0: 'open', 1: 'closed', -1: 'intermediate'})
    if not np.isfinite(jsd) or jsd < -1e-12 or jsd > 1 + 1e-12:
        raise ValueError('Invalid population JSD')
    return float(100 * (1 - np.sqrt(np.clip(jsd, 0., 1.))))



def metrics(weights, groups):
    p = populations(weights, groups)
    result = dict(zip(('open', 'closed', 'intermediate'), p.tolist()))
    result.update(recovery=recovery(weights, groups), tv=float(abs(p - TRUTH).sum() / 2),
                  ess_fraction=float(1 / (len(weights) * np.sum(weights ** 2))))
    for label, name in zip(STATES, ('open', 'closed', 'intermediate')):
        w = weights[groups == label]
        result[name + '_conditional_ess'] = float(w.sum() ** 2 / np.sum(w*w)) if np.any(w) else None
    return result


def select(rows):
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    valid = table.loc[table.converged.astype(bool) & np.isfinite(table.mse)]
    return valid.sort_values(['mse', 'arm'], kind='stable').groupby(
        ['stage', 'ensemble', 'family'], sort=False).head(1)


def gate(rows):
    selected = select(rows)
    numerical = not selected.empty and all(
        ((selected.ensemble == e) & (selected.family == f)).any()
        for e in ('ISO_BI', 'ISO_TRI') for f in FAMILIES)
    eligible = selected.loc[(selected.ensemble == 'ISO_TRI') &
        selected.family.isin(FAMILIES[2:]) & (selected.recovery > 50)] if not selected.empty else selected
    return dict(passed=bool(numerical and len(eligible)), numerical_pass=bool(numerical),
                threshold=50., metric='100*(1-sqrt(base2_JSD))', strict=True,
                qualifying=eligible[['ensemble', 'family', 'arm', 'recovery']].to_dict('records') if len(eligible) else [])


def load_manifest(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['code'] != identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for name, expected in manifest['hashes'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Input/artifact changed: {name}')
    return manifest


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        return load_manifest(output)
    source = ROOT / 'data/_self_consistent_target_features'
    provenance = json.loads((source / 'manifest.json').read_text())
    assert provenance['contact_mode'] == 'hard' and provenance['kint_unit'] == 'min^-1'
    for path, expected in provenance['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Reference trajectory mismatch: {path}')
    feature_dir = output / 'features'
    if not (feature_dir / 'manifest.json').exists():
        subprocess.run([sys.executable, str(ROOT / 'fitting/jaxENT/featurise_ISO_TRI_BI.py'),
                        '--kint-provider', 'jaxent', '--output-dir', str(feature_dir)], check=True)
    feature_manifest = json.loads((feature_dir / 'manifest.json').read_text())
    assert feature_manifest['kint_provider'] == 'jaxent' and feature_manifest['kint_unit'] == 'min^-1'
    for item in [feature_manifest['topology'], *feature_manifest['trajectories'].values()]:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Candidate trajectory provenance mismatch')
    source_keys = layout(source / 'topology_open_closed.json')
    ref = np.load(source / 'features_open_closed.npz')
    ref_logpf = .35 * ref['heavy_contacts'].astype(float) + 2 * ref['acceptor_contacts'].astype(float)
    kints = ref['k_ints'].astype(float)
    rates = kints[:, None] * np.exp(-ref_logpf)
    labels = np.repeat([0, 1], [provenance['open_frames'], provenance['closed_frames']])
    weights = np.where(labels == 0, .4 / provenance['open_frames'], .6 / provenance['closed_frames'])
    target = predict(rates, weights, labels, 'frame_uptake')
    if not np.isfinite(target).all() or np.var(target) <= 1e-8:
        raise ValueError('Degenerate target')
    scale = float(np.var(target) + 1e-8)
    np.savez_compressed(output / 'target.npz', target=target, times=TIMES, kints=kints,
                        reference_weights=weights, reference_groups=labels)
    diagnostics = [dict(ensemble='reference', prediction=m,
                       mse=float(np.mean((predict(rates, weights, labels, m)-target)**2)))
                   for m in ('frame_uptake', 'uptake', 'rate')]
    hashes = {str(source / f): digest(source / f) for f in
              ('features_open_closed.npz', 'topology_open_closed.json', 'manifest.json')}
    hashes.update(provenance['input_hashes'])
    import MDAnalysis as mda
    from MDAnalysis.analysis.rms import rmsd
    trajectory_root = Path(feature_manifest['topology']['path']).parent
    references = [mda.Universe(str(trajectory_root / f'TeaA_ref_{s}_state.pdb')).select_atoms('name CA').positions.copy()
                  for s in ('open', 'closed')]
    for ensemble in ('ISO_BI', 'ISO_TRI'):
        slug = ensemble.lower()
        assignment_path = ROOT / f'data/_clustering_results/cluster_assignments_{ensemble}.csv'
        assignments = pd.read_csv(assignment_path)
        feature_path = feature_dir / f'features_{slug}.npz'
        topology_path = feature_dir / f'topology_{slug}.json'
        index = alignment(source_keys, layout(topology_path))
        features = np.load(feature_path)
        logpf = (.35 * features['heavy_contacts'].astype(float) + 2 * features['acceptor_contacts'].astype(float))[index]
        np.testing.assert_allclose(features['k_ints'][index], kints, rtol=1e-12)
        groups = assignments.cluster_assignment.to_numpy(int)
        n = len(groups)
        if logpf.shape[1] != n or not np.array_equal(assignments.frame, np.arange(n)) or not set(groups) <= set(STATES):
            raise ValueError('Candidate frame/assignment mismatch')
        trajectory = feature_manifest['trajectories'][slug]
        universe = mda.Universe(feature_manifest['topology']['path'], trajectory['path'])
        if len(universe.trajectory) != n:
            raise ValueError('Trajectory length mismatch')
        ca = universe.select_atoms('name CA')
        measured = np.array([[rmsd(ca.positions, reference, center=True, superposition=True)
                              for reference in references] for _ in universe.trajectory])
        np.testing.assert_allclose(measured, assignments[['rmsd_open', 'rmsd_closed']], atol=1e-5, rtol=1e-5)
        reconstructed = measured.argmin(axis=1)
        reconstructed[measured.min(axis=1) > 1.] = -1
        np.testing.assert_array_equal(groups, reconstructed)
        candidate_rates = kints[:, None] * np.exp(-logpf)
        uniform = np.ones(n) / n
        truth_weights = np.zeros(n)
        for label, mass in zip(STATES, TRUTH):
            if mass and not np.any(groups == label):
                raise ValueError('Missing required candidate state')
            if np.any(groups == label):
                truth_weights[groups == label] = mass / np.sum(groups == label)
        for name, w in [('uniform', uniform), ('true_populations_uniform_within_state', truth_weights)]:
            for mode in ('frame_uptake', 'uptake', 'rate'):
                prediction = predict(candidate_rates, w, groups, mode)
                diagnostics.append(dict(ensemble=ensemble, prediction=mode, weights=name,
                    mse=float(np.mean((prediction-target)**2)), **metrics(w, groups)))
        folder = output / ensemble
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / 'input.npz', logpf=logpf, rates=candidate_rates,
            groups=groups, target=target, scale=scale, rmsd=measured, truth_weights=truth_weights)
        off = ~np.eye(n, dtype=bool)
        scalar = abs(logpf.mean(axis=0)[:, None] - logpf.mean(axis=0)[None, :])
        profile = squareform(pdist(logpf.T)) / np.sqrt(len(kints))
        graph_rows = []
        scalar_means = []
        for family, distances in [('scalar_logpf', scalar), ('profile_logpf', profile)]:
            pairs = distances[np.triu_indices(n, 1)]
            positive = pairs[pairs > 0]
            if not len(positive) or not np.isfinite(positive).all():
                raise ValueError('Degenerate graph distances')
            sigmas = np.quantile(positive, QUANTILES)
            for i, (q, sigma) in enumerate(zip(QUANTILES, sigmas)):
                kernel = np.exp(-np.minimum(.5 * (distances / sigma)**2, 80.))
                raw_mean = float(kernel[off].mean())
                factor = 1. if family == 'scalar_logpf' else scalar_means[i] / raw_mean
                kernel[off] *= factor
                if family == 'scalar_logpf':
                    scalar_means.append(raw_mean)
                np.testing.assert_allclose(kernel[off].mean(), scalar_means[i], rtol=1e-12)
                np.save(folder / f'{family}_{i}.npy', kernel)
                graph_rows.append(dict(family=family, quantile=float(q), sigma=float(sigma),
                    raw_coupling=raw_mean, factor=factor, coupling=float(kernel[off].mean()),
                    cross_state_fraction=float(kernel[groups[:,None] != groups[None,:]].sum()/kernel[off].sum())))
        write_json(folder / 'graphs.json', graph_rows)
        for path in [assignment_path, feature_path, topology_path, Path(trajectory['path']), *folder.iterdir()]:
            hashes[str(path.resolve())] = digest(path)
    pd.DataFrame(diagnostics).to_csv(output / 'preflight.csv', index=False)
    for path in [output / 'target.npz', output / 'preflight.csv', feature_dir / 'manifest.json']:
        hashes[str(path.resolve())] = digest(path)
    hashes[feature_manifest['topology']['path']] = feature_manifest['topology']['sha256']
    manifest = dict(code=identity(), hashes=hashes, target_mode='frame_uptake',
        target_populations=TRUTH.tolist(), states=list(STATES), specs=specs(),
        times_min=TIMES.tolist(), contact_mode='hard', scale=scale,
        information_fraction=float(((target > 1e-8) & (target < 1-1e-8)).mean()))
    write_json(output / 'manifest.json', manifest)
    return manifest


def worker_init():
    available = sorted(os.sched_getaffinity(0))
    slot = (multiprocessing.current_process()._identity or (1,))[0] - 1
    start = (2 * slot) % len(available)
    os.sched_setaffinity(0, {available[start], available[(start+1) % len(available)]})


def fit(data, spec, mode, kernel=None, checkpoints=(1000, 3000, 10000), window=250):
    import jax
    import jax.numpy as jnp
    import optax
    jax.config.update('jax_enable_x64', True)
    rates, groups, target = [jnp.asarray(data[k]) for k in ('rates', 'groups', 'target')]
    n = rates.shape[1]
    if kernel is not None:
        kernel = jnp.asarray(kernel)
    def objective(logits):
        w = jax.nn.softmax(logits)
        prediction = predict(rates, w, groups, mode, xp=jnp)
        loss = jnp.mean((prediction - target)**2) / float(data['scale'])
        if spec['family'] in FAMILIES[2:]:
            c = w - 1/n
            raw = n*n * (jnp.dot(w*c*c, kernel @ w) - jnp.dot(w*c, kernel @ (w*c)))
            loss = loss + spec['strength'] * jnp.maximum(raw, 0.)
        elif spec['family'] == 'maxent':
            loss = loss + spec['strength'] * jnp.mean(-jnp.log(n) - jax.nn.log_softmax(logits))
        return loss
    value_grad = jax.jit(jax.value_and_grad(objective))
    optimiser = optax.adam(.05)
    @jax.jit
    def advance(x, state, steps):
        def step(_, carry):
            x, state = carry
            _, gradient = value_grad(x)
            updates, state = optimiser.update(gradient, state, x)
            return optax.apply_updates(x, updates), state
        return jax.lax.fori_loop(0, steps, step, (x, state))
    results = []
    for initial in [np.zeros(n), np.random.default_rng(SEED).normal(0, .01, n)]:
        x = jnp.asarray(initial)
        state = optimiser.init(x)
        previous = 0
        for checkpoint in checkpoints:
            x, state = advance(x, state, checkpoint-previous-window)
            before = float(objective(x))
            x, state = advance(x, state, window)
            after, grad = value_grad(x)
            after = float(after)
            relative = abs(after-before)/max(abs(after), 1e-12)
            previous = checkpoint
            if relative <= .01:
                break
        w = np.asarray(jax.nn.softmax(x))
        results.append(dict(weights=w, objective=after, relative=relative, steps=checkpoint,
                            grad_norm=float(jnp.linalg.norm(grad))))
    objective_values = np.array([r['objective'] for r in results])
    best = int(np.argmin(objective_values))
    gap = float(abs(objective_values[0]-objective_values[1])/max(abs(objective_values[best]), 1e-12))
    finite = all(np.isfinite(r['weights']).all() and np.isfinite([r['objective'], r['grad_norm']]).all() for r in results)
    if not finite:
        raise FloatingPointError('Nonfinite fit')
    return dict(weights=results[best]['weights'], objective=objective_values[best],
        prediction=np.asarray(predict(rates, jnp.asarray(results[best]['weights']), groups, mode, xp=jnp)),
        initial_weights=np.stack([r['weights'] for r in results]),
        initial_objectives=objective_values, initial_relative=np.array([r['relative'] for r in results]),
        initial_steps=np.array([r['steps'] for r in results]),
        initial_grad_norm=np.array([r['grad_norm'] for r in results]), objective_gap=gap,
        chosen_start=best, converged=bool(gap <= .01 and all(r['relative'] <= .01 for r in results)))


def run_arm(task):
    output, stage, ensemble, spec = task
    output = Path(output)
    folder = output / stage / ensemble / f"arm_{spec['arm']:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = folder / 'complete.json'
        if marker.exists():
            complete = json.loads(marker.read_text())
            if digest(folder / 'fit.npz') != complete['sha256']:
                raise ValueError('Corrupt saved fit')
            return complete['row']
        data = dict(np.load(output / ensemble / 'input.npz'))
        kernel = None
        if spec['family'] in FAMILIES[2:]:
            i = int(np.flatnonzero(QUANTILES == spec['quantile'])[0])
            kernel = np.load(output / ensemble / f"{spec['family']}_{i}.npy", mmap_mode='r')
        try:
            result = fit(data, spec, stage, kernel)
            np.testing.assert_allclose(result['weights'].sum(), 1., atol=1e-12)
            row = dict(stage=stage, ensemble=ensemble, **spec,
                converged=bool(result['converged']), objective=float(result['objective']),
                mse=float(np.mean((result['prediction']-data['target'])**2)),
                objective_gap=float(result['objective_gap']),
                steps=int(result['initial_steps'].max()), **metrics(result['weights'], data['groups']))
            row['initial_population_tv'] = float(abs(populations(result['initial_weights'][0], data['groups'])-
                populations(result['initial_weights'][1], data['groups'])).sum()/2)
            with (folder / 'fit.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, **result)
            (folder / 'fit.npz.tmp').replace(folder / 'fit.npz')
            write_json(marker, dict(sha256=digest(folder / 'fit.npz'), row=row))
            return row
        except (FloatingPointError, ValueError, AssertionError) as error:
            row = dict(stage=stage, ensemble=ensemble, **spec, converged=False,
                       mse=None, recovery=None, error=str(error))
            write_json(folder / 'failure.json', row)
            return row


def collect(output, stage=None):
    rows = []
    stages = (stage,) if stage else ('uptake', 'rate')
    for mode in stages:
        for path in sorted((output / mode).glob('*/arm_*/complete.json')):
            item = json.loads(path.read_text())
            if digest(path.with_name('fit.npz')) != item['sha256']:
                raise ValueError('Fit checksum mismatch')
            rows.append(item['row'])
        for path in sorted((output / mode).glob('*/arm_*/failure.json')):
            if not path.with_name('complete.json').exists():
                rows.append(json.loads(path.read_text()))
    return rows


def run_stage(output, stage, workers):
    tasks = [(str(output), stage, ensemble, spec)
             for ensemble in ('ISO_BI', 'ISO_TRI') for spec in specs()]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as pool:
        for future in as_completed([pool.submit(run_arm, task) for task in tasks]):
            row = future.result()
            print(stage, row['ensemble'], row['family'], row['arm'],
                  'converged', row['converged'], 'recovery', row.get('recovery'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'report', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.workers < 1 or args.workers > 10:
        parser.error('--workers must be 1..10')
    if args.phase in ('prepare', 'all'):
        prepare(args.output)
    else:
        load_manifest(args.output)
    if args.smoke:
        worker_init()
        data = dict(np.load(args.output / 'ISO_BI/input.npz'))
        result = fit(data, specs()[0], 'uptake', checkpoints=(20,), window=10)
        write_json(args.output / 'smoke.json', dict(finite=bool(np.isfinite(result['objective'])), steps=20))
        return
    if args.phase in ('run', 'all'):
        run_stage(args.output, 'uptake', args.workers)
        decision = gate(collect(args.output, 'uptake'))
        write_json(args.output / 'gate.json', decision)
        if decision['passed']:
            run_stage(args.output, 'rate', args.workers)
        write_json(args.output / 'run_status.json', dict(status='complete', mean_rate_run=decision['passed']))
    if args.phase in ('report', 'all'):
        report = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_control_report')
        report.build(args.output)


if __name__ == '__main__':
    main()
