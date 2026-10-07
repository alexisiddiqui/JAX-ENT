"""ISO_TRI average-linkage merge graphs with scalar-logPF edge weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil

s = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_strength')
h, e = s.h, s.e
a = importlib.import_module('jaxent.examples.ATLAS_BV.analysis.graph_representation_audit')
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

OUTPUT = e.OUTPUT.with_name('omc_hierarchical_control')
DISTANCES = ('rmsd', 'w1', 'logpf_profile')


def merge_graph(distance):
    d = np.asarray(distance, float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Finite nonnegative square distance matrix required')
    np.testing.assert_allclose(d, d.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(d), 0., atol=1e-12)
    n = len(d)
    tree = linkage(squareform(d, checks=False), method='average', optimal_ordering=False)
    members = {i: np.array([i]) for i in range(n)}
    edges = []
    mask = np.zeros((n, n), bool)
    for merge, (left, right, height, _) in enumerate(tree):
        x, y = members.pop(int(left)), members.pop(int(right))
        cross = d[np.ix_(x, y)]
        ii, jj = np.where(cross == cross.min())
        choices = np.sort(np.column_stack((x[ii], y[jj])), axis=1)
        order = np.lexsort((choices[:, 1], choices[:, 0]))
        u, v = choices[order[0]]
        mask[u, v] = mask[v, u] = True
        edges.append((merge, int(u), int(v), float(d[u, v]), float(height)))
        members[n + merge] = np.sort(np.concatenate((x, y)))
    if mask.sum() != 2*(n-1) or connected_components(mask, directed=False)[0] != 1:
        raise AssertionError('Merge graph must be a spanning tree')
    return tree, mask, pd.DataFrame(edges, columns=['merge', 'left', 'right', 'distance', 'height'])


def specs():
    return [dict(arm=25+6*j+i, family='hierarchical_'+name, distance=name,
                 quantile=float(q), strength=.1, ensemble='ISO_TRI', stage='uptake')
            for j, name in enumerate(DISTANCES) for i, q in enumerate(e.QUANTILES)]


def identity():
    return dict(runner=e.digest(__file__), fitter=s.identity(), geometry=a.code_identity())


def validate(output):
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['code'] != identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path, expected in manifest['hashes'].items():
        if e.digest(path) != expected:
            raise ValueError(f'Changed source or artifact: {path}')
    return manifest


def prepare(output):
    if (output/'manifest.json').exists():
        return validate(output)
    h.validate(h.OUTPUT)
    controls = [row for row in h.collect(h.OUTPUT) if row['ensemble'] == 'ISO_TRI']
    if len(controls) != 25 or any('error' in row for row in controls):
        raise ValueError('All 25 grouped ISO_TRI controls required')
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(source.resolve())] = e.digest(source)
        hashes[str(destination.resolve())] = e.digest(destination)
    copy(h.OUTPUT/'ISO_TRI/input.npz', output/'input.npz')
    copy(h.OUTPUT/'source/topology.json', output/'topology.json')
    copy(h.OUTPUT/'manifest.json', output/'source_manifest.json')
    copy(e.OUTPUT/'features/manifest.json', output/'feature_manifest.json')
    for row in controls:
        source = h.archive(h.OUTPUT, row)
        for name in ('fit.npz', 'complete.json'):
            copy(source.with_name(name), output/'fits'/f"arm_{row['arm']:02d}"/name)
    for i in range(6):
        for family in ('scalar_logpf', 'profile_logpf', h.FAMILY):
            copy(h.OUTPUT/'ISO_TRI'/f'{family}_{i}.npy', output/f'{family}_{i}.npy')
    provenance = json.loads((output/'feature_manifest.json').read_text())
    import MDAnalysis as mda
    from MDAnalysis.analysis.rms import rmsd
    for item in (provenance['topology'], provenance['trajectories']['iso_tri']):
        if e.digest(item['path']) != item['sha256']:
            raise ValueError('Candidate trajectory provenance mismatch')
        hashes[item['path']] = item['sha256']
    universe = mda.Universe(provenance['topology']['path'], provenance['trajectories']['iso_tri']['path'])
    ca = universe.select_atoms('protein and name CA')
    xyz = np.array([ca.positions.copy() for _ in universe.trajectory], dtype=float)
    data = dict(np.load(output/'input.npz'))
    if len(xyz) != data['logpf'].shape[1]:
        raise ValueError('Frame count mismatch')
    root = Path(provenance['topology']['path']).parent
    references = []
    for state in ('open', 'closed'):
        path = root/f'TeaA_ref_{state}_state.pdb'
        hashes[str(path)] = e.digest(path)
        references.append(mda.Universe(str(path)).select_atoms('protein and name CA').positions.copy())
    measured = np.array([[rmsd(x, ref, center=True, superposition=True) for ref in references] for x in xyz])
    np.testing.assert_allclose(measured, data['rmsd'], atol=1e-5, rtol=1e-5)
    np.savez_compressed(output/'coordinates.npz', ca=xyz, frames=np.arange(len(xyz)))
    print('Frame alignment verified; building RMSD and W1 distances', flush=True)
    samples = np.array([np.sort(pdist(x)) for x in xyz], dtype=np.float32)
    signatures = np.quantile(samples, np.linspace(0, 1, 256), axis=1).T
    distances = dict(rmsd=a.rmsd_matrix(xyz), w1=cdist(signatures, signatures, metric='cityblock')/256,
                     logpf_profile=a.profile_distances(data['logpf'])['logpf_profile'])
    nn = a.nearest(distances['w1'])
    rng = np.random.default_rng(e.SEED)
    pairs = rng.integers(0, len(xyz), size=(1024, 2))
    pairs[512:, 1] = nn[pairs[512:, 0], rng.integers(0, 20, 512)]
    audit_rows = []
    for index, (u, v) in enumerate(pairs):
        exact = float(np.mean(abs(samples[u].astype(float)-samples[v])))
        audit_rows.append(dict(left=u, right=v, exact=exact, approximate=distances['w1'][u,v], sample='random' if index < 512 else 'neighbour'))
    pd.DataFrame(audit_rows).to_csv(output/'w1_pairs.csv', index=False)
    audit_nn = []
    for anchor in rng.choice(len(xyz), 16, replace=False):
        exact = np.empty(len(xyz))
        for start in range(0, len(xyz), 32):
            exact[start:start+32] = np.mean(abs(samples[start:start+32].astype(float)-samples[anchor]), axis=1)
        exact[anchor] = np.inf
        best = np.argsort(exact, kind='stable')[:20]
        audit_nn.append(dict(anchor=int(anchor), overlap=len(np.intersect1d(best, nn[anchor]))/20,
                             distance_ratio=float(exact[nn[anchor]].mean()/max(exact[best].mean(), 1e-12))))
    pd.DataFrame(audit_nn).to_csv(output/'w1_neighbours.csv', index=False)
    del samples
    metadata = json.loads((h.OUTPUT/'ISO_TRI/original_graphs.json').read_text())
    masks, graph_rows = {}, []
    for name, distance in distances.items():
        tree, mask, edges = merge_graph(distance)
        masks[name] = mask
        np.savez_compressed(output/f'graph_{name}.npz', distance=distance, linkage=tree, mask=mask, degree=mask.sum(axis=1))
        edges.to_csv(output/f'edges_{name}.csv', index=False)
        for i, q in enumerate(e.QUANTILES):
            scalar = np.load(output/f'scalar_logpf_{i}.npy')
            kernel, factor = h.match_kernel(mask, scalar)
            np.save(output/f'hierarchical_{name}_{i}.npy', kernel)
            total = float(kernel[mask].sum())
            weights = kernel[np.triu(mask)]
            state = data['groups']
            row = dict(distance=name, quantile=float(q), sigma=next(r['sigma'] for r in metadata if r['family']=='scalar_logpf' and r['quantile']==q),
                       factor=factor, coupling=total/(len(mask)*(len(mask)-1)), edges=int(mask.sum()//2),
                       degree_min=int(mask.sum(axis=1).min()), degree_median=float(np.median(mask.sum(axis=1))), degree_max=int(mask.sum(axis=1).max()),
                       max_edge_fraction=float(weights.max()/weights.sum()), effective_edges=float(weights.sum()**2/np.sum(weights**2)),
                       cross_state_coupling_fraction=float(kernel[mask & (state[:,None]!=state[None,:])].sum()/total))
            for x, label_x in zip(e.STATES, ('open','closed','intermediate')):
                for y, label_y in zip(e.STATES, ('open','closed','intermediate')):
                    row[f'{label_x}_{label_y}_coupling'] = float(kernel[mask & (state[:,None]==x) & (state[None,:]==y)].sum()/total)
            graph_rows.append(row)
    pd.DataFrame(graph_rows).to_csv(output/'graphs.csv', index=False)
    pd.DataFrame([dict(first=x, second=y, edge_jaccard=float((masks[x]&masks[y]).sum()/(masks[x]|masks[y]).sum()))
                  for i,x in enumerate(DISTANCES) for y in DISTANCES[i+1:]]).to_csv(output/'graph_overlap.csv', index=False)
    for path in output.iterdir():
        if path.is_file() and path.name != 'manifest.json':
            hashes[str(path.resolve())] = e.digest(path)
    manifest = dict(code=identity(), hashes=hashes, specs=specs(), new_arms=18, reused_arms=25,
                    contact_mode='hard', strength=.1, linkage='average', links='closest cross-child pair per merge',
                    tie_break='lexicographic original frame pair; scipy linkage in original frame order',
                    w1_pair_spearman=float(spearmanr([r['exact'] for r in audit_rows], [r['approximate'] for r in audit_rows]).statistic),
                    w1_neighbour_overlap=float(np.mean([r['overlap'] for r in audit_nn])))
    e.write_json(output/'manifest.json', manifest)
    return manifest


def kernel_path(output, row):
    index = int(np.flatnonzero(e.QUANTILES == row['quantile'])[0])
    return output/f"{row['family']}_{index}.npy"


def run_arm(task):
    output, spec = task
    output = Path(output)
    folder = output/'fits'/f"arm_{spec['arm']:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'complete.json').exists():
            saved = json.loads((folder/'complete.json').read_text())
            if e.digest(folder/'fit.npz') != saved['sha256']:
                raise ValueError('Corrupt saved fit')
            return saved['row']
        row = dict(spec)
        try:
            data = dict(np.load(output/'input.npz'))
            kernel = np.load(kernel_path(output, row), mmap_mode='r')
            result = s.fit_at_strength(data, kernel, .1)
            np.testing.assert_allclose(result['initial_weights'].sum(axis=1), 1., atol=1e-12)
            row.update(converged=bool(result['converged']), mse=float(np.mean((result['prediction']-data['target'])**2)),
                       objective=float(result['objective']), objective_gap=float(result['objective_gap']),
                       steps=int(result['initial_steps'].max()), **e.metrics(result['weights'], data['groups']))
            with (folder/'fit.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, **result)
            (folder/'fit.npz.tmp').replace(folder/'fit.npz')
            e.write_json(folder/'complete.json', dict(sha256=e.digest(folder/'fit.npz'), row=row))
        except (FloatingPointError, ValueError, AssertionError) as error:
            row.update(converged=False, mse=None, recovery=None, error=str(error))
            e.write_json(folder/'failure.json', row)
        return row


def select(rows):
    table = pd.DataFrame(rows)
    valid = table.loc[table.converged.astype(bool) & np.isfinite(table.mse)]
    return valid.sort_values(['mse', 'arm'], kind='stable').groupby('family', sort=False).head(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare','run','report','all'), default='all')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error('--workers must be 1..10')
    output = args.output.resolve()
    if args.phase in ('prepare','all'):
        prepare(output)
    else:
        validate(output)
    if args.phase in ('run','all'):
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=e.worker_init) as pool:
            for future in as_completed([pool.submit(run_arm, (str(output), spec)) for spec in specs()]):
                print(json.dumps(future.result()), flush=True)
        rows = [r for r in s.collect(output) if r['arm'] >= 25]
        e.write_json(output/'run_status.json', dict(status='complete' if len(rows)==18 else 'incomplete', new_fits=len(rows),
                     converged=sum(r['converged'] for r in rows), numerical_failures=sum('error' in r for r in rows)))
    if args.phase in ('report','all'):
        importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_report').build(output)


if __name__ == '__main__':
    main()
