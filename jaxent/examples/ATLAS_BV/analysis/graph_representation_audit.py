"""Compare structural and Bradshaw-contact logPF geometry on matching ATLAS frames."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('MPLCONFIGDIR','/tmp/atlas-graph-audit-matplotlib')

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .common import HERE, load_systems, replica_paths

OUTPUT=HERE/'outputs/analysis/pairwise_geometry/graph_representation_audit_bradshaw_0p1'
METRICS=('rmsd','w1','logpf_profile','logpf_mean')
COUNTS=(2,3,5,10,20)
SEED=20260910


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def code_identity():
    repo=HERE.parents[2]
    paths=[Path(__file__),repo/'jaxent/src/models/func/contacts.py',repo/'jaxent/src/models/config.py',
           repo/'jaxent/src/models/HDX/BV/forwardmodel.py',repo/'jaxent/src/featurise.py']
    return {str(p):digest(p) for p in paths}


def rmsd_matrix(coordinates,block=32):
    """Exact pairwise optimal-rotation RMSD, not common-reference RMSD."""
    x=np.asarray(coordinates,float)
    if x.ndim!=3 or x.shape[2]!=3 or not np.isfinite(x).all():
        raise ValueError('Finite frame-by-atom-by-3 coordinates required')
    x=x-x.mean(axis=1,keepdims=True)
    norms=np.sum(x*x,axis=(1,2))
    result=np.empty((len(x),len(x)))
    for start in range(0,len(x),block):
        end=min(start+block,len(x))
        covariance=np.einsum('fai,gaj->fgij',x[start:end],x,optimize=True)
        singular=np.linalg.svd(covariance,compute_uv=False)
        trace=singular.sum(axis=-1)-2*singular[:,:,-1]*(np.linalg.det(covariance)<0)
        result[start:end]=np.sqrt(np.maximum((norms[start:end,None]+norms[None,:]-2*trace)/x.shape[1],0.))
    result=(result+result.T)/2
    np.fill_diagonal(result,0.)
    return result


def nearest(distance,k=20,replicas=None):
    d=np.asarray(distance,float).copy()
    np.fill_diagonal(d,np.inf)
    if replicas is not None:
        d[np.asarray(replicas)[:,None]==np.asarray(replicas)[None,:]]=np.inf
    if np.min(np.isfinite(d).sum(axis=1))<k:
        raise ValueError('Insufficient eligible neighbours')
    return np.argsort(d,axis=1,kind='stable')[:,:k]


def adjacency(neighbours):
    mask=np.zeros((len(neighbours),len(neighbours)),bool)
    mask[np.arange(len(mask))[:,None],neighbours]=True
    return mask | mask.T


def overlap(left,right):
    return np.array([len(np.intersect1d(a,b))/len(a) for a,b in zip(left,right)])


def profile_distances(logpf):
    z=np.asarray(logpf,float)
    return dict(logpf_profile=squareform(pdist(z.T))/np.sqrt(z.shape[0]),
                logpf_mean=abs(z.mean(axis=0)[:,None]-z.mean(axis=0)[None,:]))


def input_identity(row):
    inputs=[HERE/row['pdb_path'],*replica_paths(row)]
    return dict(code=code_identity(),inputs={str(p):digest(p) for p in inputs},
        protocol=dict(contact_mode='bradshaw_switch',switch_scale_nc_angstrom=.1,switch_scale_nh_angstrom=.1,
            heavy_radius=6.5,oxygen_radius=2.4,residue_ignore=[-2,2],
            environment='all',bc=.35,bh=2.,temperature=300.,pd=7.,kint_unit='min^-1'),
        frames_per_replica=256,equilibration_ns=10.,frame_interval_ns=.1,
        metrics=list(METRICS),w1_quantiles=256,neighbours=20,cluster_counts=list(COUNTS),
        linkage='average',seed=SEED)


def validate(folder):
    marker=json.loads((folder/'complete.json').read_text())
    if marker['identity']['code']!=code_identity():
        raise ValueError('Scientific code changed; use a fresh output directory')
    for path,expected in marker['identity']['inputs'].items():
        if digest(path)!=expected:
            raise ValueError(f'Source changed: {path}')
    for name,expected in marker['artifacts'].items():
        if digest(folder/name)!=expected:
            raise ValueError(f'Corrupt artifact: {folder/name}')
    return marker


def sample_and_featurise(row,folder):
    import MDAnalysis as mda
    from jaxent.src.models.config import BV_model_Config
    from jaxent.src.models.HDX.BV.forwardmodel import BV_model
    from jaxent.src.custom_types.config import FeaturiserSettings
    from jaxent.src.interfaces.builder import Experiment_Builder
    from jaxent.src.featurise import run_featurise
    from jaxent.src.interfaces.topology import PTSerialiser
    pdb=HERE/row['pdb_path']
    coordinates,boxes,frames,replicas=[],[],[],[]
    for replica,path in enumerate(replica_paths(row),1):
        universe=mda.Universe(str(pdb),str(path))
        keep=np.flatnonzero(np.arange(len(universe.trajectory))*.1>10.)
        take=keep[np.linspace(0,len(keep)-1,min(256,len(keep)),dtype=int)]
        for frame in take:
            universe.trajectory[int(frame)]
            coordinates.append(universe.atoms.positions.copy())
            boxes.append(universe.dimensions.copy())
            frames.append(int(frame))
            replicas.append(replica)
    universe=mda.Universe(str(pdb))
    universe.load_new(np.asarray(coordinates),dimensions=np.asarray(boxes))
    ca=universe.select_atoms('protein and name CA')
    if len(ca)!=int(row['length']):
        raise ValueError('C-alpha count differs from ATLAS catalog')
    xyz=np.asarray(coordinates)[:,ca.indices].astype(float)
    del coordinates
    config=BV_model_Config(contact_mode='bradshaw_switch',switch_scale_nc=.1,switch_scale_nh=.1,kint_unit='min^-1')
    config.temperature,config.ph=300.,7.
    builder=Experiment_Builder(universes=[universe],forward_models=[BV_model(config)])
    features,topologies=run_featurise(builder,FeaturiserSettings(name='atlas_graph_bradshaw_0p1',batch_size=None))
    features=features[0]
    features.save(str(folder/'features.npz'))
    PTSerialiser.save_list_to_json(topologies[0],str(folder/'topology.json'))
    logpf=.35*np.asarray(features.heavy_contacts,dtype=float)+2*np.asarray(features.acceptor_contacts,dtype=float)
    if logpf.shape[1]!=len(frames) or not np.isfinite(logpf).all():
        raise ValueError('Invalid frame-feature alignment')
    np.savez_compressed(folder/'frames.npz',ca=xyz,frames=frames,replicas=replicas,logpf=logpf)
    return xyz,np.asarray(replicas),logpf


def analyse_system(row,folder):
    xyz,replicas,logpf=sample_and_featurise(row,folder)
    n=len(xyz)
    rng=np.random.default_rng(SEED)
    distributions=np.array([np.sort(pdist(frame)) for frame in xyz],dtype=np.float32)
    # Same 256-point inverse-CDF W1 approximation as existing ATLAS graph studies.
    signatures=np.quantile(distributions,np.linspace(0,1,256),axis=1).T
    distances=dict(rmsd=rmsd_matrix(xyz),w1=cdist(signatures,signatures,metric='cityblock')/256,
                   **profile_distances(logpf))
    for value in distances.values():
        np.fill_diagonal(value,0.)
        if not np.isfinite(value).all() or np.any(value<0):
            raise ValueError('Invalid distance matrix')
    nn={name:nearest(value) for name,value in distances.items()}
    cross_nn={name:nearest(value,replicas=replicas) for name,value in distances.items()}
    masks={name:adjacency(value) for name,value in nn.items()}
    left=rng.integers(n,size=1024)
    right=rng.integers(n-1,size=1024)
    right+=right>=left
    right[512:]=nn['w1'][left[512:],rng.integers(20,size=512)]
    exact=np.empty(len(left))
    for start in range(0,len(left),32):
        stop=start+32
        exact[start:stop]=np.mean(abs(distributions[left[start:stop]].astype(float)-distributions[right[start:stop]]),axis=1)
    approx=distances['w1'][left,right]
    pd.DataFrame(dict(left=left,right=right,exact_w1=exact,approximate_w1=approx,
                      sample=['random']*512+['w1_neighbour']*512)).to_csv(folder/'w1_audit.csv',index=False)
    neighbour_audit=[]
    for anchor in rng.choice(n,16,replace=False):
        full=np.empty(n)
        for start in range(0,n,32):
            full[start:start+32]=np.mean(abs(distributions[start:start+32].astype(float)-distributions[anchor]),axis=1)
        full[anchor]=np.inf
        exact_neighbours=np.argsort(full,kind='stable')[:20]
        neighbour_audit.append(dict(anchor=int(anchor),replica=int(replicas[anchor]),
            overlap=float(len(np.intersect1d(exact_neighbours,nn['w1'][anchor]))/20),
            distance_ratio=float(full[nn['w1'][anchor]].mean()/max(full[exact_neighbours].mean(),1e-12))))
    pd.DataFrame(neighbour_audit).to_csv(folder/'w1_neighbour_audit.csv',index=False)
    del distributions
    graph_rows=[]
    for name in METRICS:
        count,component=connected_components(masks[name],directed=False)
        np.savez_compressed(folder/f'graph_{name}.npz',distance=distances[name],neighbours=nn[name],
                            cross_replica_neighbours=cross_nn[name],mask=masks[name],components=component)
        graph_rows.append(dict(metric=name,components=count,largest_component_fraction=float(np.bincount(component).max()/n),
            degree_min=int(masks[name].sum(axis=1).min()),degree_median=float(np.median(masks[name].sum(axis=1))),
            degree_max=int(masks[name].sum(axis=1).max()),edge_fraction=float(masks[name].sum()/(n*(n-1))),
            same_replica_neighbour_fraction=float(np.mean(replicas[nn[name]]==replicas[:,None]))))
    pairs=[]
    indices=np.triu_indices(n,1)
    ranks=np.stack([rankdata(distances[name][indices]) for name in METRICS])
    correlations=np.corrcoef(ranks)
    for i,a in enumerate(METRICS):
        for j in range(i+1,len(METRICS)):
            b=METRICS[j]
            pairs.append(dict(first=a,second=b,spearman=float(correlations[i,j]),
                neighbour_overlap=float(overlap(nn[a],nn[b]).mean()),
                cross_replica_overlap=float(overlap(cross_nn[a],cross_nn[b]).mean()),
                edge_jaccard=float(np.sum(masks[a]&masks[b])/np.sum(masks[a]|masks[b])),
                random_neighbour_overlap=20/(n-1),random_cross_replica_overlap=float(np.mean([20/np.sum(replicas!=rep) for rep in replicas]))))
    edge_rows=[]
    for structural in ('rmsd','w1'):
        d=distances[structural]
        ranked=d.copy()
        np.fill_diagonal(ranked,np.inf)
        percentile=rankdata(ranked,axis=1)/n
        best=d[np.arange(n)[:,None],nn[structural]]
        for name in METRICS:
            observed=d[np.arange(n)[:,None],nn[name]]
            edge_rows.append(dict(metric=name,structural=structural,
                median_edge_distance=float(np.median(observed)),
                ratio_to_structural_neighbours=float(np.mean(observed)/max(np.mean(best),1e-12)),
                median_structural_percentile=float(np.median(percentile[np.arange(n)[:,None],nn[name]])),
                distant_edge_fraction=float(np.mean(percentile[np.arange(n)[:,None],nn[name]]>.95))))
    partitions={}
    cluster_rows=[]
    for name in METRICS:
        tree=linkage(squareform(distances[name],checks=False),method='average')
        labels=cut_tree(tree,n_clusters=list(COUNTS))
        np.savez_compressed(folder/f'clusters_{name}.npz',linkage=tree,labels=labels,counts=COUNTS)
        for i,k in enumerate(COUNTS):
            partitions[name,k]=labels[:,i]
            cluster_rows.append(dict(metric=name,k=k,min_cluster_size=int(np.bincount(labels[:,i]).min()),
                max_cluster_fraction=float(np.bincount(labels[:,i]).max()/n),
                silhouette=float(silhouette_score(distances[name],labels[:,i],metric='precomputed')),
                rmsd_silhouette=float(silhouette_score(distances['rmsd'],labels[:,i],metric='precomputed')),
                w1_silhouette=float(silhouette_score(distances['w1'],labels[:,i],metric='precomputed')),
                replica_nmi=float(normalized_mutual_info_score(replicas,labels[:,i]))))
    agreement=[]
    for k in COUNTS:
        for i,a in enumerate(METRICS):
            for b in METRICS[i+1:]:
                agreement.append(dict(first=a,second=b,k=k,
                    ari=float(adjusted_rand_score(partitions[a,k],partitions[b,k])),
                    nmi=float(normalized_mutual_info_score(partitions[a,k],partitions[b,k]))))
    for name,records in [('graphs',graph_rows),('pairs',pairs),('edge_geometry',edge_rows),
                         ('clusters',cluster_rows),('cluster_agreement',agreement)]:
        pd.DataFrame(records).assign(system_id=row['system_id']).to_csv(folder/f'{name}.csv',index=False)
    return dict(system_id=row['system_id'],frames=n,residues=int(row['length']),logpf_residues=len(logpf),
        w1_audit_spearman=float(spearmanr(exact,approx).statistic),
        w1_median_absolute_error=float(np.median(abs(exact-approx))),
        w1_neighbour_median_relative_error=float(np.median(abs(exact[512:]-approx[512:])/np.maximum(exact[512:],1e-12))),
        w1_exact_neighbour_overlap=float(np.mean([item['overlap'] for item in neighbour_audit])),
        max_adjacent_ca_distance=float(np.linalg.norm(np.diff(xyz,axis=1),axis=2).max()))


def worker_init():
    available=sorted(os.sched_getaffinity(0))
    slot=(multiprocessing.current_process()._identity or (1,))[0]-1
    os.sched_setaffinity(0,{available[(2*slot)%len(available)],available[(2*slot+1)%len(available)]})


def run_system(task):
    row,output=task
    folder=Path(output)/'systems'/row['system_id']
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'complete.json').exists():
            return validate(folder)['summary']
        identity=input_identity(row)
        try:
            with (folder/'run.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                summary=analyse_system(row,folder)
            artifacts={path.name:digest(path) for path in folder.iterdir() if path.suffix in ('.npz','.json','.csv') and path.name not in ('complete.json','failure.json')}
            write_json(folder/'complete.json',dict(identity=identity,summary=summary,artifacts=artifacts))
            return summary
        except Exception as error:
            write_json(folder/'failure.json',dict(system_id=row['system_id'],error=repr(error)))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('run','report','all'),default='all')
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    output=args.output.resolve()
    if not 1<=args.workers<=10:
        parser.error('--workers must be 1..10')
    systems=load_systems()
    if args.limit:
        systems=systems[:args.limit]
    output.mkdir(parents=True,exist_ok=True)
    if args.phase in ('run','all'):
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
            for future in as_completed([pool.submit(run_system,(row,str(output))) for row in systems]):
                print(json.dumps(future.result()),flush=True)
    if args.phase in ('report','all'):
        from . import graph_representation_report
        graph_representation_report.build(output)


if __name__=='__main__':
    main()
