import numpy as np
import pytest
from MDAnalysis.analysis.rms import rmsd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform

from jaxent.examples.ATLAS_BV.analysis import graph_representation_audit as a


def test_rmsd_matches_independent_pairwise_alignment():
    rng=np.random.default_rng(17)
    coordinates=rng.normal(size=(6,12,3))
    matrix=a.rmsd_matrix(coordinates,block=2)
    for i in range(6):
        for j in range(6):
            expected=rmsd(coordinates[i],coordinates[j],center=True,superposition=True)
            np.testing.assert_allclose(matrix[i,j],expected,atol=1e-7)


def test_rmsd_rigid_invariance_and_reflection():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(20,3))
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    transformed=x@rotation+np.array([10.,20.,30.])
    reflected=x*np.array([-1.,1.,1.])
    matrix=a.rmsd_matrix(np.stack([x,transformed,reflected]))
    assert matrix[0,1]<1e-6
    assert matrix[0,2]>.1


def test_cross_replica_neighbours_exclude_same_replica():
    distance=squareform(pdist(np.arange(6)[:,None]))
    replicas=np.array([1,1,2,2,3,3])
    neighbours=a.nearest(distance,k=2,replicas=replicas)
    assert np.all(replicas[neighbours]!=replicas[:,None])
    np.testing.assert_allclose(a.overlap(neighbours,neighbours),1.)
    assert not np.diag(a.adjacency(neighbours)).any()


def test_equal_mean_profiles_are_distinguished():
    z=np.array([[2.,3.,18.],[10.,10.,10.],[18.,17.,2.]])
    distances=a.profile_distances(z)
    np.testing.assert_array_equal(distances['logpf_mean'],0.)
    assert distances['logpf_profile'][0,1]<distances['logpf_profile'][0,2]


def test_exact_w1_sorted_distance_identity():
    from scipy.stats import wasserstein_distance
    rng=np.random.default_rng(9)
    x,y=rng.normal(size=(2,40))
    np.testing.assert_allclose(np.mean(abs(np.sort(x)-np.sort(y))),wasserstein_distance(x,y))


def test_fixed_cluster_counts_and_tie_breaks():
    d=squareform(pdist(np.arange(30)[:,None]))
    labels=cut_tree(linkage(squareform(d),method='average'),n_clusters=[2,3,5,10,20])
    assert [len(np.unique(labels[:,i])) for i in range(5)]==[2,3,5,10,20]
    with pytest.raises(ValueError):
        a.nearest(d,k=30)
