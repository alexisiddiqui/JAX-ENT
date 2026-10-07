import importlib

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

t = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_control')
r = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')


def test_every_merge_uses_closest_cross_child_pair():
    d = squareform(pdist(np.random.default_rng(12).normal(size=(25, 3))))
    tree, mask, edges = t.merge_graph(d)
    members = {i: [i] for i in range(len(d))}
    for i, merge in enumerate(tree):
        x, y = members[int(merge[0])], members[int(merge[1])]
        row = edges.iloc[i]
        assert (row.left in x and row.right in y) or (row.left in y and row.right in x)
        assert row.distance == d[np.ix_(x, y)].min()
        members[len(d)+i] = x+y
    assert mask.sum() == 2*(len(d)-1)
    assert connected_components(mask, directed=False)[0] == 1
    assert not mask.diagonal().any()
    np.testing.assert_array_equal(mask, mask.T)


def test_ties_and_invalid_distances():
    d = np.ones((5, 5))-np.eye(5)
    _, first, edges = t.merge_graph(d)
    _, second, repeated = t.merge_graph(d)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(edges, repeated)
    assert edges.iloc[0].left == 0 and edges.iloc[0].right == 1
    with pytest.raises(ValueError):
        t.merge_graph(np.array([[0., np.nan], [np.nan, 0.]]))


def test_masked_scalar_weights_and_total_coupling():
    d = squareform(pdist(np.arange(8)[:, None]))
    _, mask, _ = t.merge_graph(d)
    scalar = np.exp(-.5*(d/2)**2)
    kernel, factor = t.h.match_kernel(mask, scalar)
    np.testing.assert_allclose(kernel[mask], scalar[mask]*factor)
    np.testing.assert_array_equal(kernel[~mask & ~np.eye(8, dtype=bool)], 0.)
    np.testing.assert_allclose(kernel.sum()-np.trace(kernel), scalar.sum()-np.trace(scalar))
    w = np.random.default_rng(3).dirichlet(np.ones(8))
    c = w-1/len(w)
    efficient = len(w)**2*(np.dot(w*c*c, kernel@w)-np.dot(w*c, kernel@(w*c)))
    np.testing.assert_allclose(efficient, r.pairwise_penalty(w, kernel), rtol=1e-12)


def test_selection_uses_mse_and_excludes_unresolved():
    rows = [dict(arm=25, family='hierarchical_rmsd', mse=.01, converged=False, recovery=99),
            dict(arm=26, family='hierarchical_rmsd', mse=.02, converged=True, recovery=30),
            dict(arm=27, family='hierarchical_rmsd', mse=.03, converged=True, recovery=90),
            dict(arm=31, family='hierarchical_w1', mse=.01, converged=False, recovery=99)]
    assert t.select(rows).arm.tolist() == [26]
    assert len(t.specs()) == 18
    assert {row['strength'] for row in t.specs()} == {.1}
