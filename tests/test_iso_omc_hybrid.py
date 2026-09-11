import importlib

import numpy as np
import pandas as pd
import pytest

h = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_control')
r = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_report')


def test_neighbour_ties_symmetry_and_self_exclusion():
    mask = h.neighbourhood(np.zeros((2,4)), k=1)
    expected = np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]],bool)
    np.testing.assert_array_equal(mask,expected)
    assert not np.diag(mask).any()


def test_profile_prevents_scalar_cancellation_and_preserves_components():
    values = np.array([[2,3,18,17],[10,10,10,10],[18,17,2,3]])
    mask = h.neighbourhood(values,k=1)
    assert mask[0,1] and mask[2,3] and not mask[0,2]
    count,_ = h.connected_components(mask,directed=False)
    assert count == 2


def test_masked_scalar_weights_and_coupling():
    mask = h.neighbourhood(np.array([[0.,1.,3.,4.]]),k=1)
    scalar = np.exp(-.5*(np.arange(4)[:,None]-np.arange(4)[None,:])**2)
    kernel,factor = h.match_kernel(mask,scalar)
    off = ~np.eye(4,dtype=bool)
    np.testing.assert_allclose(kernel[off].mean(),scalar[off].mean(),rtol=1e-14)
    np.testing.assert_array_equal(kernel[off & ~mask],0.)
    np.testing.assert_allclose(kernel[mask],scalar[mask]*factor)
    np.testing.assert_array_equal(np.diag(kernel),1.)
    with pytest.raises(ValueError):
        h.match_kernel(np.zeros((4,4),bool),scalar)


def test_original_fitter_omc_branch_and_pairwise_objective():
    rates=np.array([[.01,.1,1.,3.],[.3,.1,.02,.01]])
    groups=np.array([0,0,1,1])
    weights=np.array([.1,.3,.2,.4])
    target=h.e.predict(rates,weights,groups,'frame_uptake')
    data=dict(rates=rates,groups=groups,target=target,scale=np.var(target)+1e-8)
    mask=h.neighbourhood(np.array([[0,1,10,11]]),k=1)
    kernel,_=h.match_kernel(mask,np.ones((4,4)))
    result=h.checked_fit(data,kernel,checkpoints=(30,),window=10)
    expected=np.mean((result['prediction']-target)**2)/data['scale']+.1*r.pairwise_penalty(result['weights'],kernel)
    np.testing.assert_allclose(result['objective'],expected,rtol=1e-10)


def sample_rows():
    return [dict(stage='uptake',ensemble='ISO_BI',family=family,arm=i,quantile=.02,
                 converged=True,mse=.01,tv=.1,recovery=80.,ess_fraction=ess,intermediate=0.)
            for i,(family,ess) in enumerate([('scalar_logpf',.1),('profile_logpf',.3),('maxent',.5),(h.FAMILY,.11)])]


def test_comparison_filter_and_ess_tolerance():
    rows=sample_rows()
    pairs,matched=r.comparisons(pd.DataFrame(rows))
    assert pairs.valid_pair.all()
    assert matched.matched.tolist()==[True,False,False]
    rows[0]['converged']=False
    pairs,_=r.comparisons(pd.DataFrame(rows))
    assert not pairs.iloc[0].valid_pair and np.isnan(pairs.iloc[0].delta_tv)


def test_selection_uses_mse_not_recovery():
    rows=sample_rows()
    rows.append(dict(rows[-1],arm=10,mse=.1,recovery=100))
    selected=h.e.select(rows)
    assert selected.loc[selected.family==h.FAMILY].iloc[0].arm==3


def test_manifest_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'identity',lambda:{'test':1})
    artifact=tmp_path/'data'
    artifact.write_text('original')
    h.e.write_json(tmp_path/'manifest.json',dict(code=h.identity(),hashes={str(artifact):h.e.digest(artifact)}))
    h.validate(tmp_path)
    artifact.write_text('changed')
    with pytest.raises(ValueError):
        h.validate(tmp_path)


def test_resume_and_checksum(tmp_path,monkeypatch):
    folder=tmp_path/'ISO_BI'
    folder.mkdir()
    groups=np.array([0,1])
    target=np.ones((5,1))*.2
    np.savez(folder/'input.npz',groups=groups,target=target)
    np.save(folder/'hybrid_logpf_0.npy',np.eye(2))
    weights=np.array([.4,.6])
    result=dict(weights=weights,initial_weights=np.stack([weights,weights]),prediction=target,
                objective=0.,converged=True,objective_gap=0.,initial_steps=np.array([1000,1000]))
    monkeypatch.setattr(h,'checked_fit',lambda *a,**k:result)
    task=(str(tmp_path),'ISO_BI',0)
    first=h.run_arm(task)
    monkeypatch.setattr(h,'checked_fit',lambda *a,**k:pytest.fail('refitted saved result'))
    assert h.run_arm(task)==first
    (tmp_path/'uptake/ISO_BI/arm_19/fit.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        h.run_arm(task)


def test_failed_fit_is_recorded_and_not_completed(tmp_path,monkeypatch):
    folder=tmp_path/'ISO_BI'
    folder.mkdir()
    np.savez(folder/'input.npz',groups=np.array([0,1]))
    np.save(folder/'hybrid_logpf_0.npy',np.eye(2))
    def fail(*args,**kwargs):
        raise FloatingPointError('nonfinite test')
    monkeypatch.setattr(h,'checked_fit',fail)
    row=h.run_arm((str(tmp_path),'ISO_BI',0))
    assert not row['converged'] and row['error']=='nonfinite test'
    assert (tmp_path/'uptake/ISO_BI/arm_19/failure.json').exists()
    assert not (tmp_path/'uptake/ISO_BI/arm_19/complete.json').exists()


def test_unconverged_hybrid_cannot_form_valid_comparisons():
    rows=sample_rows()
    rows[-1]['converged']=False
    pairs,matched=r.comparisons(pd.DataFrame(rows))
    assert not pairs.valid_pair.any()
    assert not matched.matched.any()
