import importlib

import numpy as np
import pytest

n=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_narrow')


def test_fixed_coupling_and_mask_across_widths():
    values=np.array([[0.,.2,.6,1.2],[.1,.4,.7,1.3]])
    mask=n.h.neighbourhood(values,k=2)
    reference,_=n.h.match_kernel(mask,np.ones((4,4)))
    means=[]
    off=~np.eye(4,dtype=bool)
    for scale in (.25,.5,.75,1.):
        kernel,_=n.kernel_at_scale(values,mask,.4*scale,reference)
        means.append(kernel[off].mean())
        np.testing.assert_array_equal(kernel[off & ~mask],0.)
        np.testing.assert_array_equal(np.diag(kernel),1.)
        np.testing.assert_allclose(kernel,kernel.T)
    np.testing.assert_allclose(means,reference[off].mean(),rtol=1e-14)


def test_unit_scale_reconstructs_reference_and_narrowing_changes_relative_edges():
    values=np.array([[0.,.1,.3,.7]])
    mask=n.h.neighbourhood(values,k=2)
    sigma=.2
    d=abs(values.T-values)
    raw=np.exp(-np.minimum(.5*(d/sigma)**2,80.))
    reference,_=n.h.match_kernel(mask,raw)
    reconstructed,_=n.kernel_at_scale(values,mask,sigma,reference)
    np.testing.assert_allclose(reconstructed,reference,atol=1e-14)
    narrow,_=n.kernel_at_scale(values,mask,sigma*.5,reference)
    assert narrow[0,1]/narrow[0,2]>reference[0,1]/reference[0,2]


@pytest.mark.parametrize('sigma',[0.,-1.,np.nan,np.inf])
def test_invalid_bandwidth(sigma):
    with pytest.raises(ValueError):
        n.kernel_at_scale(np.ones((2,3)),np.ones((3,3),bool),sigma,np.eye(3))


def test_manifest_rejects_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(n,'identity',lambda:{'test':True})
    artifact=tmp_path/'data'
    artifact.write_text('original')
    n.e.write_json(tmp_path/'manifest.json',dict(code=n.identity(),hashes={str(artifact):n.e.digest(artifact)}))
    n.validate(tmp_path)
    artifact.write_text('changed')
    with pytest.raises(ValueError):
        n.validate(tmp_path)


def test_resume_checks_archive_without_refitting(tmp_path,monkeypatch):
    groups=np.array([0,1])
    weights=np.array([.4,.6])
    target=np.ones((5,1))*.2
    np.savez(tmp_path/'input.npz',groups=groups,target=target)
    np.save(tmp_path/'kernel_0.5.npy',np.eye(2))
    result=dict(weights=weights,initial_weights=np.stack([weights,weights]),prediction=target,
                objective=0.,converged=True,objective_gap=0.,initial_steps=np.array([1000,1000]))
    monkeypatch.setattr(n.h,'checked_fit',lambda *a,**k:result)
    task=(str(tmp_path),.5)
    row=n.run_arm(task)
    monkeypatch.setattr(n.h,'checked_fit',lambda *a,**k:pytest.fail('unexpected refit'))
    assert n.run_arm(task)==row
    assert n.collect(tmp_path)==[row]
    (tmp_path/'fits/scale_0.5/fit.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        n.run_arm(task)


def test_numerical_failure_is_excluded_from_completion(tmp_path,monkeypatch):
    np.savez(tmp_path/'input.npz',groups=np.array([0,1]))
    np.save(tmp_path/'kernel_0.25.npy',np.eye(2))
    def fail(*args,**kwargs):
        raise FloatingPointError('test failure')
    monkeypatch.setattr(n.h,'checked_fit',fail)
    row=n.run_arm((str(tmp_path),.25))
    assert not row['converged']
    assert (tmp_path/'fits/scale_0.25/failure.json').exists()
    assert not (tmp_path/'fits/scale_0.25/complete.json').exists()
