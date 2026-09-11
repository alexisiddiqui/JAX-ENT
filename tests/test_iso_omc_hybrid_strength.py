import importlib

import numpy as np
import pandas as pd
import pytest

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_strength')
r=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hybrid_strength_report')


def test_requested_strength_reaches_general_fitter(monkeypatch):
    seen=[]
    def fake(data,spec,mode,kernel,**kwargs):
        seen.append((spec,mode,kernel))
        return 'fit'
    monkeypatch.setattr(s.e,'fit',fake)
    monkeypatch.setattr(s.h,'checked_fit',lambda *a,**k:pytest.fail('hardcoded strength wrapper used'))
    kernel=np.eye(2)
    for strength in s.STRENGTHS:
        assert s.fit_at_strength({},kernel,strength)=='fit'
    assert [entry[0]['strength'] for entry in seen]==[.01,.03]
    assert all(entry[0]['family']=='scalar_logpf' and entry[1]=='uptake' and entry[2] is kernel for entry in seen)


def test_strength_changes_only_regularisation_at_fixed_weights():
    weights=np.array([.2,.3,.5])
    rates=np.array([[.1,.4,1.],[.2,.01,.3]])
    groups=np.array([0,1,-1])
    data=dict(rates=rates,groups=groups,target=np.zeros((5,2)),scale=.2)
    kernel=np.array([[1.,.2,.5],[.2,1.,.1],[.5,.1,1.]])
    penalty=.5*9*np.sum(weights[:,None]*weights[None,:]*kernel*(weights[:,None]-weights[None,:])**2)
    np.testing.assert_allclose(r.objective_at_weights(data,weights,kernel,.03)-r.objective_at_weights(data,weights,kernel,.01),.02*penalty,atol=1e-14)


def test_fitted_objective_matches_requested_strength():
    rates=np.array([[.01,.1,1.],[.2,.4,.03]])
    groups=np.array([0,1,-1])
    target=s.e.predict(rates,np.array([.4,.6,0.]),groups,'frame_uptake')
    data=dict(rates=rates,groups=groups,target=target,scale=np.var(target)+1e-8)
    kernel=np.ones((3,3))
    for strength in s.STRENGTHS:
        result=s.fit_at_strength(data,kernel,strength,checkpoints=(30,),window=10)
        np.testing.assert_allclose(result['objective'],r.objective_at_weights(data,result['weights'],kernel,strength),rtol=1e-10)


def rows():
    return [dict(arm=arm,stage='uptake',ensemble='ISO_TRI',family=family,quantile=q,strength=strength,
                 mse=mse,converged=valid,recovery=70.,tv=.2,intermediate=.2,ess_fraction=ess)
            for arm,family,q,strength,mse,valid,ess in [
                (22,s.h.FAMILY,.16,.1,.01,True,.3),(25,s.h.FAMILY,.16,.01,.005,True,.4),
                (26,s.h.FAMILY,.16,.03,.001,False,.2),(1,'maxent',None,.01,.004,True,.39)]]


def test_selection_and_pair_filtering():
    values=rows()
    assert s.select(values).iloc[0].arm==25
    values[0]['recovery']=100
    assert s.select(values).iloc[0].arm==25
    pairs,matched=r.comparisons(pd.DataFrame(values))
    assert pairs.valid_pair.tolist()==[True,False]
    assert matched.matched.tolist()==[False,True,False,False]
    assert np.isnan(pairs.iloc[1].delta_mse)


def test_resume_and_checksum(tmp_path,monkeypatch):
    weights=np.array([.4,.6])
    target=np.ones((5,1))*.2
    np.savez(tmp_path/'input.npz',groups=np.array([0,1]),target=target)
    np.save(tmp_path/'kernel_3.npy',np.eye(2))
    result=dict(weights=weights,initial_weights=np.stack([weights,weights]),prediction=target,
                objective=0.,converged=True,objective_gap=0.,initial_steps=np.array([1000,1000]))
    monkeypatch.setattr(s,'fit_at_strength',lambda *a,**k:result)
    task=(str(tmp_path),s.specs()[0])
    row=s.run_arm(task)
    monkeypatch.setattr(s,'fit_at_strength',lambda *a,**k:pytest.fail('unexpected refit'))
    assert s.run_arm(task)==row
    (tmp_path/'fits/arm_25/fit.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        s.run_arm(task)


def test_manifest_rejects_changed_kernel(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'identity',lambda:{'test':True})
    path=tmp_path/'kernel.npy'
    np.save(path,np.eye(2))
    s.e.write_json(tmp_path/'manifest.json',dict(code=s.identity(),hashes={str(path):s.e.digest(path)}))
    s.validate(tmp_path)
    np.save(path,np.ones((2,2)))
    with pytest.raises(ValueError):
        s.validate(tmp_path)
