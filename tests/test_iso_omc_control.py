import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)
e = importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_control')


def test_forward_parity_and_zero_group():
    from jaxent.src.models.HDX.BV.features import BV_input_features
    from jaxent.src.models.HDX.forward import BV_uptake_ForwardPass
    from jaxent.src.models.config import BV_model_Config
    heavy = np.array([[1.,3.,5.],[2.,4.,1.]])
    acceptor = np.array([[0.,1.,0.],[1.,0.,1.]])
    kints = np.array([.5,1.2])
    features = BV_input_features(heavy_contacts=heavy,acceptor_contacts=acceptor,k_ints=kints)
    config = BV_model_Config(timepoints=jnp.asarray(e.TIMES),kint_unit='min^-1')
    groups = np.array([0,1,-1])
    rates = kints[:,None]*np.exp(-.35*heavy-2*acceptor)
    for weights in (np.array([.4,.6,0.]),np.array([.2,.5,.3])):
        for mode in ('uptake','rate','frame_uptake'):
            forward = BV_uptake_ForwardPass(mode,groups)
            expected = np.asarray(forward.average_frames(features,config.forward_parameters,jnp.asarray(weights)).uptake)
            np.testing.assert_allclose(e.predict(rates,weights,groups,mode),expected,atol=1e-12)
            gradient = jax.grad(lambda w:jnp.sum(e.predict(jnp.asarray(rates),w,jnp.asarray(groups),mode,xp=jnp)))(jnp.asarray(weights))
            assert np.isfinite(gradient).all()


def rows(score=51):
    return [dict(stage='uptake',ensemble=ensemble,family=family,arm=i,mse=.01,converged=True,recovery=score)
            for ensemble in ('ISO_BI','ISO_TRI') for i,family in enumerate(e.FAMILIES)]


@pytest.mark.parametrize('score,expected',[(49.9,False),(50,False),(50.1,True)])
def test_gate_threshold(score,expected):
    assert e.gate(rows(score))['passed'] is expected


def test_gate_requires_numerical_and_mse_selected():
    values = rows(49)
    values.append(dict(stage='uptake',ensemble='ISO_TRI',family='scalar_logpf',arm=18,mse=.1,converged=True,recovery=99))
    assert not e.gate(values)['passed']
    values = rows(70)
    values[0]['converged'] = False
    assert not e.gate(values)['passed']
    values = rows(70)
    for row in values:
        if row['ensemble']=='ISO_TRI' and row['family'] in e.FAMILIES[2:]:
            row['converged']=False
    assert not e.gate(values)['passed']


def test_selection_does_not_use_truth():
    values = rows()
    before = e.select(values).arm.tolist()
    for row in values:
        row['recovery'] = -999
    assert e.select(values).arm.tolist()==before


def test_recovery_includes_intermediate():
    groups=np.array([0,1,-1])
    assert e.recovery(np.array([.4,.6,0.]),groups)==100
    assert e.recovery(np.array([0.,0.,1.]),groups)==0


def test_alignment():
    assert e.alignment([('A',(1,)),('A',(2,))],[('A',(2,)),('A',(1,))]).tolist()==[1,0]
    with pytest.raises(ValueError):
        e.alignment([('A',(1,))],[('B',(1,))])


def test_fit_objective_and_two_starts():
    rates=np.array([[.1,.3,.6],[.3,.05,.4]])
    groups=np.array([0,1,-1])
    target=e.predict(rates,np.array([.4,.6,0.]),groups,'frame_uptake')
    data=dict(rates=rates,groups=groups,target=target,scale=np.var(target)+1e-8)
    kernel=np.array([[1.,.8,.2],[.8,1.,.3],[.2,.3,1.]])
    spec=dict(family='scalar_logpf',strength=.1)
    result=e.fit(data,spec,'uptake',kernel,checkpoints=(30,),window=10)
    w=result['weights']
    penalty=.5*len(w)**2*np.sum(w[:,None]*w[None,:]*kernel*(w[:,None]-w[None,:])**2)
    expected=np.mean((result['prediction']-target)**2)/data['scale']+.1*penalty
    np.testing.assert_allclose(result['objective'],expected,rtol=1e-10)
    assert result['initial_weights'].shape==(2,3)
    assert np.isfinite(result['initial_grad_norm']).all()


def test_manifest_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'identity',lambda:{'code':'test'})
    artifact=tmp_path/'input'
    artifact.write_text('good')
    e.write_json(tmp_path/'manifest.json',dict(code=e.identity(),hashes={str(artifact):e.digest(artifact)}))
    e.load_manifest(tmp_path)
    artifact.write_text('bad')
    with pytest.raises(ValueError):
        e.load_manifest(tmp_path)


def test_nearest_ess_unmatched():
    import pandas as pd
    r=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_control_report')
    values=[dict(stage='uptake',ensemble='ISO_TRI',family=f,arm=i,converged=True,ess_fraction=ess,tv=.1,mse=.01)
            for i,(f,ess) in enumerate([('maxent',.3),('scalar_logpf',.6)])]
    result=r.nearest_ess(pd.DataFrame(values))
    assert not result.iloc[0].matched
    assert np.isnan(result.iloc[0].delta_tv)


def test_time_unit_equivalence():
    rates=np.array([[.01,.1],[.2,.03]])
    groups=np.array([0,1])
    weights=np.array([.4,.6])
    for mode in ('frame_uptake','uptake','rate'):
        np.testing.assert_allclose(e.predict(rates,weights,groups,mode),
            e.predict(rates/60,weights,groups,mode,times=e.TIMES*60),rtol=1e-14,atol=1e-14)


def test_resume_and_corruption(tmp_path,monkeypatch):
    folder=tmp_path/'ISO_BI'
    folder.mkdir()
    groups=np.array([0,1,-1])
    w=np.array([.4,.6,0.])
    target=np.ones((5,2))*.2
    np.savez(folder/'input.npz',groups=groups,target=target)
    result=dict(weights=w,prediction=target,objective=0.,converged=True,objective_gap=0.,
                initial_steps=np.array([1000,1000]),initial_weights=np.stack([w,w]))
    monkeypatch.setattr(e,'fit',lambda *a,**k:result)
    task=(str(tmp_path),'uptake','ISO_BI',e.specs()[0])
    first=e.run_arm(task)
    monkeypatch.setattr(e,'fit',lambda *a,**k:pytest.fail('resume refitted'))
    assert e.run_arm(task)==first
    path=tmp_path/'uptake/ISO_BI/arm_00/fit.npz'
    path.write_bytes(b'broken')
    with pytest.raises(ValueError):
        e.run_arm(task)


def test_numerical_failure_record(tmp_path,monkeypatch):
    folder=tmp_path/'ISO_BI'
    folder.mkdir()
    np.savez(folder/'input.npz',groups=np.array([0,1]))
    def fail(*args,**kwargs):
        raise FloatingPointError('test failure')
    monkeypatch.setattr(e,'fit',fail)
    row=e.run_arm((str(tmp_path),'uptake','ISO_BI',e.specs()[0]))
    assert not row['converged']
    assert (tmp_path/'uptake/ISO_BI/arm_00/failure.json').exists()
    assert not (tmp_path/'uptake/ISO_BI/arm_00/complete.json').exists()


def test_grouped_rate_average_is_not_frame_or_logpf_average():
    rates=np.array([[.01,1.,.1,3.]])
    groups=np.array([0,0,1,1])
    weights=np.array([.1,.3,.2,.4])
    expected=.4*-np.expm1(-e.TIMES*(.1*.01+.3*1.)/.4)
    expected+=.6*-np.expm1(-e.TIMES*(.2*.1+.4*3.)/.6)
    grouped=e.predict(rates,weights,groups,'uptake')[:,0]
    np.testing.assert_allclose(grouped,expected)
    assert not np.allclose(grouped,e.predict(rates,weights,groups,'frame_uptake')[:,0])
    assert not np.allclose(grouped,e.predict(rates,weights,groups,'rate')[:,0])
    geometric=.4*-np.expm1(-e.TIMES*np.exp((.1*np.log(.01)+.3*np.log(1.))/.4))
    geometric+=.6*-np.expm1(-e.TIMES*np.exp((.2*np.log(.1)+.4*np.log(3.))/.6))
    assert not np.allclose(grouped,geometric)
