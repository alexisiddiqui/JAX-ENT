import importlib
import numpy as np

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_rate')


def test_mean_rate_averages_rates_before_exponential():
    rates=np.array([[.1,3.],[1.,5.]])
    w=np.array([.4,.6])
    groups=np.array([0,1])
    times=np.array([.2,1.])
    actual=s.e.predict(rates,w,groups,'rate',times=times)
    np.testing.assert_allclose(actual,1-np.exp(-times[:,None]*(rates@w)[None,:]))
    assert not np.allclose(actual,s.e.predict(rates,w,groups,'uptake',times=times))
    np.testing.assert_array_equal(actual,s.e.predict(rates,w,np.array([1,0]),'rate',times=times))


def test_runner_routes_mean_rate_and_omc_kernel(monkeypatch,tmp_path):
    observed=[]
    def fake(task):
        folder,mode,ensemble,spec=task
        assert 'stage' not in spec and 'ensemble' not in spec
        observed.append((mode,ensemble,spec['family'],spec['strength']))
        return dict(spec,stage=mode,converged=False,error='test stub')
    monkeypatch.setattr(s.e,'run_arm',fake)
    rows=s.specs()
    assert len(rows)==18
    for row in rows:
        result=s.run_arm((str(tmp_path),row))
        assert result['family']==row['family'] and result['stage']=='rate'
    assert observed==[('rate','ISO_TRI','scalar_logpf',.1)]*18
    assert {row['quantile'] for row in rows}=={.02,.04,.08,.16,.32,.64}


def test_actual_executor_saves_mean_rate_result(monkeypatch,tmp_path):
    row=s.specs()[0]
    folder=tmp_path/'rmsd/ISO_TRI'
    folder.mkdir(parents=True)
    rates=np.array([[.1,1.,3.],[.3,2.,5.]])
    groups=np.array([0,1,-1])
    w=np.array([.4,.5,.1])
    target=s.e.predict(rates,w,groups,'rate')
    np.savez(folder/'input.npz',rates=rates,groups=groups,target=target,scale=1.)
    np.save(folder/'scalar_logpf_0.npy',np.eye(3))
    def fake_fit(data,spec,mode,kernel):
        assert mode=='rate' and spec['strength']==.1
        np.testing.assert_array_equal(kernel,np.eye(3))
        return dict(weights=w,initial_weights=np.stack([w,w]),converged=True,objective=0.,
                    prediction=target,objective_gap=0.,initial_steps=np.array([1000,1000]))
    monkeypatch.setattr(s.e,'fit',fake_fit)
    result=s.run_arm((str(tmp_path),row))
    assert result['converged'] and result['family']=='hierarchical_rmsd'
    assert result['stage']=='rate' and result['ensemble']=='ISO_TRI'
    assert (tmp_path/'fits/arm_25/complete.json').exists()
    assert s.run_arm((str(tmp_path),row))==result
