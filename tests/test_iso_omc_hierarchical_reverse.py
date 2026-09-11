import importlib
import numpy as np

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_reverse')


def test_reference_target_uses_reference_mean_rates():
    features=dict(heavy_contacts=np.array([[1.,3.,8.],[4.,2.,1.]]),acceptor_contacts=np.zeros((2,3)),k_ints=np.array([1.,2.]))
    weights=np.array([.2,.2,.6])
    groups=np.array([0,0,1])
    target,rates=s.reference_target(features,weights,groups)
    expected=1-np.exp(-s.e.TIMES[:,None]*(rates@weights)[None,:])
    np.testing.assert_allclose(target,expected)
    mixture=s.e.predict(rates,weights,groups,'frame_uptake')
    assert np.all(target>=mixture-1e-12)
    assert not np.allclose(target,mixture)


def test_real_executor_routes_new_target_and_grouped_prediction(monkeypatch,tmp_path):
    for row in (s.specs()[0],s.specs()[-1]):
        distance=row.get('distance','baseline')
        folder=tmp_path/distance/'ISO_TRI'
        folder.mkdir(parents=True,exist_ok=True)
        rates=np.array([[.1,1.,3.],[.3,2.,5.]])
        groups=np.array([0,1,-1])
        weights=np.array([.4,.5,.1])
        target=s.e.predict(rates,weights,groups,'rate')
        np.savez(folder/'input.npz',rates=rates,groups=groups,target=target,scale=1.)
        np.save(folder/'scalar_logpf_5.npy',np.eye(3))
        def fake_fit(data,spec,mode,kernel):
            assert mode=='uptake'
            np.testing.assert_array_equal(data['target'],target)
            prediction=s.e.predict(data['rates'],weights,groups,mode)
            assert not np.allclose(prediction,target)
            return dict(weights=weights,initial_weights=np.stack([weights,weights]),converged=True,objective=0.,
                        prediction=prediction,objective_gap=0.,initial_steps=np.array([1000,1000]))
        monkeypatch.setattr(s.e,'fit',fake_fit)
        result=s.run_arm((str(tmp_path),row))
        assert result['family']==row['family'] and result['target_mode']=='rate' and result['stage']=='uptake'
        assert s.run_arm((str(tmp_path),row))==result
    assert len(s.specs())==37 and len({r['arm'] for r in s.specs()})==37
