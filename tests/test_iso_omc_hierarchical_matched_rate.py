import importlib
import numpy as np

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_matched_rate')


def test_consistent_mean_rate_recovers_target_at_generating_weights():
    features=dict(heavy_contacts=np.array([[1.,3.,8.],[4.,2.,1.]]),acceptor_contacts=np.zeros((2,3)),k_ints=np.array([1.,2.]))
    w=np.array([.2,.2,.6])
    groups=np.array([0,0,1])
    target,rates=s.reverse.reference_target(features,w,groups)
    np.testing.assert_allclose(s.e.predict(rates,w,groups,'rate'),target,atol=1e-14)
    assert not np.allclose(s.e.predict(rates,w,groups,'uptake'),target)


def test_real_executor_mean_rate_routing_and_resume(monkeypatch,tmp_path):
    for row in (s.specs()[0],s.specs()[-1]):
        folder=tmp_path/row.get('distance','baseline')/'ISO_TRI'
        folder.mkdir(parents=True,exist_ok=True)
        rates=np.array([[.1,1.,3.],[.3,2.,5.]])
        groups=np.array([0,1,-1])
        weights=np.array([.4,.5,.1])
        target=s.e.predict(rates,weights,groups,'rate')
        np.savez(folder/'input.npz',rates=rates,groups=groups,target=target,scale=1.)
        np.save(folder/'scalar_logpf_5.npy',np.eye(3))
        def fake_fit(data,spec,mode,kernel):
            assert mode=='rate'
            prediction=s.e.predict(data['rates'],weights,groups,mode)
            np.testing.assert_array_equal(prediction,data['target'])
            return dict(weights=weights,initial_weights=np.stack([weights,weights]),converged=True,objective=0.,
                        prediction=prediction,objective_gap=0.,initial_steps=np.array([1000,1000]))
        monkeypatch.setattr(s.e,'fit',fake_fit)
        result=s.run_arm((str(tmp_path),row))
        assert result['family']==row['family'] and result['target_mode']=='rate' and result['stage']=='rate'
        assert s.run_arm((str(tmp_path),row))==result
    rows=s.specs()
    assert len(rows)==37 and len({r['arm'] for r in rows})==37
    assert {r['strength'] for r in rows if r['arm']>=25}=={.1}
