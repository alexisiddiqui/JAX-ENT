import importlib
import numpy as np
import pandas as pd

s=importlib.import_module('jaxent.examples.1_IsoValidation_OMass.analysis.omc_hierarchical_strength')


def test_grid_and_strength_forwarding(monkeypatch,tmp_path):
    specs=s.specs()
    assert len(specs)==18 and len({r['arm'] for r in specs})==18
    assert {r['strength'] for r in specs}=={.01,.03}
    assert {r['quantile'] for r in specs}=={.16,.32,.64}
    observed=[]
    def fake(task):
        folder,row=task
        assert str(tmp_path/row['distance'])==folder
        observed.append(row['strength'])
        return row
    monkeypatch.setattr(s.s,'run_arm',fake)
    for row in specs:
        assert s.run_arm((str(tmp_path),row))==row
    assert observed==[r['strength'] for r in specs]


def test_selection_and_valid_paired_comparisons():
    rows=[dict(arm=28,family='hierarchical_rmsd',quantile=.16,strength=.1,converged=False,mse=.01,recovery=70,intermediate=.2,ess_fraction=.004),
          dict(arm=43,family='hierarchical_rmsd',quantile=.16,strength=.01,converged=True,mse=.02,recovery=60,intermediate=.3,ess_fraction=.005),
          dict(arm=45,family='hierarchical_rmsd',quantile=.32,strength=.01,converged=False,mse=.001,recovery=90,intermediate=.1,ess_fraction=.01)]
    assert s.select(rows).arm.tolist()==[43]
    paired=s.comparisons(pd.DataFrame(rows[:2]))
    assert not paired.valid_pair.iloc[0]
    assert np.isnan(paired.delta_recovery.iloc[0])


def test_kernel_lookup_is_strength_independent(tmp_path):
    row=s.specs()[0]
    first=s.kernel_path(tmp_path,row)
    row['strength']=.03
    assert s.kernel_path(tmp_path,row)==first==tmp_path/'rmsd/kernel_3.npy'
