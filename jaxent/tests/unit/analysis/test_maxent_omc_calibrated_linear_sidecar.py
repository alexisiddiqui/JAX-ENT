from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"examples/1_IsoValidation_OMass/fitting/jaxENT"))
import run_maxent_omc_calibrated_linear_sidecar as sidecar


def grid(tmp_path):
    args=SimpleNamespace(output_dir=tmp_path,features_dir=tmp_path,datasplit_dir=tmp_path,
        clustering_dir=tmp_path,n_steps=5000,n_splits=3,strengths=list(sidecar.strength.STRENGTHS),
        bandwidths=list(sidecar.strength.BANDWIDTHS))
    values={"uniform_mean":tuple(np.arange(5)/10),"gt_mean":tuple(-np.arange(5)/10)}
    return sidecar.build_specs(args,values)


def test_grid_has_two_frozen_calibration_panels(tmp_path):
    specs=grid(tmp_path)
    assert len(specs)==len({s.run_id for s in specs})==288
    assert {s.calibration_source for s in specs}==set(sidecar.CALIBRATIONS)
    assert {s.uptake_mode for s in specs}=={"linear"}
    assert {s.sigma_source for s in specs}=={"mse"}
    assert all(len(s.interval_offsets)==5 and s.graph_k==0 for s in specs)
    assert sum(s.method=="maxent" for s in specs)==36


def test_base_configures_exact_frozen_offsets(tmp_path):
    spec=grid(tmp_path)[0]
    model=sidecar.base.configure_spec_model(spec,np.array([0,1]))
    np.testing.assert_allclose(model.params.interval_offsets,spec.interval_offsets)
    assert np.isclose(float(model.params.bv_bc),.35)
    assert np.isclose(float(model.params.bv_bh),2.)


def test_plot_has_calibration_panels(tmp_path):
    rows=[dict(calibration_source=s.calibration_source,method=s.method,
        bandwidth=s.bandwidth if s.method=="omc" else np.nan,split_idx=s.split_idx,
        data_strength=s.sweep_maxent,recovery_percent=45.) for s in grid(tmp_path)]
    sidecar.plot_metric(pd.DataFrame(rows),"recovery_percent",tmp_path/"plot")
    assert (tmp_path/"plot.png").is_file() and (tmp_path/"plot.svg").is_file()
