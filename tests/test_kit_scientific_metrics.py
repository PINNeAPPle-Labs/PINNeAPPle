"""Unit tests for multi-seed and conservation postprocessing."""
import numpy as np
import pytest
from pinneapple_lab.experiments.kit_scientific_metrics import (
    seed_summary, ensemble_interval, energy_balance,
)
from examples.real_world_thermal.run_cross_campaign import rc_rollout


def test_seed_statistics():
    rows=[{"model":"x","seed":0,"rmse_c":1.0},
          {"model":"x","seed":1,"rmse_c":3.0},
          {"model":"rc","seed":None,"rmse_c":2.0}]
    summary=seed_summary(rows)
    assert summary["x"]["rmse_mean_c"]==2.0
    assert summary["x"]["n"]==2
    assert "rc" not in summary


def test_ensemble_interval_shapes():
    lo,hi=ensemble_interval([[1,2,3],[3,4,5]],coverage=.9)
    assert lo.shape==(3,)
    assert np.all(hi>=lo)
    with pytest.raises(ValueError):
        ensemble_interval([[1,2]])


def test_exact_euler_energy_balance():
    n=20
    dt=60
    temp=np.empty(n)
    temp[0]=20
    ambient=np.full(n,5.0)
    q=np.full(n,500.0)
    c,h,eta=200000.0,40.0,.9
    for i in range(n-1):
        temp[i+1]=temp[i]+dt*(h*(ambient[i]-temp[i])+eta*q[i])/c
    result=energy_balance(temp,ambient,q,dt,capacity_j_per_k=c,
                          conductance_w_per_k=h,heating_efficiency=eta)
    assert result["residual_rmse_w"]<1e-8


def test_autonomous_rollout_does_not_use_future_observations():
    coefficients=np.array([.1,.2,0.0])
    a=np.full(10,5.0)
    heat=np.ones(10)
    pred=rc_rollout(20,a,heat,coefficients)
    assert len(pred)==11
    assert pred[1]==pytest.approx(18.7)
