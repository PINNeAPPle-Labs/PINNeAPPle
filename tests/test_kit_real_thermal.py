"""Smoke and scientific guardrails for real-world thermal benchmark.

Synthetic arrays below are explicitly unit-test fixtures only, never evidence
that a neural model beats experimentally observed measurements.
"""
import numpy as np
import pandas as pd
import pytest
from pinneapple_lab.experiments.real_world_thermal import (
    chronological, load_real, rc_baseline, score,
)


def test_chronological_split():
    parts = chronological(100)
    assert [len(parts[x]) for x in ("train", "val", "test")] == [60,20,20]
    assert max(parts["train"]) < min(parts["val"])
    assert max(parts["val"]) < min(parts["test"])


def test_known_rc_coefficients_unit_fixture():
    # Variation makes parameters identifiable, unlike constant-temperature data.
    n = 400
    t = np.arange(n, dtype=float)
    x = np.column_stack((20 + np.sin(t/17), 7 + np.cos(t/29), 1 + np.sin(t/11)))
    true = np.array([.02, .12, .001])
    y = x[:,0] + true[0]*(x[:,1]-x[:,0]) + true[1]*x[:,2] + true[2]
    coeff, pred = rc_baseline(x,y,np.arange(240))
    assert np.allclose(coeff,true,atol=1e-7)
    assert score(y,pred)["rmse_c"] < 1e-7


def test_no_synthetic_fallback_when_source_missing(tmp_path):
    mapping = {"time":"stamp","temperature":"T","outdoor":"ambient","heat_w":"heat"}
    with pytest.raises(FileNotFoundError):
        load_real(tmp_path / "nonexistent.csv",mapping)


def test_time_gap_rejected(tmp_path):
    n=400
    times=pd.date_range("2025-12-20", periods=n, freq="min", tz="UTC").to_series().reset_index(drop=True)
    times.iloc[300] += pd.Timedelta(minutes=1)
    source=tmp_path/"gapped.csv"
    pd.DataFrame({"stamp":times,"T":np.full(n,21.0),
                  "ambient":np.full(n,3.0),"heat":np.full(n,1000.0)}).to_csv(source,index=False)
    mapping = {"time":"stamp","temperature":"T","outdoor":"ambient","heat_w":"heat"}
    with pytest.raises(ValueError,match="Temporal gaps"):
        load_real(source,mapping)
