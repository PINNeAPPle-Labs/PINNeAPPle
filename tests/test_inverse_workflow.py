"""The standard inverse-modeling workflow (#1189): identifiability before the fit, estimates whose intervals hold the
truth, and a structurally non-identifiable design refused with the combination named."""
from __future__ import annotations

import numpy as np

from pinneapple_analysis.inverse_problems.workflow import inverse_workflow
from pinneapple_physics.reactors.kinetics import first_order_pfr_conversion as pfr

TOOL = "pinneapple_physics.reactors.kinetics:first_order_pfr_conversion"
P = {"A": {"lower": 1e4, "upper": 1e9, "initial": 1e6}, "Ea": {"lower": 3e4, "upper": 1e5, "initial": 5e4}}
TRUTH = {"A": 5e6, "Ea": 6e4}


def _data(T, tau, seed=0, sigma=0.01):
    y = np.array([pfr(t, u, **TRUTH)["X"] for t, u in zip(T, tau, strict=True)])
    return {"inputs": {"T": T, "tau": tau}, "values": y + sigma * np.random.default_rng(seed).standard_normal(len(y)),
            "sigma": sigma}


def test_several_temperatures_identify_a_and_ea_inside_their_intervals():
    T = np.repeat(np.linspace(320, 380, 7), 3)
    tau = np.tile([20.0, 60.0, 180.0], 7)
    r = inverse_workflow(tool=TOOL, measurements=_data(T, tau), output="X", parameters=P, n_recovery=6)
    assert r["estimated"] and all(c["status"] == "pass" for c in r["checks"]), r["checks"]
    for k, v in TRUTH.items():
        lo, hi = r["intervals"][k]
        assert lo <= v <= hi


def test_one_temperature_is_refused_with_the_combination_named():
    T = np.full(9, 350.0)
    tau = np.tile([20.0, 60.0, 180.0], 3)
    r = inverse_workflow(tool=TOOL, measurements=_data(T, tau), output="X", parameters=P)
    assert r["estimated"] is False
    c = r["checks"][0]
    assert c["code"] == "identify.structural" and c["status"] == "fail" and "rank 1 of 2" in c["detail"]


def test_a_wrong_model_leaves_structured_residuals():
    T = np.repeat(np.linspace(320, 380, 7), 3)
    tau = np.tile([20.0, 60.0, 180.0], 7)
    meas = _data(T, tau)
    meas["values"] = np.asarray(meas["values"]) + 0.08 * np.sin(np.linspace(0, 3 * np.pi, len(T)))  # model gap
    r = inverse_workflow(tool=TOOL, measurements=meas, output="X", parameters=P, n_recovery=0)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["identify.fits_data"] == "fail"


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "inverse.workflow" in REGISTRY.types()
    assert match("estimar parâmetros cinéticos a partir das medições (identificabilidade)")[0][1].id == \
        "inverse.workflow"
