"""The identifiability / calibration / inverse-studio core (pinneapple_analysis.inverse_problems).

References: the analytic pin fin (Incropera, Table 3.4 case A) for the finite-volume rod; central finite
differences for every exact derivative; known truths of synthetic presets for the estimators; the closed-form
ambiguities (fin temperatures depend on h/k only, a cooling curve on h/C only, one-temperature kinetics on the rate
constant only) for the identifiability verdicts.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import pinneapple_analysis.inverse_problems as inv
from pinneapple_analysis.inverse_problems.heat_transfer import fin_exact


def _theta3():
    return {"h1": 20.0, "h2": 60.0, "h3": 30.0, "k": 50.0, "T_base": 90.0, "T_air": 20.0}


def _rod_obs():
    return inv.Observations({"x_mm": [5.0, 30.0, 50.0, 70.0, 95.0, 0.0]}, 0.3,
                            values=np.array([80, 60, 50, 45, 40, 3.0]),
                            observable=np.array(["T"] * 5 + ["q_base"], dtype=object))


def test_rod_matches_analytic_fin():
    rod = inv.RodModel(5, 50, n_zones=1, n_cells=200)
    th = {"h": 25.0, "k": 16.0, "T_base": 80.0, "T_air": 25.0}
    x = np.array([2.0, 10.0, 25.0, 40.0, 49.0])
    T = rod.predict(th, {"x_mm": x})
    exact = 25 + 55 * fin_exact(x / 50, 25.0, 16.0, 0.005, 0.05)
    assert np.max(np.abs(T - exact)) < 5e-3
    q = rod.predict(th, {"x_mm": [0.0]}, np.array(["q_base"], dtype=object))[0]
    q_exact = inv.PinFinModel(5, 50).predict(th, {"x_mm": [0.0]}, np.array(["q_base"], dtype=object))[0]
    assert abs(q - q_exact) / q_exact < 1e-3


def _fd(model, th, obs, names):
    out = np.zeros((obs.n, len(names)))
    for j, n in enumerate(names):
        e = 1e-5 * max(1.0, abs(th[n]))
        a, b = dict(th), dict(th)
        a[n] += e
        b[n] -= e
        out[:, j] = (model.predict(a, obs.inputs, obs.observable) - model.predict(b, obs.inputs, obs.observable)) / (2 * e)
    return out


def test_rod_sensitivities_adjoint_and_torch_are_exact():
    import torch

    rod, th, obs = inv.RodModel(5, 100, n_zones=3), _theta3(), _rod_obs()
    names = list(th)
    pred, S = rod.sensitivities(th, obs, names)
    S_fd = _fd(rod, th, obs, names)
    assert np.abs(S - S_fd).max() / np.abs(S_fd).max() < 1e-6
    phi, g = rod.adjoint_gradient(th, obs, names)
    r = (pred - obs.values) / obs.sigma**2
    assert np.allclose(g, r @ S_fd, rtol=1e-5)
    assert phi == pytest.approx(0.5 * np.sum(((pred - obs.values) / obs.sigma) ** 2))
    tt = {k: torch.tensor(v, dtype=torch.float64) for k, v in th.items()}
    assert np.allclose(rod.predict_torch(tt, obs.inputs, obs.observable).numpy(), pred, atol=1e-9)


def test_fin_temperatures_only_fix_h_over_k_and_design_asks_for_heat_flow():
    fin = inv.PinFinModel(5, 50)
    ps = fin.default_parameters()
    ps[1].fixed = False                                   # k unknown too
    obs = inv.Observations({"x_mm": [10, 20, 30, 40, 50.0]}, 0.5, observable=np.array(["T"] * 5, dtype=object))
    rep = inv.analyze_identifiability(fin, obs, ps, posterior=False)
    assert rep.rank == 1
    assert {p["status"] for p in rep.parameters} == {"structurally non-identifiable"}
    amb = rep.ambiguities[0]
    assert amb["kind"] == "structural" and amb["fixed_combination"] == "h/k"
    assert rep.suggestions[0]["observable"] == "q_base" and rep.suggestions[0]["rank_after"] == 2
    json.dumps(rep.to_dict())

    with_q = obs.concat(inv.Observations({"x_mm": [0.0]}, 0.012, observable=np.array(["q_base"], dtype=object)))
    rep2 = inv.analyze_identifiability(fin, with_q, ps, mcmc_steps=1500)
    assert all(p["status"] == "identifiable" for p in rep2.parameters)
    for p in rep2.parameters:                             # the profile interval contains the nominal (= truth)
        lo, hi = p["profile_ci95"]
        assert lo < p["nominal"] < hi
    assert rep2.posterior is not None and rep2.posterior["n_samples"] > 100


def test_one_temperature_kinetics_is_ambiguous_two_are_not():
    rep1 = inv.analyze_identifiability(inv.get_preset("arrhenius_one_T").build()["model"],
                                       inv.get_preset("arrhenius_one_T").build()["observations"],
                                       inv.get_preset("arrhenius_one_T").build()["parameters"], profile=False,
                                       posterior=False)
    assert all(p["status"] == "structurally non-identifiable" for p in rep1.parameters)
    assert rep1.suggestions[0]["inputs"]["T_K"] != 360.0          # the design asks for another temperature
    b = inv.get_preset("arrhenius_two_T").build()
    rep2 = inv.analyze_identifiability(b["model"], b["observations"], b["parameters"], posterior=False)
    assert all(p["status"] == "identifiable" for p in rep2.parameters)


def test_calibration_reports_not_estimable_and_intervals_that_contain_the_truth():
    b = inv.get_preset("fin_h_k").build()
    r = inv.calibrate(b["model"], b["observations"], b["parameters"], holdout=b["validation"], bootstrap=0)
    for p in r.parameters:
        assert p["ci_jacobian"] is None and p["ci_bootstrap"] is None and "not_estimable" in p
    b = inv.get_preset("fin_h_k_flux").build()
    r = inv.calibrate(b["model"], b["observations"], b["parameters"], holdout=b["validation"], bootstrap=40)
    for p in r.parameters:
        assert p["ci_jacobian"][0] < p["truth"] < p["ci_jacobian"][1]
        assert p["ci_bootstrap"][0] < p["calibrated"] < p["ci_bootstrap"][1]
    assert r.metrics["holdout"]["after"]["rmse"] < r.metrics["holdout"]["before"]["rmse"]
    json.dumps(r.to_dict())


def test_calibration_extrapolation_holdout_by_time():
    b = inv.get_preset("cooling_h").build()
    obs = b["observations"].concat(b["validation"])
    r = inv.calibrate(b["model"], obs, b["parameters"], holdout_fraction=0.2, holdout_by="t_s", bootstrap=0)
    hold_t = np.array(r.rows["inputs"]["t_s"])[np.array(r.rows["set"]) == "holdout"]
    fit_t = np.array(r.rows["inputs"]["t_s"])[np.array(r.rows["set"]) == "fit"]
    assert hold_t.min() > fit_t.max()
    assert r.theta["h"] == pytest.approx(11.0, rel=0.1)
    assert r.metrics["holdout"]["after"]["rmse"] < 3 * 0.3


@pytest.mark.parametrize("preset", ["rod_h_field", "rod_base_bc", "cooling_h", "fin_h_k_flux"])
def test_studio_methods_recover_the_truth(preset):
    p = inv.get_preset(preset)
    out = inv.compare_methods(p.problem(), [m for m in p.methods if m != "pinn"])
    json.dumps(out)
    done = [s for s in out["solutions"] if s["status"] == "done"]
    assert len(done) == len([m for m in p.methods if m != "pinn"])
    for s in done:
        assert s["truth_recovered"], (s["method"], s["parameters"])
    assert out["best"] in [s["method"] for s in done]


def test_studio_flags_non_identifiable_answers_and_unavailable_methods():
    prob = inv.get_preset("fin_h_k").problem()
    avail = inv.available_methods(prob)
    assert not avail["adjoint"]["available"] and not avail["pinn"]["available"]
    sol = inv.solve_inverse(prob, "least_squares",
                            identifiability=inv.analyze_identifiability(prob.model, prob.observations,
                                                                        prob.parameters, profile=False,
                                                                        posterior=False))
    assert all(r["interval95"] is None and "note" in r for r in sol["parameters"])
    assert sol["truth_recovered"] is None
    assert inv.solve_inverse(prob, "adjoint")["status"] == "not applicable"


def test_rod_field_reconstruction_error_is_small():
    p = inv.get_preset("rod_h_field")
    sol = inv.solve_inverse(p.problem(), "adjoint")
    assert sol["fields"]["T"]["values"] and sol["field_error"]["T"]["max_abs"] < 0.5
    assert sol["derivatives"] == "adjoint gradient"


def test_expression_and_table_models():
    cool = inv.LumpedCoolingModel(0.012)
    params = cool.default_parameters()
    expr = inv.ExpressionModel({"T": "T_air + (T0 - T_air) * exp(-h * 0.012 * t_s / C)"}, inputs=["t_s"],
                               parameters=params)
    th = {"h": 11.0, "C": 450.0, "T0": 94.0, "T_air": 23.0}
    t = np.linspace(0, 1800, 7)
    assert np.allclose(expr.predict(th, {"t_s": t}), cool.predict(th, {"t_s": t}))
    with pytest.raises(ValueError):
        inv.ExpressionModel("__import__('os').system('true')", inputs=["t"], parameters=params)

    import pandas as pd

    rows = [{"h": h, "t_s": ti, "T": float(cool.predict({**th, "h": h}, {"t_s": [ti]})[0])}
            for h in np.linspace(5, 20, 16) for ti in np.linspace(0, 1800, 13)]
    tab = inv.TabulatedModel(pd.DataFrame(rows), parameters=["h"], inputs=["t_s"], output="T")
    obs = inv.Observations({"t_s": t}, 0.2, values=cool.predict(th, {"t_s": t}))
    r = inv.calibrate(tab, obs, [inv.Parameter("h", 8.0, 5.0, 20.0)], bootstrap=0)
    assert r.theta["h"] == pytest.approx(11.0, rel=0.02)


def test_observations_from_uploaded_csv():
    from pinneapple_data.data_health import load_table

    csv = b"time_s;temp_C;sensor\n0;94,1;TC1\n60;90,0;TC1\n120;x;TC1\n180;83,2;TC1\n"
    df, _ = load_table(csv, "log.csv")
    obs = inv.Observations.from_frame(df, inputs=["time_s"], value="temp_C", sigma=0.3)
    assert obs.n == 3 and obs.meta["dropped_rows"] == 1
    assert obs.values[1] == pytest.approx(90.0)


def test_model_from_spec_validates_options():
    assert isinstance(inv.model_from_spec({"type": "rod_fv", "options": {"n_zones": 3}}), inv.RodModel)
    with pytest.raises(ValueError):
        inv.model_from_spec({"type": "pin_fin", "options": {"bogus": 1}})
    with pytest.raises(ValueError):
        inv.model_from_spec({"type": "nope"})
    assert {p["id"] for p in inv.list_presets()} >= {"fin_h", "plate_h", "rod_h_field"}


def test_contracts_round_trip():
    contracts = pytest.importorskip("pinneapple_data.contracts")
    if not hasattr(contracts, "ParameterSpace"):
        pytest.skip("pinneapple_data.contracts is not on this branch")
    from pinneapple_analysis.inverse_problems import contracts as ic

    b = inv.get_preset("fin_h_k_flux").build()
    space = ic.to_parameter_space(b["parameters"], "fin", groups={"h": "boundary", "k": "material"})
    back = ic.parameters_from_space(contracts.ParameterSpace.from_json(space.to_json()))
    assert [p.name for p in back if not p.fixed] == ["h", "k"]
    rep = inv.analyze_identifiability(b["model"], b["observations"], b["parameters"], posterior=False)
    st = ic.identifiability_study(rep.to_dict(), b["parameters"], model="pin_fin")
    assert st.study_type == "identifiability" and st.passed
    cal = inv.calibrate(b["model"], b["observations"], b["parameters"], holdout=b["validation"], bootstrap=0)
    sc = ic.calibration_study(cal.to_dict(), b["parameters"], model="pin_fin")
    assert sc.models and sc.models[0].metrics_split == "holdout"
    assert contracts.from_dict(json.loads(sc.to_json())).id == sc.id
    out = inv.compare_methods(inv.get_preset("rod_base_bc").problem(), ["least_squares", "adjoint"])
    si = ic.inverse_study(out, inv.get_preset("rod_base_bc").build()["parameters"])
    assert si.passed and len(si.table.columns["method"]) == 4
    bad = ic.identifiability_study(inv.analyze_identifiability(
        inv.get_preset("fin_h_k").build()["model"], inv.get_preset("fin_h_k").build()["observations"],
        inv.get_preset("fin_h_k").build()["parameters"], profile=False, posterior=False).to_dict(),
        inv.get_preset("fin_h_k").build()["parameters"])
    assert not bad.passed
