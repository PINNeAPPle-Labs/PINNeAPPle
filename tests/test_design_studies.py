"""Design studies: problem description, methods under one budget, verification, active learning, presets."""
from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from pinneapple_design.design_optimizer import (
    CommandEvaluator,
    Constraint,
    DesignProblem,
    DesignSpace,
    FunctionEvaluator,
    Objective,
    Variable,
    available_methods,
    compare_methods,
    feasibility_report,
    from_model_ref,
    hypervolume,
    presets,
    run,
    select_candidates,
    surrogate_assisted,
    verify_candidates,
)


# --------------------------------------------------------------------------- an analytic bi-objective problem
def _fn(X):
    x, y = X[:, 0], X[:, 1]
    return {"f1": x ** 2 + y ** 2, "f2": (x - 2) ** 2 + y ** 2, "g": x + y}


def _torch_fn(X):
    x, y = X[:, 0], X[:, 1]
    return {"f1": x ** 2 + y ** 2, "f2": (x - 2) ** 2 + y ** 2, "g": x + y}


def schaffer(differentiable=True, fidelity="model"):
    """Unconstrained Pareto set: y = 0, 0 <= x <= 2. With x + y <= 1.5 it is y = 0 for 0 <= x <= 1.5, then the
    constraint boundary x = 1.5 - y down to y = -1/4 (where d f2 / dy = 0 along it)."""
    space = DesignSpace([Variable("x", -1.0, 3.0), Variable("y", -1.0, 1.0)])
    ev = FunctionEvaluator(_fn, ["f1", "f2", "g"], torch_fn=_torch_fn if differentiable else None, fidelity=fidelity)
    return DesignProblem("schaffer", space, [Objective("f1"), Objective("f2")], [Constraint("g", upper=1.5)], ev)


def _true_set(n=20001):
    a = np.linspace(0, 1.5, n)
    b = np.linspace(-0.25, 0, n)
    return np.r_[np.c_[a, 0 * a], np.c_[1.5 - b, b]]


def _exact_hv(ref=(5.0, 5.0)):
    # the hypervolume of a dense sampling of the true Pareto set (the staircase error is O(1/n))
    P = _true_set()
    return hypervolume(np.c_[P[:, 0] ** 2 + P[:, 1] ** 2, (P[:, 0] - 2) ** 2 + P[:, 1] ** 2], np.array(ref))


def _distance_to_true_set(X):
    P = _true_set(2001)
    return np.sqrt(((X[:, None, :] - P[None]) ** 2).sum(-1)).min(1)


def test_space_repair_sample_and_parameter_space():
    s = DesignSpace([Variable("a", 0, 1), Variable("n", 2, 9, kind="integer")])
    X = s.repair([[1.7, 3.4], [-1, 8.6]])
    assert X.tolist() == [[1.0, 3.0], [0.0, 9.0]]
    S = s.sample(32, seed=1)
    assert S.shape == (32, 2) and np.all(S[:, 1] == np.round(S[:, 1])) and S[:, 0].min() >= 0
    ps = {"parameters": [{"name": "a", "low": 0, "high": 2, "unit": "m"}, {"name": "n", "bounds": [1, 4],
                                                                          "type": "int"}]}
    s2 = DesignSpace.from_parameter_space(ps)
    assert s2.names == ["a", "n"] and s2.integer.tolist() == [False, True] and s2.variables[0].unit == "m"
    assert DesignSpace.from_dict(s2.to_dict()).bounds.tolist() == [[0, 2], [1, 4]]
    with pytest.raises(ValueError):
        Variable("bad", 1, 1)


def test_constraint_violation_margin_slack():
    c = Constraint("t", upper=80.0)
    assert c.violation(np.array([70.0, 88.0])).tolist() == [0.0, 0.1]
    assert c.margin(np.array([70.0])).tolist() == [10.0]
    assert np.isclose(c.slack(70.0), 0.125)
    lo = Constraint("gap", lower=1.5)
    assert np.isclose(lo.violation(np.array([1.2]))[0], 0.2)
    assert np.isinf(c.violation(np.array([np.nan]))[0])


def test_hypervolume_exact_and_monte_carlo():
    F = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.5, 3.5]])
    assert hypervolume(F, np.array([4.0, 4.0])) == pytest.approx(3 + 2 + 1)
    cube = hypervolume(np.array([[0.5, 0.5, 0.5]]), np.ones(3), n_mc=40000)
    assert cube == pytest.approx(0.125, rel=1e-9)          # one point: the MC box is exactly its dominated box
    two = hypervolume(np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]), np.ones(3), n_mc=40000)
    assert two == pytest.approx(0.25 + 0.25 - 0.125, rel=0.03)


@pytest.mark.parametrize("method", ["random", "nsga2", "scipy-de", "scipy-slsqp", "gradient", "pymoo-nsga2"])
def test_methods_respect_budget_and_find_the_front(method):
    p = schaffer()
    r = run(p, method, budget=240, seed=0)
    if r.status == "skipped":
        pytest.skip(r.message)
    assert r.status == "ok", r.message
    assert r.n_evals <= 240
    ev = r.evaluations
    front = r.pareto()
    assert len(front) >= 1
    assert np.all(ev.feasible[front]) and np.all(ev.Y["g"][front] <= 1.5 + 1e-12)
    if method != "random":
        # the front found within 240 evaluations lies near the true Pareto set
        d = _distance_to_true_set(ev.X[front])
        assert np.median(d) < 0.09 and d.max() < 0.4


def test_nsga2_hypervolume_close_to_exact():
    p = schaffer()
    r = run(p, "nsga2", budget=1200, seed=0)
    hv = hypervolume(r.evaluations.F[r.pareto()], np.array([5.0, 5.0]))
    assert hv == pytest.approx(_exact_hv(), rel=0.01)
    assert hv <= _exact_hv() * (1 + 1e-9)                    # nothing beats the true front


def test_bayesian_batch_acquisition_matches_pointwise():
    from pinneapple_design.design_optimizer.optimizer import (
        BayesianDesignOptimizer,
        DesignOptimizerConfig,
    )
    for acq in ("ei", "ucb"):
        o = BayesianDesignOptimizer(DesignOptimizerConfig(n_initial_random=4, acquisition=acq), seed=0)
        rng = np.random.default_rng(1)
        for _ in range(10):
            x = rng.random(3)
            o.update(x, [float((x ** 2).sum())])
        o.y_obs = [float(y[0]) for y in o.Y_obs]
        o._fit_gp()
        X = rng.random((40, 3))
        assert np.allclose(o._acquisition_batch(X), [o._acquisition(x) for x in X], atol=1e-12)


def test_bayesian_runs_within_budget():
    r = run(schaffer(), "bayesian", budget=24, seed=0)
    assert r.status == "ok" and r.n_evals == 24 and len(r.pareto()) >= 1


def test_gradient_skipped_without_autograd_and_relaxed_integers():
    r = run(schaffer(differentiable=False), "gradient", budget=50)
    assert r.status == "skipped" and "not differentiable" in r.message
    assert not available_methods(schaffer(differentiable=False))["gradient"]["available"]
    # integer variable on a relaxable differentiable evaluator: iterates are relaxed, end points are integers
    space = DesignSpace([Variable("x", 0.0, 4.0), Variable("n", 1, 6, kind="integer")])
    f = lambda X: {"f": (X[:, 0] - 1.3) ** 2 + (X[:, 1] - 2.6) ** 2}  # noqa: E731
    ev = FunctionEvaluator(f, ["f"], torch_fn=f, relaxable=True)
    p = DesignProblem("int", space, [Objective("f")], [], ev)
    r = run(p, "gradient", budget=120, seed=0)
    e = r.evaluations
    assert e.relaxed.any()
    best = r.best()
    assert best is not None and e.X[best, 1] == 3.0 and not e.relaxed[best]


def test_initial_designs_are_evaluated_first():
    p = schaffer()
    r = run(p, "nsga2", budget=60, initial=np.array([[1.0, 0.0]]))
    assert r.evaluations.X[0].tolist() == [1.0, 0.0] and r.evaluations.tag[0] == "initial designs"


def test_feasibility_report_active_and_violated():
    p = schaffer()
    ev = p.evaluate(np.array([[1.49, 0.0], [0.5, 0.0], [2.0, 0.0]]))
    rep = feasibility_report(p, ev, [0, 1, 2])
    assert rep[0]["active"] == ["g"] and rep[0]["feasible"]
    assert rep[1]["active"] == [] and rep[1]["constraints"][0]["status"] == "inactive"
    assert rep[1]["constraints"][0]["margin"] == pytest.approx(1.0)
    assert rep[2]["violated"] == ["g"] and not rep[2]["feasible"]


def test_verification_statuses_and_errors():
    p = schaffer(fidelity="surrogate")
    r = run(p, "nsga2", budget=300, seed=0)
    chosen = select_candidates(r.evaluations, r.pareto(), 4)
    # a reference whose g is 10 % larger: designs with g in (1.364, 1.5] are rejected by it
    ref = FunctionEvaluator(lambda X: {**_fn(X), "f1": _fn(X)["f1"] * 1.02, "g": _fn(X)["g"] * 1.1},
                            ["f1", "f2", "g"], fidelity="reference")
    rep = verify_candidates(p, r.evaluations, chosen, ref)
    assert {x["status"] for x in rep.rows} <= {"verified", "rejected by reference"}
    assert rep.errors["f1"]["max_rel"] == pytest.approx(0.02 / 1.02, rel=1e-6)
    for x in rep.rows:
        g_ref = x["reference"]["g"]
        assert (x["status"] == "verified") == (g_ref <= 1.5)
    bad = FunctionEvaluator(lambda X: (_ for _ in ()).throw(RuntimeError("solver crashed")), ["f1", "f2", "g"],
                            fidelity="reference")
    rep2 = verify_candidates(p, r.evaluations, chosen, bad)
    assert all(x["status"] == "reference failed" and "solver crashed" in x["message"] for x in rep2.rows)


def test_surrogate_assisted_front_is_reference_verified():
    pytest.importorskip("sklearn")
    p = schaffer(fidelity="surrogate")
    ref = FunctionEvaluator(_fn, ["f1", "f2", "g"], fidelity="reference")
    al = surrogate_assisted(p, ref, n_initial=12, rounds=2, n_verify=4, surrogate_budget=300)
    assert al.n_reference <= 12 + 2 * 4
    assert all(f == "reference" for f in al.reference_evals.fidelity)
    front = al.final.pareto()
    assert len(front) >= 2
    assert np.all(al.reference_evals.Y["g"][front] <= 1.5)
    # the GP prediction at designs it had not seen is within 1 of the reference (f spans 0-9 on the domain); the
    # pessimistic constraint keeps every verified design feasible
    abs_err = [abs(r["surrogate"][k] - r["reference"][k]) for r in al.verification.rows for k in ("f1", "f2")]
    assert max(abs_err) < 1.0
    assert all(r["status"] == "verified" for r in al.verification.rows)
    assert set(al.rounds[-1]["held_out_error"]) >= {"f1", "f2", "g"}


def test_compare_methods_same_budget():
    c = compare_methods(schaffer(), ["random", "nsga2", "gradient"], budget=150, seeds=(0, 1))
    t = {r["method"]: r for r in c.table()}
    assert set(t) == {"random", "nsga2", "gradient"}
    assert all(r["n_evals"] <= 150 for r in c.rows)
    assert t["nsga2"]["hv_mean"] > t["random"]["hv_mean"]
    assert c.winner() in t
    assert all(0 < r["hv_fraction_of_union"] <= 1 + 1e-9 for r in c.rows)
    assert c.to_dict()["curves"]
    ev, labels = c.union()
    assert len(ev) == len(labels) == sum(r.n_evals for r in c.results)
    front = ev.pareto()
    assert len(front) and {labels[i].split("#")[0] for i in front} <= {"random", "nsga2", "gradient"}


def test_command_evaluator_runs_an_external_program(tmp_path):
    script = tmp_path / "solver.py"
    script.write_text(textwrap.dedent("""
        import csv, sys
        rows = list(csv.DictReader(open(sys.argv[1])))
        with open(sys.argv[2], "w", newline="") as f:
            w = csv.writer(f); w.writerow(["f1", "f2", "g"])
            for r in rows:
                x, y = float(r["x"]), float(r["y"])
                w.writerow([x * x + y * y, (x - 2) ** 2 + y * y, x + y])
    """))
    space = schaffer().space
    ev = CommandEvaluator([sys.executable, str(script), "{inputs}", "{outputs}"], space, ["f1", "f2", "g"])
    X = np.array([[1.0, 0.5], [0.0, 0.0]])
    out = ev(X)
    for k, v in _fn(X).items():
        assert np.allclose(out[k], v)
    ref = from_model_ref({"uri": f"cmd:{sys.executable} {script} {{inputs}} {{outputs}}",
                          "outputs": ["f1", "f2", "g"]}, space)
    assert np.allclose(ref(X)["g"], [1.5, 0.0]) and ref.fidelity == "reference"


# --------------------------------------------------------------------------- presets built from library physics
def test_heatsink_geometry_matches_library_closed_forms():
    from pinneapple_design.thermal.heatsink.engine import DesignInput
    from pinneapple_physics.closed_form.cantilever_fatigue import stress_from_moment_pa

    p = presets.heatsink_problem("closed_form")
    b = presets.heatsink_baseline()
    ev = p.evaluate(p.space.from_dicts([b]))
    d = DesignInput(**b)
    assert ev.Y["mass_g"][0] == pytest.approx(d.geometry().mass_kg() * 1e3, rel=1e-12)
    op = p.evaluator.op
    H, t, D = b["fin_height_mm"] * 1e-3, b["fin_thickness_mm"] * 1e-3, b["base_depth_mm"] * 1e-3
    moment = 2700.0 * t * D * op.shock_g * 9.80665 * H ** 2 / 2
    assert ev.Y["fin_stress_mpa"][0] == pytest.approx(stress_from_moment_pa(D, t, moment) * 1e-6, rel=1e-12)
    assert ev.Y["fin_gap_mm"][0] == pytest.approx(d.geometry().fin_gap * 1e3)


def test_heatsink_fidelities_agree_and_torch_matches_numpy():
    import torch
    b = presets.heatsink_baseline()
    vals = {}
    for f in presets.HEATSINK_FIDELITIES:
        p = presets.heatsink_problem(f)
        vals[f] = p.evaluate(p.space.from_dicts([b]))
        assert vals[f].feasible[0]
    t_ref = vals["fvm"].Y["t_source_c"][0]
    # surrogate card: R error p95 about 1 %; the closed form uses the circular-equivalent spreading model
    assert vals["surrogate"].Y["t_source_c"][0] == pytest.approx(t_ref, rel=0.02)
    assert vals["closed_form"].Y["t_source_c"][0] == pytest.approx(t_ref, rel=0.05)
    sur = presets.heatsink_problem("surrogate").evaluator
    X = presets.heatsink_problem().space.sample(16, seed=3)
    out_t = sur.torch(torch.tensor(X))
    out_n = sur(X)
    ok = out_n["fin_gap_mm"] > 0                             # fins that do not fit are not sent to the model
    assert ok.sum() >= 8
    for k in ("t_source_c", "pressure_drop_pa", "mass_g", "cost", "fin_stress_mpa"):
        assert np.allclose(out_t[k].detach().numpy()[ok], out_n[k][ok], rtol=1e-5), k
    assert np.array_equal(out_t["_valid"].numpy(), out_n["_valid"])


def test_heatsink_optimum_beats_baseline_and_is_confirmed_by_fvm():
    p = presets.heatsink_problem("surrogate")
    x0 = p.space.from_dicts([presets.heatsink_baseline()])
    r = run(p, "nsga2", budget=600, seed=0, initial=x0)
    best = r.best(0)
    assert r.evaluations.Y["mass_g"][best] < 0.5 * r.evaluations.Y["mass_g"][0]
    rep = verify_candidates(p, r.evaluations, [best], presets.HeatsinkEvaluator("fvm"))
    row = rep.rows[0]
    assert row["status"] in ("verified", "rejected by reference")
    assert row["rel_error"]["t_source_c"] < 0.03


def test_airfoil_preset_gradient_and_exports():
    import torch
    p = presets.airfoil_problem()
    x0 = p.space.from_dicts([presets.airfoil_baseline()])
    ev = p.evaluate(x0)
    assert ev.feasible[0] and 0.005 < ev.Y["Cd"][0] < 0.02
    X = torch.tensor(x0, requires_grad=True)
    p.evaluator.torch(X)["Cd"].sum().backward()
    assert torch.isfinite(X.grad).all() and X.grad.abs().sum() > 0
    out_n = p.evaluator(x0)
    out_t = p.evaluator.torch(torch.tensor(x0))
    for k in ("Cd", "Cl", "spar_depth", "aft_thickness", "thickness"):
        assert np.allclose(out_t[k].detach().numpy(), out_n[k], rtol=1e-5), k
    stl = presets.design_file(presets.airfoil_bodies(presets.airfoil_baseline()), "stl", "naca2412")
    n_tri = int.from_bytes(stl[80:84], "little")
    assert len(stl) == 84 + 50 * n_tri and n_tri == 4 * (2 * 81 - 1)
    bodies = presets.plate_fin_bodies(presets.heatsink_baseline())
    assert len(bodies) == 21
    step = presets.design_file(bodies, "step", "heatsink").decode()
    assert step.startswith("ISO-10303-21;") and step.count("FACETED_BREP(") == 21


def test_zoo_model_reference_evaluator():
    space = DesignSpace([Variable("alpha_deg", 0.0, 8.0)])
    fixed = dict(zip(("T0", "T1", "T2", "M0", "M1", "M2"), presets.airfoil_baseline().values(), strict=False))
    ev = from_model_ref({"uri": "zoo:airfoil_cst_rans_mlp"}, space, fixed=fixed)
    out = ev(np.array([[4.0]]))
    assert out["Cl"][0] == pytest.approx(0.66, abs=0.02) and out["_valid"][0]


# --------------------------------------------------------------------------- specs, contracts, data-trained models
def test_problem_from_spec_heatsink_and_errors():
    from pinneapple_design.design_optimizer import SpecError, problem_from_spec, reference_for_spec
    p, fit = problem_from_spec({"preset": "heatsink", "operating": {"power_w": 80}, "limits": {"t_source_max_c": 90},
                                "variables": [{"name": "base_width_mm", "upper": 120}],
                                "objectives": [{"name": "mass_g"}, {"name": "pressure_drop_pa"}]})
    assert fit is None and p.space.bounds[0].tolist() == [40.0, 120.0] and p.evaluator.op.power_w == 80
    assert p.constraints[0].upper == 90 and [o.name for o in p.objectives] == ["mass_g", "pressure_drop_pa"]
    assert reference_for_spec({"preset": "heatsink"}).fidelity == "reference"
    assert reference_for_spec({"source": "zoo", "model": "x"}) is None
    for bad in ({"preset": "nope"}, {"preset": "heatsink", "limits": {"foo": 1}},
                {"preset": "heatsink", "objectives": [{"name": "volume"}]},
                {"preset": "heatsink", "variables": [{"name": "n_fins", "lower": 9, "upper": 5}]},
                {"source": "zoo", "model": "not-a-model", "objectives": [{"name": "a"}]}):
        with pytest.raises(SpecError):
            problem_from_spec(bad)


def test_zoo_spec_bounds_from_validity_range():
    from pinneapple_design.design_optimizer import problem_from_spec
    p, _ = problem_from_spec({"source": "zoo", "model": "airfoil_cst_rans_mlp", "fixed": {"alpha_deg": 4},
                              "objectives": [{"name": "Cd"}], "constraints": [{"name": "Cl", "lower": 0.6}]})
    assert p.space.names == ["T0", "T1", "T2", "M0", "M1", "M2"]
    assert p.space.bounds[0].tolist() == [0.1, 0.24]
    r = run(p, "nsga2", budget=200)
    best = r.best()
    assert r.evaluations.Y["Cl"][best] >= 0.6


def test_parameter_space_contract_round_trip_and_study_result():
    pytest.importorskip("pinneapple_data.contracts.models")  # the shared data contracts (apps runtime)
    from pinneapple_data.contracts import ParameterSpace, StudyResult
    from pinneapple_design.design_optimizer import study_result, to_parameter_space
    from pinneapple_design.design_optimizer.study import constants_of
    p = presets.heatsink_problem("surrogate")
    ps = to_parameter_space(p.space, "heatsink")
    assert isinstance(ps, ParameterSpace) and ps.parameters[3].kind == "integer"
    back = DesignSpace.from_parameter_space(ps)
    assert back.names == p.space.names and back.integer.tolist() == p.space.integer.tolist()
    with_const = ps.to_dict()
    with_const["parameters"].append({"name": "power", "kind": "constant", "value": 60})
    assert DesignSpace.from_parameter_space(with_const).names == p.space.names
    assert constants_of(with_const) == {"power": 60}
    r = run(p, "nsga2", budget=200, seed=1)
    cand = select_candidates(r.evaluations, r.pareto(), 3)
    sr = study_result(p, r, study_id="s1", app="test", candidates=cand)
    names = {c.name: c.status for c in sr.checks}
    assert names["high_fidelity_verification"] == "NOT_RUN"          # nothing ran: never reported as passed
    rep = verify_candidates(p, r.evaluations, cand, presets.HeatsinkEvaluator("fvm"))
    sr2 = study_result(p, r, study_id="s2", app="test", candidates=cand, verification=rep)
    hf = next(c for c in sr2.checks if c.name == "high_fidelity_verification")
    assert hf.status in ("PASS", "FAIL") and hf.value.endswith(f"/{len(cand)}")
    assert len(sr2.table.columns["verification"]) == len(cand)
    assert isinstance(StudyResult.from_json(sr2.to_json()), StudyResult)


def test_surrogate_from_table_held_out_and_table_reference():
    import pandas as pd

    from pinneapple_design.design_optimizer import (
        model_ref_for_surrogate,
        surrogate_from_table,
        table_evaluator,
    )
    p = schaffer()
    X = p.space.sample(60, seed=4)
    Y = _fn(X)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], **Y})
    fit = surrogate_from_table(df, ["x", "y"], ["f1", "f2", "g"], seed=0)
    assert fit.n_test == 12 and fit.n_train == 48
    assert fit.metrics["g"]["r2"] > 0.999 and fit.metrics["f1"]["r2"] > 0.99       # smooth: a GP nails them
    assert fit.space.bounds[0, 0] == pytest.approx(X[:, 0].min())
    try:
        import pinneapple_data.contracts.models  # noqa: F401
        ref = model_ref_for_surrogate(fit, "gp", "app:test/1")
        assert ref.metrics_split == "test" and ref.inputs[0].low == pytest.approx(X[:, 0].min())
    except ImportError:
        pass                                                   # contracts not installed: ModelRef not checked here
    tab = table_evaluator(fit.space, df, ["f1"])
    out = tab(np.vstack([X[:2], [[0.123, 0.456]]]))
    assert np.allclose(out["f1"][:2], Y["f1"][:2]) and np.isnan(out["f1"][2]) and out["_valid"].tolist() == [1, 1, 0]
    with pytest.raises(ValueError):
        surrogate_from_table(df.head(5), ["x", "y"], ["f1"])


def test_progress_callback_and_cancel():
    p = schaffer()
    seen = []
    run(p, "nsga2", budget=100, progress=lambda u, b: seen.append((u, b)))
    assert seen[-1] == (100, 100) and all(b == 100 for _, b in seen)

    class Cancelled(Exception):
        pass

    def stop(u, b):
        if u >= 40:
            raise Cancelled
    with pytest.raises(Cancelled):
        run(p, "nsga2", budget=100, progress=stop)


def test_verify_front_files_and_chart_points(tmp_path):
    pytest.importorskip("pinneapple_data.contracts.models")
    import pandas as pd

    from pinneapple_design.design_optimizer import (
        DESIGN_FILES,
        chart_points,
        verify_front,
        write_design_files,
    )
    p = schaffer(fidelity="surrogate")
    r = run(p, "nsga2", budget=300, seed=0)
    ev, front = r.evaluations, r.pareto()
    cand, rep, status = verify_front(p, ev, front, 4, None)
    assert cand == select_candidates(ev, front, 4) and rep is None and status == {}
    ref = FunctionEvaluator(_fn, ["f1", "f2", "g"], fidelity="reference")
    cand, rep, status = verify_front(p, ev, front, 4, ref)
    assert sorted(status) == sorted(cand) and set(status.values()) == {"verified"}
    assert rep.errors["f1"]["max_rel"] == pytest.approx(0.0, abs=1e-12)     # same function: no error
    labels = [f"m{i % 2}#0" for i in range(len(ev))]
    files = write_design_files(tmp_path, p, ev, cand, status, labels=labels)
    assert files == list(DESIGN_FILES) and all((tmp_path / f).exists() for f in files)
    cdf = pd.read_csv(tmp_path / "candidates.csv")
    assert cdf["verification"].tolist() == ["verified"] * len(cand)
    assert cdf["found_by"].tolist() == [labels[i] for i in cand]
    assert pd.read_csv(tmp_path / "candidates_to_run.csv").columns.tolist() == ["x", "y"]
    assert len(pd.read_csv(tmp_path / "evaluations.csv")) == len(ev)
    pts = chart_points(p, ev, front, limit=50, labels=labels)
    assert len(pts) <= 50 + len(front) and {q["i"] for q in pts if q["p"]} == {int(i) for i in front}
    assert set(pts[0]) >= {"i", "f", "p", "r", "f1", "f2", "g", "run"} and pts[0]["run"] in ("m0", "m1")
    assert len(chart_points(p, ev)) == len(ev)


def test_verify_uploaded_results_matches_rows_and_checks_constraints(tmp_path):
    pytest.importorskip("pinneapple_data.contracts.models")
    import pandas as pd

    from pinneapple_design.design_optimizer import (
        verify_front,
        verify_uploaded_results,
        write_design_files,
    )
    p = schaffer(fidelity="surrogate")
    r = run(p, "nsga2", budget=300, seed=0)
    cand, _, _ = verify_front(p, r.evaluations, r.pareto(), 4)
    write_design_files(tmp_path, p, r.evaluations, cand)
    cdf = pd.read_csv(tmp_path / "candidates.csv")
    X = pd.read_csv(tmp_path / "candidates_to_run.csv")
    # your solver: g 10 % larger than the search model, so designs near the g <= 1.5 limit fail there
    g = _fn(X.to_numpy())["g"] * 1.1
    res = X.assign(g=g)
    rep = verify_uploaded_results(p.space, cdf, res, p.objectives, p.constraints)
    assert [x["status"] for x in rep.rows] == ["verified" if v <= 1.5 else "rejected by reference" for v in g]
    assert rep.errors["g"]["max_rel"] == pytest.approx(0.1 / 1.1, rel=1e-6)
    missing = verify_uploaded_results(p.space, cdf, res.iloc[1:], p.objectives, p.constraints)
    assert missing.rows[0]["status"] == "reference failed"                  # no row for that design
    with pytest.raises(ValueError):
        verify_uploaded_results(p.space, cdf, X.assign(other=1.0), p.objectives, p.constraints)


def test_comparison_union_result_and_fronts():
    c = compare_methods(schaffer(), ["random", "nsga2"], budget=120, seeds=(0, 1))
    u = c.union_result()
    assert u.method == "comparison: random, nsga2" and u.budget == 120 * 4
    assert u.n_evals == sum(r.n_evals for r in c.results) and u.seed == 0
    fr = c.fronts()
    assert set(fr) == {"random", "nsga2"}
    r0 = next(r for r in c.results if r.method == "nsga2" and r.seed == 0)
    assert fr["nsga2"] == [[float(r0.evaluations.Y["f1"][i]), float(r0.evaluations.Y["f2"][i])] for i in r0.pareto()]
    assert c.fronts(seed=1)["random"] != fr["random"]


def test_preset_info_and_bodies():
    hs, af = presets.preset_info("heatsink"), presets.preset_info("airfoil")
    assert [v["name"] for v in hs["variables"]] == presets.heatsink_problem().space.names
    assert hs["limits"]["fin_gap_min_mm"] > 0 and hs["objectives"][:2] == ["mass_g", "cost"]
    assert af["limits"] == {"cl_min": 0.6, "cm_min": -0.12, "aft_min": 0.012} and af["reference"] is None
    assert [v["name"] for v in af["variables"]] == presets.airfoil_problem().space.names
    with pytest.raises(ValueError):
        presets.preset_info("wing")
    assert presets.preset_bodies("heatsink", presets.heatsink_baseline())
    assert presets.preset_bodies("airfoil", presets.airfoil_baseline())
