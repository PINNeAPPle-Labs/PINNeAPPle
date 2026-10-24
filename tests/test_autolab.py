"""pinneapple_autolab: mission, ledger, surrogates, validated domain, trust triage and the whole cycle on an
analytic simulator (no OpenFOAM)."""
import json
import os

import numpy as np
import pytest

from pinneapple_autolab import (
    AutonomousLab,
    Budget,
    BudgetExhaustedError,
    FieldSurrogate,
    FunctionSimulator,
    Ledger,
    Limit,
    Mission,
    RunRecord,
    UnresolvedLimitError,
    ValidatedDomain,
    assess_candidates,
    choose_next,
    compare_models,
)
from pinneapple_design.design_optimizer.study import Constraint, Objective, Variable

PTS = np.linspace(0, 1, 40)


def toy(p):
    a, b = p["a"], p["b"]
    return {"drag": 1 + (a - 0.3) ** 2 + 0.5 * b ** 2 + 0.2 * a * b,
            "down": 2 + a + 0.8 * b - 0.3 * a ** 2,
            "__fields__": {"cp": np.sin(3 * PTS * (1 + a)) + b * PTS ** 2}}


def toy_sim(**kw):
    return FunctionSimulator(toy, ["drag", "down"], field_names=["cp"], regions={"tail": PTS > 0.7}, **kw)


def toy_mission(runs=12, reserve=1):
    return Mission("toy", "lower drag, keep downforce", [Variable("a", 0, 1), Variable("b", 0, 1)],
                   [Objective("drag", "min")], [Limit("down", lower_frac_of_baseline=1.0)],
                   baseline={"a": 0.5, "b": 0.5}, budget=Budget(runs, reserve),
                   acceptance={"drag": 0.03, "down": 0.03})


def test_relative_limit_needs_the_baseline():
    m = toy_mission()
    with pytest.raises(UnresolvedLimitError):
        m.constraints_absolute()
    cons = m.resolve({"down": 2.8})
    assert cons[0].lower == pytest.approx(2.8) and m.resolved
    again = Mission.from_dict(json.loads(json.dumps(m.to_dict())))
    assert again.constraints_absolute()[0].lower == pytest.approx(2.8)


def test_baseline_must_be_inside_bounds():
    with pytest.raises(ValueError):
        Mission("x", "", [Variable("a", 0, 1)], [Objective("f")], baseline={"a": 2.0})


def test_ledger_budget_and_events(tmp_path):
    seen = []
    led = Ledger(str(tmp_path), max_runs=2, listeners=[seen.append])
    for i in range(2):
        led.charge()
        led.add(RunRecord(led.next_id("doe"), "doe", {"a": i}, True, {"f": float(i)}, cost_s=1.5))
    with pytest.raises(BudgetExhaustedError):
        led.charge()
    led.add(RunRecord("p", "probe", {"a": 9}, True, {"f": 9.0}))
    X, Y = led.arrays(["a"], ["f"])
    assert X.ravel().tolist() == [0, 1]                  # probes never join the training data
    assert led.used_wall_s == pytest.approx(3.0) and len(seen) == 3
    path = led.save()
    assert Ledger.load(path).used_runs == 3
    assert len(open(os.path.join(str(tmp_path), "events.jsonl")).readlines()) == 3


def test_compare_models_scores_held_out_and_picks_the_best():
    rng = np.random.default_rng(0)
    U = rng.random((30, 2))
    y = 1 + (U[:, 0] - 0.3) ** 2 + 0.5 * U[:, 1] ** 2       # exactly quadratic
    board = compare_models(U, {"f": y}, ("gp", "quadratic"), k=5)
    t = board.table["f"]
    assert board.chosen["f"] == "quadratic" and t["quadratic"]["rmse"] < 1e-4   # ridge 1e-6
    assert t["gp"]["n"] == 30 and 0 <= t["gp"]["coverage95"] <= 1
    m, s = board.predict(np.array([[0.3, 0.0]]))["f"]
    assert m[0] == pytest.approx(1.0, abs=1e-4)


def test_field_surrogate_reconstructs_low_rank_fields():
    rng = np.random.default_rng(1)
    U = rng.random((16, 2))
    S = np.stack([np.sin(3 * PTS * (1 + u[0])) + u[1] * PTS ** 2 for u in U])
    fm = FieldSurrogate("gp", energy=0.99999, max_modes=15).fit(U, S)
    p, sd = fm.predict(U[:2])
    assert np.abs(p - S[:2]).max() < 0.02 and sd.shape == p.shape and fm.n_modes >= 2


def test_validated_domain_flags_extrapolation():
    U = np.array([[0.2, 0.2], [0.4, 0.2], [0.2, 0.4], [0.4, 0.4], [0.3, 0.3]])
    dom = ValidatedDomain(U, ["a", "b"], np.array([[0, 10], [0, 10]]))
    inside, far = dom.assess(np.array([[0.3, 0.25], [0.95, 0.3]]))
    assert inside.inside and not far.inside
    assert any("a = 9.5 is above the simulated range 2 to 4" in r for r in far.reasons)


def test_triage_labels_and_next_experiment_rule():
    names = ["a"]
    X = np.array([[0.1], [0.2], [0.9]])
    pred = {"drag": (np.array([1.00, 0.98, 0.80]), np.array([0.005, 0.005, 0.01])),
            "down": (np.array([3.20, 3.02, 3.5]), np.array([0.01, 0.05, 0.01]))}
    dom = ValidatedDomain(np.array([[0.0], [0.1], [0.2], [0.3]]), names)
    dv = dom.assess(X)
    cands = assess_candidates(X, names, pred, [Objective("drag")], [Constraint("down", lower=3.0)], dv, 1.05,
                              {"drag": 0.03, "down": 0.03})
    assert [c.label for c in cands] == ["favourable", "uncertain", "out_of_domain"]
    assert cands[2].report.decisao == "REJECT"           # extrapolation is a critical failed check
    nxt = choose_next(cands, Objective("drag"))
    assert nxt.rule == "best prediction is out of domain" and nxt.candidate.index == 2
    nxt2 = choose_next(cands, Objective("drag"), exclude=[2])
    assert nxt2.candidate.index in (0, 1) and nxt2.reason


def test_lab_refuses_invalid_designs_and_failed_runs_are_not_evidence(tmp_path):
    sim = toy_sim(checker=lambda p: ["too thin"] if p["b"] < 0.05 else [],
                  fail=lambda p: "diverged" if p["a"] > 0.9 else None)
    lab = AutonomousLab(toy_mission(), sim, str(tmp_path), families=("gp",))
    assert lab.simulate({"a": 0.5, "b": 0.01}, "probe") is None and lab.ledger.used_runs == 0
    rec = lab.simulate({"a": 0.95, "b": 0.5}, "doe")
    assert not rec.ok and rec.outputs == {} and "diverged" in rec.message
    assert lab.ledger.arrays(["a", "b"], ["drag"])[0].shape == (0, 2)


def test_full_cycle_on_an_analytic_simulator(tmp_path):
    lab = AutonomousLab(toy_mission(12, 1), toy_sim(), str(tmp_path), families=("gp", "quadratic"),
                        critical_region="tail", field_tol=0.05, nsga_budget=200, n_pool=100)
    s = lab.run(n_doe=6)
    assert s["budget"]["used_runs"] == 12 and lab.ledger.remaining() == 0
    base = s["baseline"]["outputs"]
    best = s["best"]
    assert best is not None and best["outputs"]["down"] >= base["down"] - 1e-12
    assert best["outputs"]["drag"] <= base["drag"]
    assert toy(best["params"])["drag"] == pytest.approx(best["outputs"]["drag"])   # a simulated value
    assert s["decisions"] and all(d["label_after"] in ("verified", "rejected by reference", "reference failed")
                                  for d in s["decisions"])
    assert all(h["n_train"] >= 7 for h in s["held_out"])
    assert s["field_cv"]["per_region"]["tail"]["rmse"] < 0.2
    assert os.path.exists(os.path.join(str(tmp_path), "ledger.json"))
