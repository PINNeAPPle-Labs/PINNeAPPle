"""Validation arena (#707): red team / blue team on Darcy 2-D and Burgers 1-D at CI size, and the replay of the
stored regression cases by ``pinneapple check --arena-cases``.
"""
import json
import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation import arena
from pinneapple_analysis.validation.arena import (
    ArenaBudget,
    Defence,
    builtin_cases,
    calibrate_scores,
    minimize,
    replay,
    run_arena,
)
from pinneapple_analysis.validation.fault_injection import DarcyProblem, check_library
from pinneapple_check.cli import check_main


@pytest.fixture(scope="module", params=["darcy_2d", "burgers_1d"])
def report(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run_arena(request.param, budget="small", alpha=0.1, seed=0)


def test_round0_red_team_finds_an_escape_or_says_none(report):
    r = report.rounds[0]
    md = report.to_markdown()
    if r.worst_escape is None:
        assert "none found within the budget" in md
    else:
        assert r.worst_escape.true_error >= 0.1 and r.worst_escape.margin <= 0
    assert r.worst_false_alarm is not None and "Worst false alarm" in md


def test_after_blue_round_acceptance(report):
    r = report.rounds[-1]
    assert r.stored_escapes_rejected[0] == r.stored_escapes_rejected[1]
    assert r.stored_false_alarms_accepted[0] == r.stored_false_alarms_accepted[1]
    assert r.heldout_false_alarm_after <= report.alpha
    assert report.passed, report.failures()


def test_report_sections(report):
    md = report.to_markdown()
    for s in ("mutation score", "ROC AUC over fault magnitude", "min. detectable magnitude", "Catalogue"):
        assert s in md
    assert report.check_power()["checks"]
    json.dumps(report.to_dict(), default=float)


def test_stored_cases_replay(report, tmp_path):
    path = report.save_cases(tmp_path / f"{report.problem}.json")
    res = replay(path)
    assert res.ok, res.lines()
    assert len(res.rows) == len(report.stored)


def test_seeded_and_reproducible():
    kw = dict(budget=ArenaBudget(n_bases=1, evals=12, fa_seeds=1, fa_evals=8, n_calibration=20, n_holdout=8,
                                 n_mutation_cases=2), alpha=0.2, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a, b = run_arena("darcy_2d", **kw), run_arena("darcy_2d", **kw)
    assert [f.to_dict() for f in a.stored] == [f.to_dict() for f in b.stored]
    assert a.defence.thresholds == b.defence.thresholds


def test_builtin_cases_replay_and_cli(capsys):
    files = builtin_cases()
    assert {f.stem for f in files} >= {"darcy_2d", "burgers_1d"}
    for f in files:
        res = replay(f)
        assert res.ok, res.lines()
        kinds = {r["kind"] for r in res.rows}
        assert "escape" in kinds
    assert check_main(["--arena-cases"]) == 0
    assert "every stored arena case holds" in capsys.readouterr().out


def test_replay_flags_a_regression(tmp_path):
    src = builtin_cases()[0]
    doc = json.loads(src.read_text())
    doc["defence"]["thresholds"] = {k: 1e30 for k in doc["defence"]["thresholds"]}   # a defence that accepts all
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    assert not replay(bad).ok
    assert check_main(["--arena-cases", str(bad)]) == 1


def test_combined_test_controls_family_wise_false_alarms():
    # 6 independent checks, uniform scores: the combined test keeps the family-wise rate near alpha_cal
    rng = np.random.default_rng(0)
    p = DarcyProblem()
    checks = list(check_library(p).values())[:6]
    cal = {c.name: rng.uniform(size=2000) for c in checks}
    d = calibrate_scores(checks, cal, alpha=0.1, safety=1.0, floor=0.0)
    assert d.alpha_guaranteed
    new = rng.uniform(size=(20000, len(checks)))
    rate = np.mean([d.rejects_values(dict(zip(d.names, row, strict=True))) for row in new])
    assert 0.07 < rate < 0.12
    per_check = 1 - (1 - 0.1) ** (1 / len(checks))                 # the per-check rate the combination implies
    assert all(0.3 * per_check < t_rate < 3 * per_check for t_rate in
               (np.mean(new[:, j] > d.thresholds[n]) for j, n in enumerate(d.names)))


def test_defence_margins():
    p = DarcyProblem()
    lib = check_library(p)
    c = [lib["bounds (0 <= u <= max torsion / a_min)"], lib["residual (finite differences)"]]
    d = Defence.from_checks(c)
    m = d.margins({c[0].name: 0.0, c[1].name: 0.3})
    assert m[c[0].name] < 0 and abs(m[c[1].name] - (0.3 / 0.6 - 1)) < 1e-12
    assert d.rejects_values({c[0].name: 0.01, c[1].name: 0.0})


@pytest.mark.parametrize("method", ["de", "cmaes", "lbfgs"])
def test_minimizers(method):
    r = minimize(lambda x: float(((x - 0.3) ** 2).sum()), 3, 300, method, seed=1)
    assert r.fun < 0.05 and r.evaluations <= 300


def test_module_is_callable():
    import pinneapple as pp
    assert callable(arena) and pp.validate.arena is arena


def test_audit_model_with_the_shipped_defence():
    from pinneapple_analysis.validation.arena import audit_model
    from pinneapple_analysis.validation.fault_injection import get_problem
    p = get_problem("darcy_2d")
    good = audit_model("darcy_2d", p.predict_reference, n_cases=4)
    assert good.true_error.max() < 0.1 and not good.escapes
    bad = audit_model("darcy_2d", lambda i, q: 1.3 * p.predict_reference(i, q), n_cases=4)
    assert bad.rejected.all() and not bad.escapes
