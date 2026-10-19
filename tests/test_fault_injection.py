"""Fault-injection suite (#697): the detection matrix on Darcy 2-D and Burgers 1-D at CI size.

Acceptance: every fault is caught by at least one check at its documented magnitude, no check raises a false alarm
on the clean reference cases, results are seeded.
"""
import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation.fault_injection import (
    DATA_FAULTS,
    MODEL_FAULTS,
    BurgersProblem,
    DarcyProblem,
    auc,
    check_library,
    default_checks,
    get_fault,
    run_fault_injection,
    score,
)
from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d, darcy_operator

SMALL = {"n_cases": 6, "n_datasets": 3, "dataset_size": (6, 3)}


@pytest.fixture(scope="module", params=["darcy_2d", "burgers_1d"])
def report(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run_fault_injection(request.param, seed=0, **SMALL)


def test_every_fault_caught_at_documented_magnitude(report):
    for mat in (report.model, report.data):
        for f in mat.faults:
            assert mat.caught_at_documented(f), (report.problem, mat.level, f)
    assert report.passed, report.failures()


def test_no_false_alarm_on_clean_reference(report):
    for mat in (report.model, report.data):
        for c in mat.checks:
            assert mat.false_alarm_rate(c) == 0.0, (report.problem, c)


def test_report_renders(report):
    md = report.to_markdown()
    assert "Check power" in md and "Acceptance" in md
    d = report.to_dict()
    assert set(d["model"]["faults"]) == {f.name for f in MODEL_FAULTS}
    assert set(d["data"]["faults"]) == {f.name for f in DATA_FAULTS}
    power = report.check_power()
    assert power["checks"] and all("false_alarm_rate" in v for v in power["checks"].values())


def test_seeded_and_reproducible():
    kw = dict(n_cases=3, include_data=False, model_faults=[get_fault("noise"), get_fault("bias")])
    a, b = run_fault_injection("darcy_2d", seed=3, **kw), run_fault_injection("darcy_2d", seed=3, **kw)
    for k in a.model.scores:
        for c in a.model.checks:
            np.testing.assert_array_equal(a.model.scores[k][c], b.model.scores[k][c])


def test_cases_regenerate_from_seed_and_variation():
    p = DarcyProblem()
    c1, c2 = p.case(11, contrast=1.2, refine=2.5), p.case(11, contrast=1.2, refine=2.5)
    np.testing.assert_array_equal(c1.field, c2.field)
    b = BurgersProblem()
    assert b.case(5).field.shape == (b.options["n_t"], b.options["n"])


def test_darcy_operator_is_the_solver_stencil():
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 5, (12, 12))
    u = darcy_2d(a, 1.0)
    assert np.abs(darcy_operator(a) @ u.ravel() - 1.0).max() < 1e-9
    u_aniso = darcy_2d(a, 1.0, axis_weights=(1.0, 2.0))
    assert np.abs(darcy_operator(a, (1.0, 2.0)) @ u_aniso.ravel() - 1.0).max() < 1e-9
    assert np.abs(u_aniso - u).max() > 1e-3


def test_strong_checks_are_round_off_on_the_solver_output():
    for p in (DarcyProblem(), BurgersProblem()):
        lib = check_library(p)
        c = p.case(1)
        strong = [n for n in lib if "conservative" in n or "one-step" in n]
        assert strong
        for n in strong:
            assert score(lib[n], c) < 1e-10, (p.name, n)
        assert len(default_checks(p)) == 4


def test_auc():
    assert auc([0, 1, 2], [3, 4]) == 1.0
    assert auc([0, 1], [0, 1]) == 0.5
