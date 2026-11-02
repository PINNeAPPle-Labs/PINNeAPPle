"""Core coverage (#1129, toward 1.0) for the validation arena's building blocks and the fault-injection library,
each called directly at CI size (tests/test_validation_arena.py runs the whole loop).

The search methods are checked on a quadratic with a known minimum; the conformal calibration against its false
alarm guarantee on held-out clean cases; faults and generators against the definitions they state.
"""
import json
import math
import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation.arena import blue, cases, generators, red, search
from pinneapple_analysis.validation.fault_injection import checks as CK
from pinneapple_analysis.validation.fault_injection import faults as FT
from pinneapple_analysis.validation.fault_injection import get_problem, run_fault_injection
from pinneapple_analysis.validation.fault_injection import problems as PB


@pytest.fixture(scope="module")
def burgers():
    return get_problem("burgers_1d")


@pytest.fixture(scope="module", autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# ------------------------------------------------------------------------------------------------ search
@pytest.mark.parametrize("fn", [search.differential_evolution, search.cmaes, search.lbfgs])
def test_search_methods_find_the_minimum_of_a_quadratic(fn):
    target = np.array([0.3, 0.7, 0.55])
    r = fn(lambda x: float(np.sum((x - target) ** 2)), 3, 600, seed=1)
    assert isinstance(r, search.SearchResult) and r.evaluations <= 600
    assert r.fun < 1e-3 and np.allclose(r.x, target, atol=0.05)                  # exact minimum: 0 at target
    assert r.fun == pytest.approx(float(np.sum((r.x - target) ** 2)))


def test_minimize_dispatches_and_rejects_unknown_methods():
    r = search.minimize(lambda x: float((x[0] - 0.25) ** 2), 1, 200, method="cmaes")
    assert r.method == "cmaes" and abs(r.x[0] - 0.25) < 0.02
    with pytest.raises(ValueError):
        search.minimize(lambda x: 0.0, 1, 10, method="nope")


# ------------------------------------------------------------------------------------------------ problems, faults
def test_case_and_dataset_containers(burgers):
    c = burgers.case(3)
    assert isinstance(c, PB.Case) and c.problem == "burgers_1d"
    d = c.with_field(np.zeros_like(c.field), note="zeroed")
    assert np.all(d.field == 0) and d.meta["note"] == "zeroed" and c.field.any()      # the original is untouched
    ds = PB.Dataset(train=[c, d], test=[c])
    assert ds.cases == [c, d, c]


def test_fault_library_definitions(burgers):
    names = {f.name for f in FT.model_faults(burgers)}
    assert {"bias", "scale", "noise"} <= names and all(f.target == "data" for f in FT.data_faults())
    smoothing = FT.get_fault("smoothing")
    assert isinstance(smoothing, FT.Fault) and smoothing.documented_for("burgers_1d") == 4.0
    assert FT.get_fault("bias").documented_for("darcy_2d") == 0.05
    c = burgers.case(0)
    faulty, ref = FT.get_fault("scale").apply(burgers, c, 0.1, np.random.default_rng(0))
    assert np.allclose(ref, c.field) and np.allclose(faulty.field, 1.1 * c.field)       # scale: u -> (1 + m) u
    assert burgers.true_error(faulty, ref) == pytest.approx(0.1, rel=1e-6)


def test_a_check_flags_a_wrong_case_and_passes_the_right_one(burgers):
    lib = CK.check_library(burgers)
    mass = lib["mass conservation"]
    c = burgers.case(1)
    assert not CK.flags(mass, c)
    assert CK.flags(mass, c.with_field(c.field + 0.3))                               # adds mass
    assert not CK.flags(mass, c.with_field(c.field + 0.3), tolerance=1e9)            # tolerance override


def test_dataset_checks_see_duplicates_and_leakage(burgers):
    train = burgers.sample(6, seed=0)
    good = PB.Dataset(train=train, test=burgers.sample(3, seed=100))
    bad = PB.Dataset(train=train + train[:2], test=[train[0]])
    dc = {c.name: c for c in CK.dataset_checks(burgers)}
    dup, leak = dc["duplicated or non-diverse samples"], dc["train/test leakage"]
    assert dup.fn(None, good) == 0.0 and dup.fn(None, bad) > 0.0                    # dataset checks take (_, ds)
    assert leak.fn(None, good) == 0.0 and leak.fn(None, bad) == pytest.approx(1.0)  # the one test input is a copy


# ------------------------------------------------------------------------------------------------ generators
def test_generators_map_the_unit_box_to_their_bounds(burgers):
    gens = generators.generators_for(burgers)
    assert gens and all("burgers_1d" in g.problems for g in gens)
    g = generators.get_generator("viscosity")
    assert isinstance(g, generators.Generator) and g.dim == 1
    assert g.params(np.array([0.0])) == {"log_factor": -2.0} and g.params(np.array([1.0])) == {"log_factor": 2.0}
    faulty, ref = g.apply(burgers, burgers.case(2), {"log_factor": 0.0})
    assert faulty.meta["generator"] == "viscosity" and burgers.true_error(faulty, ref) < 1e-6   # factor 1: exact
    with pytest.raises(KeyError):
        generators.get_generator("nope")
    assert generators.get_generator("fault:bias").name.endswith("bias")


# ------------------------------------------------------------------------------------------------ blue team
def test_calibration_keeps_its_false_alarm_rate_on_held_out_clean_cases(burgers):
    lib = CK.check_library(burgers)
    chk = [lib["mass conservation"], lib["residual (one-step re-integration)"]]
    cal, held = burgers.sample_varied(40, seed=0), burgers.sample_varied(40, seed=1000)
    m = blue.score_matrix(chk, cal)
    assert set(m) == {c.name for c in chk} and all(len(v) == 40 for v in m.values())
    d = blue.calibrate(chk, cal, alpha=0.2, seed=0)
    assert isinstance(d, blue.Defence) and d.n_calibration > 0
    fa = np.mean([d.rejects(c) for c in held])
    assert fa <= 0.35                                                    # alpha = 0.2, binomial slack at n = 40


def test_harden_adds_checks_until_the_stored_escape_is_rejected(burgers):
    lib = CK.check_library(burgers)
    clean = burgers.sample_varied(30, seed=0)
    c = burgers.case(5)
    escape = c.with_field(c.field + 0.3)                                 # mass is wrong; a residual check misses it
    weak = [lib["periodicity"]]
    assert not blue.calibrate(weak, clean, alpha=0.1).rejects(escape)
    d = blue.harden(weak, lib, clean, escapes=[escape], alpha=0.1)
    assert d.rejects(escape) and len(d.checks) > 1


# ------------------------------------------------------------------------------------------------ red team
def test_red_team_searches_report_consistent_findings(burgers):
    lib = CK.check_library(burgers)
    d = blue.calibrate([lib["mass conservation"]], burgers.sample_varied(30, seed=0), alpha=0.1)
    f = red.evasion_search(burgers, d, generators.get_generator("mutant"), burgers.case(0), budget=40, seed=0)
    assert isinstance(f, red.Finding) and f.kind == "escape" and f.evaluations <= 40
    if f.found:                                                          # wrong and accepted, by definition
        assert f.true_error >= 0.1 and f.margin <= 0
    fas = red.false_alarm_search(burgers, d, seeds=[0, 1], budget=12)
    assert len(fas) == 2 and all(x.kind == "false_alarm" for x in fas)
    assert [x.margin for x in fas] == sorted((x.margin for x in fas), reverse=True)


def test_mutation_score_and_escapes_from_a_detection_matrix(burgers):
    rep = run_fault_injection(burgers, model_faults=[FT.get_fault("bias"), FT.get_fault("scale")], n_cases=4,
                              include_data=False)
    mat = rep.model
    tight = {c: 0.0 for c in mat.tolerances}                             # zero tolerance rejects every mutant
    loose = {c: math.inf for c in mat.tolerances}                        # infinite tolerance accepts every one
    s_tight, s_loose = red.mutation_score(mat, tight), red.mutation_score(mat, loose)
    assert s_tight["mutants"] == s_loose["mutants"] > 0
    assert s_loose["score"] == 0.0 and s_loose["killed"] == 0
    esc = red.mutation_escapes(mat, loose, seed=0, per_fault=1)
    assert {e.generator for e in esc} <= {"fault:bias", "fault:scale"} and all(e.true_error >= 0.1 for e in esc)


# ------------------------------------------------------------------------------------------------ stored cases
def test_load_cases_and_load_defence(tmp_path):
    f = next(p for p in cases.builtin_cases() if p.stem == "burgers_1d")
    doc = cases.load_cases(f)
    assert doc["problem"] == "burgers_1d" and doc["cases"]
    p, d = cases.load_defence(f)
    assert p.name == "burgers_1d" and set(d.thresholds) == set(doc["defence"]["thresholds"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something else"}))
    with pytest.raises(ValueError):
        cases.load_cases(bad)
