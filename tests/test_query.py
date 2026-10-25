"""pinneapple_query and the chip-package methods: physical references, read vs predicted answers, rigorous
intervals that contain the reference, exact superposition, and a planner that takes the cheapest method meeting
the accuracy target."""
import numpy as np
import pytest

from pinneapple_condense import fv
from pinneapple_condense.package import hotspot_power, reference_stack, uniform_package
from pinneapple_condense.query import package_engine
from pinneapple_condense.substructure import reference_observables
from pinneapple_query import parse_ref

ST = reference_stack(n=4)


def test_physical_references():
    r = parse_ref("die@0.25,0.5")
    assert (r.component, r.local, r.kind) == ("die", (0.25, 0.5), "point")
    assert parse_ref("ubump[3,4]").index == (3, 4) and parse_ref("ubump[3,4]").kind == "feature"
    i = parse_ref("interface:die/tim[0:2,1:3]")
    assert i.kind == "interface" and i.span == ((0, 2), (1, 3)) and i.component == "die/tim"
    assert parse_ref("lid:top").kind == "face:top"
    assert str(parse_ref("die@0.25,0.5,1")) == "die@0.25,0.5,1"
    with pytest.raises(ValueError):
        parse_ref("die@1.5,0.2")                                   # local coordinates are fractions


@pytest.fixture(scope="module")
def engine():
    pkg = uniform_package((6, 6), ST, seed=1)
    eng, model = package_engine(pkg, m=8, reference_cells=10 ** 7, calibrate=False)
    return pkg, eng, model


def _truth(pkg, P):
    T, _ = fv.solve_package(pkg, P, method="direct")
    return T, reference_observables(pkg, T)


def test_solved_then_read_with_physical_points(engine):
    pkg, eng, model = engine
    a = eng.query("Tj_max", "region:die", accuracy=1e-9)          # nothing solved: the reference solves
    assert a.kind == "solved" and a.method == "reference"
    b = eng.query("T", "die@0.4,0.6")
    c = eng.query("T", "die@0.4,0.6,0.5")
    assert b.kind == c.kind == "read" and b.method == "cached"
    T, ref = _truth(pkg, pkg.power)
    n = ST.n
    assert b.value == pytest.approx(T[int(0.4 * 6 * n), int(0.6 * 6 * n), ST.active_k], abs=1e-9)
    assert eng.query("q", "interface:die/tim").value == pytest.approx(ref["q_tim"].sum(), rel=1e-9)
    assert eng.query("q", "lid:top").value == pytest.approx(ref["q_top"].sum(), rel=1e-9)


def test_superposition_is_exact_and_rigorous_from_exact_states(engine):
    pkg, eng, model = engine
    P1 = hotspot_power(ST, pkg.shape, pkg.power.sum(), seed=5)
    eng.query("Tj_max", "region:die", conditions={"power": P1}, accuracy=1e-9)
    Pc = 0.6 * pkg.power + 0.4 * P1
    a = eng.query("Tj_max", "region:die", conditions={"power": Pc}, accuracy=1e-6)
    assert a.kind == "predicted" and a.method == "superposition" and a.rigorous
    _, ref = _truth(pkg, Pc)
    assert abs(a.value - ref["Tj"].max()) < 1e-8


def test_bound_contains_truth_and_loose_targets_take_it(engine):
    pkg, eng, model = engine
    Pn = hotspot_power(ST, pkg.shape, 1.5 * pkg.power.sum(), seed=9, hotspots=1)
    T, ref = _truth(pkg, Pn)
    n = ST.n
    rng = np.random.default_rng(0)
    for u, v in rng.uniform(0, 1, (25, 2)):
        a = eng.query("T", f"die@{u},{v}", conditions={"power": Pn}, accuracy=1e6)
        assert a.method in ("bound", "superposition") and a.kind == "predicted"
        if a.method == "bound":
            lo, hi = a.interval
            assert lo <= T[int(u * 6 * n), int(v * 6 * n), ST.active_k] <= hi
    a = eng.query("Tj_max", "region:die", conditions={"power": Pn}, accuracy=1e6)
    if a.method == "bound":
        assert a.interval[0] <= ref["Tj"].max() <= a.interval[1]
    assert "Pn" not in model.states                                  # nothing was solved for it


def test_tight_target_solves_once_then_reads(engine):
    pkg, eng, model = engine
    Pn = hotspot_power(ST, pkg.shape, pkg.power.sum(), seed=21, hotspots=2)
    n_states = len(model.states)
    a = eng.query("T", "die@0.5,0.5", conditions={"power": Pn}, accuracy=1e-6)
    assert a.kind == "solved" and len(model.states) == n_states + 1
    b = eng.query("T", "die@0.1,0.9", conditions={"power": Pn}, accuracy=1e-6)
    assert b.kind == "read"


def test_condensed_method_on_a_package_too_big_for_the_reference():
    pkg = uniform_package((6, 6), ST, seed=2)
    eng, model = package_engine(pkg, m=8, reference_cells=0)        # no fine solve allowed
    a = eng.query("Tj_max", "region:die")
    assert a.method == "cached" and a.kind == "read"                # the calibration solve
    T, ref = _truth(pkg, pkg.power)
    assert abs(a.value - ref["Tj"].max()) < max(5 * a.uncertainty, 0.05)
    c = eng.query("T", "die@0.3,0.3,0.25")                          # interior: the block rebuilt on demand
    n = ST.n
    die = ST.layer_slices()["die"]
    k = die.start + int(0.25 * (die.stop - die.start))
    assert abs(c.value - T[int(0.3 * 6 * n), int(0.3 * 6 * n), k]) < 0.05


def test_heat_from_condensed_states_carries_their_uncertainty():
    # a heat answer combined from condensed states is not exact: it says so, and the reference falls inside
    pkg = uniform_package((6, 6), ST, seed=3)
    eng, model = package_engine(pkg, m=4, reference_cells=0)
    P1 = hotspot_power(ST, pkg.shape, pkg.power.sum(), seed=7)
    eng.query("Tj_max", "region:die", conditions={"power": P1}, accuracy=1e-9)
    for P in (pkg.power, 0.5 * pkg.power + 0.5 * P1):
        _, ref = _truth(pkg, P)
        for where, truth in (("interface:die/tim", ref["q_tim"].sum()), ("ubump[2,3]", ref["q_ubump"][2, 3])):
            a = eng.query("q", where, conditions={"power": P})
            assert a.uncertainty > 0 and not a.rigorous
            assert abs(a.value - truth) <= 3 * a.uncertainty + 1e-12, (where, a.method, a.value, truth)
