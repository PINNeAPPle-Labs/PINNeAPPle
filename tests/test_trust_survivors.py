"""The trust channel of the NSGA-II survivor rule (#297): what it does, and the numbers it is worth.

The three mechanisms are tested as mechanics (:func:`trust_aware_survivors` on constructed cases, where the
alternative rules must *disagree*), and then the decision itself is tested as the experiment it was made from:
:func:`~pinneapple_design.design_optimizer.trust_survivors.benchmark_trust_survivors` on the airfoil front, with the
headline numbers of the module docstring asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.surrogate_loop import Constraint
from pinneapple_analysis.uncertainty.applicability import ApplicabilityDomain
from pinneapple_design.design_optimizer.optimizer import DesignOptimizerConfig, EvolutionaryDesignOptimizer
from pinneapple_design.design_optimizer.survivors import (TrustVerdict, constrained_select,
                                                          trust_aware_survivors)
from pinneapple_design.design_optimizer.trust_survivors import (AIRFOIL_CL, TOL_CD, TOL_CL, TOL_THICKNESS,
                                                                benchmark_trust_survivors, search, verdicts)



# --------------------------------------------------------------------------- a small domain to read verdicts against
@pytest.fixture(scope="module")
def domain() -> ApplicabilityDomain:
    """40 rows filling [0.1, 0.9]^2: an applicability domain with a range and a k-NN level to hit both codes."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 0.9, size=(40, 2))
    return ApplicabilityDomain.fit(X, names=["a", "b"], k=3, level=0.9)


def _payload(x, value=0.1, sd=0.0, constraint=1.0, constraint_sd=0.0):
    return (np.array([[value, constraint]], float), np.array([[sd, constraint_sd]], float))


# --------------------------------------------------------------------------- the verdict
def test_a_clean_candidate_is_trusted_and_carries_no_violation(domain):
    mean, sd = _payload([0.5, 0.5], value=0.1, sd=1e-6, constraint=1.0)
    v = verdicts([[0.5, 0.5]], mean, sd, [Constraint("g", 0.5)], tolerance=1e-3, domain=domain)[0]
    assert v.vouched and v.level == "trusted"
    assert v.violation == 0.0 and v.codes == ()


def test_out_of_range_contradicts_and_sits_above_the_indeterminate_band(domain):
    """``outside_range`` is untrusted with a violation above 1; ``too_uncertain`` is undetermined and below 1.

    The two bands are the point of the ladder: a survivor rule that summed them into one violation would be unable to
    say that one candidate is contradicted and the other merely unjudged.
    """
    X = [[0.5, 0.5], [5.0, 0.5], [0.5, 0.5]]
    mean = np.array([[0.1, 1.0]] * 3)
    sd = np.zeros((3, 2))
    sd[2, 0] = 1.0                                      # a 95 % band of +-1.96 against a 1e-3 tolerance
    out = verdicts(X, mean, sd, [Constraint("g", 0.5)], tolerance=1e-3, domain=domain)
    assert out[0].vouched
    assert out[1].level == "untrusted" and "outside_range" in out[1].codes
    assert out[2].level == "undetermined" and out[2].codes == ("too_uncertain",)
    assert 0.0 < out[2].violation <= 1.0 < out[1].violation


def test_the_tolerance_is_per_objective_and_optional(domain):
    """An interval too wide for objective 1 is not too wide for objective 0, and no tolerance means no judgement."""
    X = [[0.5, 0.5]]
    mean = np.array([[0.1, 0.2]])
    sd = np.array([[1e-9, 1.0]])
    per_objective = verdicts(X, mean, sd, tolerance=(1e-3, 1e-3), domain=domain)[0]
    assert per_objective.level == "undetermined"
    assert verdicts(X, mean, sd, tolerance=None, domain=domain)[0].vouched
    assert verdicts(X, mean, sd, tolerance=[1e-3], domain=domain)[0].level == "undetermined"      # broadcast
    with pytest.raises(ValueError):
        verdicts(X, mean, sd, tolerance=(1e-3, 1e-3, 1e-3), domain=domain)


def test_a_constraint_interval_that_straddles_the_limit_is_undetermined_for_the_design(domain):
    v = verdicts([[0.5, 0.5]], np.array([[0.1, 0.5]]), np.array([[1e-9, 0.05]]), [Constraint("cl", 0.5)],
                 tolerance=None, domain=domain)[0]
    assert v.level == "untrusted"
    assert v.codes == ("cl:straddles_limit",)
    assert 0.0 < v.violation <= 1.0
    clear = verdicts([[0.5, 0.5]], np.array([[0.1, 0.9]]), np.array([[1e-9, 0.05]]), [Constraint("cl", 0.5)],
                     tolerance=None, domain=domain)[0]
    assert clear.vouched


def test_a_candidate_shaped_differently_from_the_prediction_is_refused(domain):
    with pytest.raises(ValueError):
        verdicts([[0.5, 0.5]], np.zeros((2, 1)), np.zeros((2, 1)), domain=domain)


# --------------------------------------------------------------------------- the three mechanisms
def _case():
    """12 vouched designs that violate a constraint by 10, against 4 unvouched designs that violate nothing."""
    obj = np.array([[i * 0.1, 1.0 - i * 0.1] for i in range(16)], dtype=float)
    pen = np.array([10.0] * 12 + [0.0] * 4)
    trust = [TrustVerdict()] * 12 + [TrustVerdict("untrusted", 1.2, ("outside_range",))] * 4
    return obj, pen, trust


def test_the_front_rule_spends_the_slots_on_the_vouched_band_first():
    obj, pen, trust = _case()
    plain = constrained_select(obj, pen, 8)
    front = trust_aware_survivors(obj, pen, trust, 8, "front")
    assert len(front) == 8
    assert np.all(front < 12)                                   # every slot went to a vouched design
    assert np.all(plain[:4] >= 12)                              # the plain rule put the feasible unvouched ones first


def test_the_penalty_rule_trades_a_vouched_infeasible_design_for_an_unvouched_feasible_one():
    """The documented failure mode of the summed form, asserted rather than argued: it *does* make that trade."""
    obj, pen, trust = _case()
    penalty = trust_aware_survivors(obj, pen, trust, 8, "penalty")
    front = trust_aware_survivors(obj, pen, trust, 8, "front")
    assert np.all(penalty[:4] >= 12)                            # the unvouched designs won on the sum: 1.2 + 0 < 0 + 10
    assert not np.any(front >= 12)


def test_remove_shrinks_the_population_and_falls_back_when_nothing_is_vouched():
    obj, pen, trust = _case()
    one_vouched = [TrustVerdict()] + [TrustVerdict("untrusted", 1.2, ("outside_range",))] * 15
    removed = trust_aware_survivors(obj, pen, one_vouched, 8, "remove")     # mu = 8, but only one is vouched
    assert len(removed) == 1 and removed[0] == 0
    none_vouched = [TrustVerdict("untrusted", 1.1, ("outside_range",))] * 16
    assert len(trust_aware_survivors(obj, pen, none_vouched, 8, "remove")) == 8   # nothing to keep: plain rule


def test_the_plain_rule_is_the_nsga2_that_was_already_there():
    rng = np.random.default_rng(3)
    obj = rng.random((24, 2))
    pen = np.where(rng.random(24) < 0.3, 0.0, rng.random(24))
    assert np.array_equal(EvolutionaryDesignOptimizer._nsga2_select(obj, pen, 10),
                          constrained_select(obj, pen, 10))


def test_an_unknown_rule_is_refused():
    obj, pen, trust = _case()
    with pytest.raises(ValueError):
        trust_aware_survivors(obj, pen, trust, 4, "regret")
    with pytest.raises(ValueError):
        TrustVerdict("maybe")
    with pytest.raises(ValueError):
        TrustVerdict(violation=-1.0)


# --------------------------------------------------------------------------- the search, end to end
class _Quadratic:
    """A surrogate with no fitting: ``f = (x - c)^2`` for two objectives and a constant lift, with a fixed spread.

    Enough to drive :func:`search` through the optimiser without depending on a fit, which is what the contract
    tests below need; the benchmark uses the real surrogates.
    """

    def __init__(self, spread: float = 1e-6) -> None:
        self.spread = spread

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        f = np.column_stack([(X[:, 0] - 0.2) ** 2, (X[:, 1] - 0.8) ** 2, 1.0 + 0.0 * X[:, 0]])
        return f, np.full_like(f, self.spread)


def test_without_a_domain_and_a_tolerance_the_channel_changes_nothing():
    """``tolerance=None`` and no domain: every candidate is trusted, so ``front`` *is* plain NSGA-II."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    plain = search(bounds, _Quadratic(), None, mu=8, generations=5, seed=0, rule="plain")
    front = search(bounds, _Quadratic(), None, mu=8, generations=5, seed=0, rule="front")
    assert np.array_equal(plain.population, front.population)
    assert all(v.vouched for v in front.verdicts)


def test_the_population_keeps_its_size_and_the_reported_front_only_holds_vouched_designs():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 0.9, size=(30, 2))
    dom = ApplicabilityDomain.fit(X, names=["a", "b"], k=3, level=0.9)
    res = search([(0.0, 1.0), (0.0, 1.0)], _Quadratic(spread=0.5), dom, constraints=[Constraint("g", 0.5)],
                 tolerance=1e-3, mu=8, generations=5, seed=0, rule="front")
    assert res.population_size == 8
    assert res.front().size == 0 or all(res.verdicts[i].vouched for i in res.front())
    assert res.trust_counts()["trusted"] + res.trust_counts()["undetermined"] + res.trust_counts()["untrusted"] == 8
    assert "front_vouched" in res.to_dict()


def test_tell_carries_the_verdicts_across_generations():
    opt = EvolutionaryDesignOptimizer(DesignOptimizerConfig(population_size=6), seed=0, multi_objective=True)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    xs = opt.ask(bounds, 6)
    opt.tell(xs, [float(x[0]) for x in xs], obj_vecs=[[float(x[0]), float(x[1])] for x in xs],
             penalties=[0.0] * 6, trust=[TrustVerdict("untrusted", 2.0, ("outside_range",))] * 6)
    assert opt.trust is not None and all(v.level == "untrusted" for v in opt.trust)
    assert len(opt.population) == 6
    opt.tell(xs, [float(x[0]) for x in xs], obj_vecs=[[float(x[0]), float(x[1])] for x in xs], penalties=[0.0] * 6)
    assert opt.trust is None                     # a plain call drops the channel rather than half-keeping it


# --------------------------------------------------------------------------- the measurement
@pytest.fixture(scope="module")
def measured() -> dict:
    """The benchmark, once: both surrogates, every rule, the shipped configuration (about 25 s)."""
    return {"ensemble": benchmark_trust_survivors(surrogate="ensemble"),
            "gp": benchmark_trust_survivors(surrogate="gp")}


def test_the_vouched_first_rule_raises_the_share_of_recommendations_that_survive_verification(measured):
    """The headline of #297: with the ensemble surrogate, on both regimes, in the corner by more than double.

    Asserted on the mean *and* per seed: the plain rule wins no seed of either regime, so this is not an average of a
    good seed against a bad one.
    """
    ensemble = measured["ensemble"]["regimes"]
    covered, corner = ensemble["covered"], ensemble["corner"]
    assert covered["plain"]["honest"] == pytest.approx(0.542, abs=0.06)
    assert covered["front"]["honest"] == pytest.approx(0.750, abs=0.06)
    assert corner["plain"]["honest"] == pytest.approx(0.479, abs=0.06)
    assert corner["front"]["honest"] == pytest.approx(0.938, abs=0.05)
    for regime in (covered, corner):
        plain, front = regime["plain"], regime["front"]
        assert front["honest"] > plain["honest"]
        assert all(f > p for f, p in zip(front["per_seed"]["honest"], plain["per_seed"]["honest"], strict=True))
        assert front["trust"]["trusted"] == pytest.approx(1.0)
        assert plain["trust"]["trusted"] < 0.5


def test_the_cost_of_the_channel_is_spread_and_it_does_not_pay_for_itself_in_front(measured):
    """The price, asserted rather than tuned away: less hypervolume, and a front that stops short of the thin end."""
    corner = measured["ensemble"]["regimes"]["corner"]
    plain, front = corner["plain"], corner["front"]
    assert front["hv"] < plain["hv"]                                   # 12.76e-5 against 14.57e-5
    assert front["hv"] / plain["hv"] == pytest.approx(0.876, abs=0.06)
    assert front["thickness_min"] > plain["thickness_min"] + 0.03      # 0.142 against 0.084: the thin end is gone
    assert front["vouched_hv"] >= plain["vouched_hv"] * 0.99           # what survives the gate is not lost


def test_the_three_mechanisms_are_not_separated_by_this_benchmark(measured):
    """Why the choice between them is argued from their failure modes and not from the numbers here.

    In ``covered`` the spread between the three rules is smaller than the spread of ``front`` across its own seeds,
    and no rule wins both the honest share and the hypervolume; in ``corner`` they sit within 0.02 of each other.
    That is the finding, not a gap in the argument.
    """
    covered = measured["ensemble"]["regimes"]["covered"]
    honest = [covered[r]["honest"] for r in ("front", "penalty", "remove")]
    assert max(honest) - min(honest) < 0.12
    seed_spread = max(covered["front"]["per_seed"]["honest"]) - min(covered["front"]["per_seed"]["honest"])
    assert seed_spread > 0.2                                      # 0.562-0.875: the seeds move far more than the rules
    assert max(honest) - min(honest) < seed_spread
    assert covered["front"]["hv"] > covered["remove"]["hv"]        # and the ordering flips between the two columns
    for regime in (covered, measured["ensemble"]["regimes"]["corner"]):
        for rule in ("penalty", "remove"):
            assert regime[rule]["honest"] > regime["plain"]["honest"]
    corner = measured["ensemble"]["regimes"]["corner"]
    assert max(corner[r]["honest"] for r in ("front", "penalty", "remove")) - \
        min(corner[r]["honest"] for r in ("front", "penalty", "remove")) < 0.05
    assert corner["front"]["per_seed"]["honest"] == corner["penalty"]["per_seed"]["honest"]     # same populations


def test_trust_as_a_third_objective_does_not_buy_the_honesty(measured):
    """The rejected alternative, measured: it keeps the spread and leaves most of the population unvouched."""
    corner = measured["ensemble"]["regimes"]["corner"]
    assert corner["objective"]["honest"] < corner["plain"]["honest"]
    assert corner["objective"]["trust"]["trusted"] < 0.25
    assert corner["objective"]["hv"] > corner["front"]["hv"]           # it does keep the spread it pays for


def test_with_a_surrogate_that_does_not_lie_the_channel_is_a_pure_cost(measured):
    """The row that makes the channel opt-in: with the Gaussian process the plain population was already honest."""
    corner = measured["gp"]["regimes"]["corner"]
    assert corner["plain"]["honest"] > 0.9
    assert corner["front"]["honest"] == pytest.approx(1.0)
    assert corner["front"]["hv"] < corner["plain"]["hv"] * 0.85


def test_the_benchmark_reports_what_it_measured(measured):
    report = measured["ensemble"]
    assert report["tolerance"] == [TOL_CD, TOL_THICKNESS] and report["tolerance_cl"] == TOL_CL
    assert report["cl_target"] == AIRFOIL_CL
    for regime, arms in report["regimes"].items():
        assert set(arms) == {"plain", "front", "objective", "penalty", "remove"}
        for row in arms.values():
            assert 0.0 <= row["honest"] <= 1.0 and row["hv"] > 0.0
            assert row["population"] == report["mu"]              # the budget is the same in every arm
            assert sum(row["trust"].values()) == pytest.approx(1.0)
        assert set(report["regimes"]) == {"covered", "corner"}
