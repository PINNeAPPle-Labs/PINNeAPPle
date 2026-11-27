"""Measured checks of the adaptive-fidelity router: the surrogate only where it can be trusted, and the reason kept.

The acceptance of #161 is that the routing decision and its reason are stored with each request, so the first tests
check the records; the ones that matter run the router against a case where the surrogate *would* be wrong and count
how often that answer escaped.

Measured on this machine (``2026-11-26``): a GP fitted on 60 points of ``exp(1.5 a) cos(2 b) + 0.3 a^2`` inside
``[0, 1]^2``, 400 requests uniform on ``[-0.15, 1.15]^2``, tolerance 0.05, every request verified against the solver:

======================================  ====================  ========================  ========================
policy                                  served by surrogate   served beyond tolerance   max error served
======================================  ====================  ========================  ========================
APPROVED + domain + tolerance           0.593 (237 requests)  0                         0.009
APPROVED + tolerance, no domain         0.870 (348 requests)  5                         0.081
APPROVED + domain, no tolerance         0.593 (237 requests)  0                         0.009
======================================  ====================  ========================  ========================

The middle row is why the domain is not optional: without it the router serves five answers it should not. The cost
ratio (a solver call priced at 100 surrogate calls) is 0.413 for the full router.

The horizon is the fifth condition (#347), and it is measured the same way: with the useful lead at 120 h and a 48 h
request the router serves the same 237 requests as without a horizon at all, and with the useful lead at 24 h it serves
**none** of them — every one of the 237 goes to the solver with ``past_horizon`` as its only code (cost ratio 1.0).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # noqa: E402

from pinneapple_analysis.fidelity import (  # noqa: E402
    SOLVER_COST,
    SURROGATE_COST,
    FidelityPolicy,
    audit,
    serve,
)
from pinneapple_analysis.trust.adapters import weather_horizon  # noqa: E402
from pinneapple_analysis.trust.record import (  # noqa: E402
    GOOD,
    Horizon,
    Measurement,
    Trust,
    TrustValue,
    Uncertainty,
)
from pinneapple_analysis.trust.trust_report import (  # noqa: E402
    APPROVED,
    CONTRADICTS,
    SUPPORTS,
    Check,
    TrustReport,
)
from pinneapple_analysis.uncertainty.applicability import ApplicabilityDomain  # noqa: E402

TOL = 0.05
N_REQUESTS = 400
N_SERVED = 237                 # what the verdict, the domain and the tolerance accept, out of the 400 (measured below)


def truth(X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.exp(1.5 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.3 * X[:, 0] ** 2


def horizon(requested: float, *, useful: float = 120.0, no_skill: float = 240.0) -> Horizon:
    """A horizon as #345 builds it — from the weather adapter, so the numbers in these tests are its, not made up."""
    return weather_horizon({"useful_hours": useful, "no_skill_hours": no_skill}, requested_lead=requested)


class Surrogate:
    """A GP over ``[0, 1]^2`` that extrapolates badly outside it — the situation the router exists for."""

    def __init__(self, n_train: int = 60, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, (n_train, 2))
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(2), (1e-1, 1e1), nu=2.5)
                  + WhiteKernel(1e-6, (1e-10, 1e-1)))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                               random_state=0).fit(X, truth(X))
        self.X_ref = X

    def predict(self, X):
        m, s = self.gp.predict(np.atleast_2d(np.asarray(X, dtype=float)), return_std=True)
        return m, m - 1.96 * s, m + 1.96 * s


@pytest.fixture(scope="module")
def rig():
    s = Surrogate()
    domain = ApplicabilityDomain.fit(s.X_ref, ["a", "b"])
    ok = TrustReport(checks=[Check("accuracy", SUPPORTS), Check("calibration", SUPPORTS)])
    X = np.random.default_rng(1).uniform(-0.15, 1.15, (400, 2))
    return s, domain, ok, X


def _run(rig, *, domain="default", tolerance=TOL, trust="ok", verify=True, horizon=None, unknown_horizon="solver"):
    s, dom, ok, X = rig
    return serve(X, s, truth, policy=FidelityPolicy(tolerance=tolerance, unknown_horizon=unknown_horizon),
                 domain=dom if domain == "default" else None,
                 trust=ok if trust == "ok" else trust,
                 horizon=horizon,
                 verify=range(len(X)) if verify else None)


def test_every_request_leaves_with_its_decision_and_its_reason(rig):
    r = _run(rig)
    assert len(r.records) == 400
    for rec in r.records:
        assert rec["route"] in ("surrogate", "high_fidelity")
        assert isinstance(rec["codes"], list) and isinstance(rec["reasons"], list)
        assert rec["trust"] == APPROVED
        if rec["route"] == "high_fidelity":
            assert rec["reasons"] and rec["solver_value"] is not None
            assert rec["value"] == rec["solver_value"]
        else:
            assert not rec["reasons"] and rec["solver_value"] is None
    json.dumps(r.to_dict())                                        # what the app stores with the request
    assert r.policy["tolerance"] == TOL and r.policy["domain"] is not None


def test_the_router_serves_no_answer_beyond_the_tolerance(rig):
    """The number that matters: with the domain and the tolerance, zero violations escape."""
    a = audit(_run(rig))
    assert a["served_beyond_tolerance"] == 0, a
    assert a["max_error_served"] < TOL
    assert 0.4 < a["share_served_by_surrogate"] < 0.8, a
    assert a["reasons"]["outside_range"] > 100 and a["reasons"]["too_uncertain"] > 10
    assert a["cost_ratio"] < 0.5, a                                     # the solver is what the router saves


def test_without_the_domain_the_router_serves_answers_it_should_not(rig):
    """The measured justification for the domain: the same surrogate, the same tolerance, no domain."""
    a = audit(_run(rig, domain=None))
    assert a["served_beyond_tolerance"] > 0, a
    assert a["max_error_served"] > TOL
    assert a["share_served_by_surrogate"] > 0.8


def test_the_tolerance_is_what_sends_the_uncertain_requests_away(rig):
    loose = audit(_run(rig, tolerance=None))
    assert loose["reasons"]["too_uncertain"] == 0
    assert loose["share_served_by_surrogate"] >= audit(_run(rig))["share_served_by_surrogate"]


def test_a_model_under_review_is_never_used_even_inside_the_domain(rig):
    review = TrustReport(checks=[Check("accuracy", SUPPORTS), Check("drift", CONTRADICTS, reason="input drift")])
    r = _run(rig, trust=review)
    assert all(rec["route"] == "high_fidelity" for rec in r.records)
    assert all("trust_review" in rec["codes"] for rec in r.records)
    assert any("drift" in why for why in r.records[0]["reasons"])
    a = audit(r)
    assert a["share_served_by_surrogate"] == 0.0 and a["cost_ratio"] == 1.0


def test_a_request_without_a_verdict_goes_to_the_solver(rig):
    r = _run(rig, trust=None)
    assert all(rec["route"] == "high_fidelity" for rec in r.records)
    assert all("no_verdict" in rec["codes"] for rec in r.records)
    r2 = serve(rig[3], rig[0], truth, policy=FidelityPolicy(tolerance=TOL, unknown_trust="surrogate"),
               domain=rig[1])
    assert sum(rec["route"] == "surrogate" for rec in r2.records) > 0


def test_a_per_request_verdict_is_read(rig):
    """A verdict that depends on the case: one per request, from the same callable."""
    s, dom, ok, X = rig

    def per_request(Z):
        Z = np.atleast_2d(Z)
        inside = np.all((Z >= 0.0) & (Z <= 1.0), axis=1)
        return [(APPROVED, []) if ok_ else ("REJECT", ["outside the model's declared regime"])
                for ok_ in inside]

    r = serve(X, s, truth, policy=FidelityPolicy(tolerance=None), domain=dom, trust=per_request)
    rejected = [rec for rec in r.records if "trust_reject" in rec["codes"]]
    accepted = [rec for rec in r.records if "trust_reject" not in rec["codes"]]
    assert rejected and accepted
    assert all(rec["route"] == "high_fidelity" for rec in rejected)
    assert all(any("declared regime" in why for why in rec["reasons"]) for rec in rejected)
    assert all(rec["trust"] == "REJECT" for rec in rejected)
    assert sum(rec["route"] == "surrogate" for rec in accepted) > 0


def test_an_interval_that_straddles_the_limit_cannot_decide(rig):
    s, dom, ok, X = rig
    mean = s.predict(X)[0]
    lim = float(np.median(mean))                    # a pass/fail limit where the intervals actually reach
    r = serve(X, s, truth, policy=FidelityPolicy(tolerance=None, limits=(lim, None)), domain=dom, trust=ok)
    straddling = [rec for rec in r.records if rec["interval"][0] <= lim <= rec["interval"][1]]
    assert len(straddling) > 10
    for rec in straddling:
        assert "straddles_limit" in rec["codes"] and rec["route"] == "high_fidelity"
    clear = [rec for rec in r.records if rec["interval"][0] > lim]
    assert clear and all("straddles_limit" not in rec["codes"] for rec in clear)
    assert any(rec["verdict"] == "meets limits" for rec in clear)


def test_verification_only_touches_the_answers_the_surrogate_gave(rig):
    s, dom, ok, X = rig
    r = serve(X[:20], s, truth, policy=FidelityPolicy(tolerance=TOL), domain=dom, trust=ok, verify=range(20))
    verified = [rec for rec in r.records if "error" in rec]
    assert verified and all(rec["route"] == "surrogate" for rec in verified)
    assert all(rec["error"] == pytest.approx(abs(rec["surrogate_value"] - rec["verified_against"])) for rec in verified)
    assert audit(r)["verified"] == len(verified)


def test_the_cost_accounting_matches_the_routes(rig):
    a = audit(_run(rig))
    spent = sum(rec["cost"] for rec in _run(rig).records)
    assert a["cost"] == spent
    assert a["cost_solver_only"] == SOLVER_COST * a["n"]
    assert a["cost"] == SURROGATE_COST * a["surrogate"] + SOLVER_COST * a["high_fidelity"]


# ------------------------------------------------------------------------------- the horizon as a condition (#347)
def test_a_short_horizon_sends_every_otherwise_acceptable_request_to_the_solver(rig):
    """The deliverable of #347. Verdict APPROVED, input inside the domain, interval inside the tolerance — the horizon
    is the only thing wrong — and the router escalates on it, with nothing else on the record."""
    base = _run(rig)
    accepted = [rec["i"] for rec in base.records if rec["route"] == "surrogate"]
    assert len(accepted) == N_SERVED and base.n == N_REQUESTS

    long_lead = _run(rig, horizon=horizon(48.0))                    # 48 h asked, the model is useful to 120 h
    assert not [rec for rec in long_lead.records if "past_horizon" in rec["codes"]]
    assert [rec["route"] for rec in long_lead.records] == [rec["route"] for rec in base.records]

    short_lead = _run(rig, horizon=horizon(48.0, useful=24.0))      # same request, useless past 24 h
    a = audit(short_lead)
    assert a["reasons"]["past_horizon"] == N_REQUESTS
    assert a["share_served_by_surrogate"] == 0.0 and a["cost_ratio"] == 1.0
    assert a["served_beyond_tolerance"] == 0
    for i in accepted:                                             # why each one went: the horizon, and only it
        assert short_lead.records[i]["codes"] == ["past_horizon"]
        assert short_lead.records[i]["route"] == "high_fidelity"
        assert short_lead.records[i]["trust"] == APPROVED
    assert "past the lead at which the model stops being trustworthy (24 h" in short_lead.records[accepted[0]]["reasons"][0]


def test_the_horizon_escalates_only_the_requests_asking_for_a_lead_it_cannot_cover(rig):
    """One horizon per request: the leads the model covers are still answered by the surrogate, the others are not."""
    base = _run(rig)
    accepted = [rec["i"] for rec in base.records if rec["route"] == "surrogate"]
    per_request = [horizon(48.0, useful=24.0) if i % 2 else horizon(48.0) for i in range(N_REQUESTS)]
    r = _run(rig, horizon=per_request)
    odd, even = [i for i in accepted if i % 2], [i for i in accepted if not i % 2]
    assert odd and even
    assert all(r.records[i]["route"] == "high_fidelity" and r.records[i]["codes"] == ["past_horizon"] for i in odd)
    assert all(r.records[i]["route"] == "surrogate" and r.records[i]["codes"] == [] for i in even)
    a = audit(r)
    assert a["surrogate"] == len(even) == 123
    assert a["reasons"]["past_horizon"] == N_REQUESTS // 2


def test_a_horizon_that_measured_nothing_is_not_a_pass(rig):
    """A missing check never counts as a pass: a horizon was given and it cannot vouch, so the solver answers — unless
    the caller says otherwise, and then it gets the routes of a run without a horizon."""
    nothing = Horizon.not_measured("the rollout was never scored against truth")
    r = _run(rig, horizon=nothing)
    assert all(rec["route"] == "high_fidelity" for rec in r.records)
    assert all("horizon_not_measured" in rec["codes"] for rec in r.records)
    accepted = [rec for rec in _run(rig).records if rec["route"] == "surrogate"]
    assert all(r.records[rec["i"]]["codes"] == ["horizon_not_measured"] for rec in accepted)
    assert "never scored against truth" in r.records[accepted[0]["i"]]["reasons"][0]
    loose = _run(rig, horizon=nothing, unknown_horizon="surrogate")
    assert [rec["route"] for rec in loose.records] == [rec["route"] for rec in _run(rig).records]
    assert all("horizon_not_measured" not in rec["codes"] for rec in loose.records)


def _trust_with(h: Horizon) -> Trust:
    """A :class:`~pinneapple_analysis.trust.record.Trust` of #345 with every gate measured and passing: the object a
    consumer reads, carrying the horizon."""
    m = TrustValue.measured
    return Trust(kind="surrogate",
                 in_domain=m(1.0, source="test gate", reference=1.0, reference_source="test"),
                 domain_score=m(0.94, source="test gate"),
                 horizon=h,
                 measurement=Measurement(m(0.0, source="test measurement_check")),
                 uncertainty=Uncertainty(level=0.9, coverage=m(0.93, source="test calibration"),
                                         width=m(0.02, source="test calibration"),
                                         ece=m(0.01, source="test calibration")))


def test_the_router_reads_the_horizon_a_trust_object_carries(rig):
    """``pp.trust``'s ``Horizon`` is one of the router's conditions: the verdict comes from the badge, the horizon from
    the object — no second argument, no re-measurement."""
    s, dom, ok, X = rig
    covered = _trust_with(horizon(48.0))
    assert covered.badge() == GOOD
    r = serve(X, s, truth, policy=FidelityPolicy(tolerance=TOL), domain=dom, trust=covered)
    assert r.records[0]["trust"] == APPROVED
    assert [rec["route"] for rec in r.records] == [rec["route"] for rec in _run(rig).records]
    assert r.policy["horizon"]["requested"]["value"] == 48.0
    json.dumps(r.to_dict())                            # the horizon goes into what the app stores with the request

    past = _trust_with(horizon(48.0, useful=24.0))
    assert past.badge() == GOOD                       # the badge only refuses past the no-skill lead (240 h)
    r2 = serve(X, s, truth, policy=FidelityPolicy(tolerance=TOL), domain=dom, trust=past)
    assert all(rec["route"] == "high_fidelity" for rec in r2.records)
    assert all("past_horizon" in rec["codes"] for rec in r2.records)


def test_the_headline_numbers_this_module_quotes(rig):
    """Every number in this module's docstring, asserted here: the share, the violations and the price."""
    full = audit(_run(rig))
    assert full["n"] == N_REQUESTS and full["surrogate"] == N_SERVED
    assert full["share_served_by_surrogate"] == pytest.approx(0.593, abs=5e-4)
    assert full["served_beyond_tolerance"] == 0 and full["max_error_served"] < TOL
    assert full["max_error_served"] == pytest.approx(0.009, abs=5e-4)
    assert full["cost_ratio"] == pytest.approx(0.413, abs=5e-4)
    no_domain = audit(_run(rig, domain=None))
    assert no_domain["share_served_by_surrogate"] == pytest.approx(0.870, abs=5e-4)
    assert no_domain["served_beyond_tolerance"] == 5
    assert no_domain["max_error_served"] == pytest.approx(0.081, abs=5e-4)
    no_tolerance = audit(_run(rig, tolerance=None))
    assert no_tolerance["surrogate"] == N_SERVED and no_tolerance["served_beyond_tolerance"] == 0


def test_the_contract_rejects_what_it_cannot_route(rig):
    s, dom, ok, X = rig
    with pytest.raises(ValueError, match="no requests"):
        serve(np.empty((0, 2)), s, truth, trust=ok)
    with pytest.raises(ValueError, match="returned"):
        serve(X[:5], lambda Z: (np.zeros(5), np.zeros(4), np.zeros(5)), truth, trust=ok)
    with pytest.raises(ValueError, match="cannot read a trust verdict"):
        serve(X[:5], s, truth, trust=object())
    with pytest.raises(ValueError, match="solver must return|returned 2 values"):
        serve(X[:5], s, lambda Z: np.zeros(2), policy=FidelityPolicy(tolerance=None), trust="REVIEW")
    with pytest.raises(ValueError, match="must return"):
        serve(X[:5], lambda Z: np.zeros(5), truth, trust=ok)
    with pytest.raises(ValueError, match="horizons for 5 requests"):
        serve(X[:5], s, truth, trust=ok, horizon=[horizon(48.0)] * 3)
    with pytest.raises(ValueError, match="cannot read a horizon"):
        serve(X[:5], s, truth, trust=ok, horizon=[horizon(48.0)] * 4 + [42.0])
