"""Measured checks of the adaptive-fidelity router: the surrogate only where it can be trusted, and the reason kept.

The acceptance of #161 is that the routing decision and its reason are stored with each request, so the first tests
check the records; the ones that matter run the router against a case where the surrogate *would* be wrong and count
how often that answer escaped.

Measured on this machine (``2026-11-25``): a GP fitted on 60 points of ``exp(1.5 a) cos(2 b) + 0.3 a^2`` inside
``[0, 1]^2``, 400 requests uniform on ``[-0.15, 1.15]^2``, tolerance 0.05, every request verified against the solver:

======================================  ==================  =====================  ======================
policy                                  served by surrogate  served beyond tolerance  max error served
======================================  ==================  =====================  ======================
APPROVED + domain + tolerance           0.632                0                      0.012
APPROVED + tolerance, no domain         0.877                7                      0.078
APPROVED + domain, no tolerance         0.632                0                      0.012
======================================  ==================  =====================  ======================

The middle row is why the domain is not optional: without it the router serves seven answers it should not. The
cost ratio (a solver call priced at 100 surrogate calls) is 0.374 for the full router.
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
from pinneapple_analysis.trust.trust_report import (  # noqa: E402
    APPROVED,
    CONTRADICTS,
    SUPPORTS,
    Check,
    TrustReport,
)
from pinneapple_analysis.uncertainty.applicability import ApplicabilityDomain  # noqa: E402

TOL = 0.05


def truth(X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.exp(1.5 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.3 * X[:, 0] ** 2


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


def _run(rig, *, domain="default", tolerance=TOL, trust="ok", verify=True):
    s, dom, ok, X = rig
    return serve(X, s, truth, policy=FidelityPolicy(tolerance=tolerance),
                 domain=dom if domain == "default" else None,
                 trust=ok if trust == "ok" else trust,
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
