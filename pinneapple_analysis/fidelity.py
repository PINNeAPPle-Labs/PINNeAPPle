"""Adaptive fidelity: serve each request with the surrogate or with the solver, and say why (#161).

Phase 5 of ``docs/strategy/STRATEGY_PHYSICS_AI.md``. The library already knows how to decide whether a *model* can be
trusted (:class:`pinneapple_analysis.trust.trust_report.TrustReport`: ``APPROVED`` / ``REVIEW`` / ``REJECT``), how to
decide whether a *case* is inside the data (:class:`pinneapple_analysis.uncertainty.applicability.ApplicabilityDomain`)
and how to turn that into a route with reasons
(:func:`pinneapple_analysis.uncertainty.applicability.route_cases`). What was missing is the engine that puts them
together and *executes* the route: the surrogate answers when the model is APPROVED **and** the case is inside the
domain, the solver answers otherwise, and every request leaves with the decision, the reason and both numbers attached.

:func:`serve` is that engine. The decision is per request and is written into the request's record::

    {"i": 0, "route": "surrogate", "codes": [], "reasons": [], "trust": "APPROVED",
     "value": 1.02, "interval": [0.99, 1.05], "surrogate_value": 1.02, "solver_value": None, "cost": 1}

A record with ``route == "high_fidelity"`` always says which condition fired: the model's verdict, the input outside
the validity range, a sparse neighbourhood, an interval wider than the tolerance, an interval that straddles a
pass/fail limit, or a lead this request asks to cover that is past the horizon at which the model stops being
trustworthy (:class:`pinneapple_analysis.trust.record.Horizon`, #345 — the weather lead of
:func:`pinneapple_physics.weather.evaluate.horizons` or the rollout lead of
:func:`pinneapple_analysis.validation.rollout.divergence_time`). The horizon is read from the ``horizon`` argument or,
when the router is handed a :class:`~pinneapple_analysis.trust.record.Trust`, from the one the object carries: a
surrogate that is approved, in domain and narrow can still be asked about a lead it measured itself to be useless at.

The measurement that backs it is in ``tests/test_fidelity.py``: over 400 requests against a surrogate fitted on a
sub-box, the router keeps **every** answer within the tolerance (0 violations served by the surrogate, 237 requests
answered by it), while the same surrogate with the domain switched off serves 5 answers outside it; the share answered
by the surrogate is what the solver is saved, and the test reports it. The horizon is measured the same way: the same
237 requests the verdict, the domain and the tolerance all accept are served by the surrogate at a 120 h useful lead
and **all 237** go to the solver at a 24 h one, with ``past_horizon`` as the only code on those records.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pinneapple_analysis.trust.record import FADING, GOOD, NO_SKILL, Horizon, Trust
from pinneapple_analysis.trust.trust_report import APPROVED, TrustReport
from pinneapple_analysis.uncertainty.applicability import (
    ROUTE_CODES,
    ApplicabilityDomain,
    route_cases,
    routing_summary,
)

# the verdict a :class:`Trust` badge stands for: the badge is a decision over measured gates, so it maps onto the
# verdict the router already reads instead of adding a second vocabulary
_BADGE_VERDICT = {GOOD: APPROVED, FADING: "REVIEW", NO_SKILL: "REJECT"}

# the price of each answer, in surrogate calls: one solver call is counted as 100 of them (the ratio the docs quote
# for a mid-fidelity RANS run against a zoo surrogate), so the saving of a route is a number and not a claim
SURROGATE_COST = 1
SOLVER_COST = 100


@dataclass
class FidelityPolicy:
    """When a request goes to the solver.

    ``trust`` is the verdict the surrogate needs to be used at all; ``escalate`` lists the verdicts that send a request
    to the solver (``REVIEW`` counts: a model under review is not approved). ``tolerance`` is the accuracy the decision
    needs in output units — a request whose surrogate interval is wider than that is escalated even inside the domain.
    ``limits`` is a pass/fail band: an interval that contains a limit cannot decide, so it is escalated too.
    ``unknown_trust`` is what to do with a request nobody has a verdict for, and ``unknown_horizon`` what to do when a
    horizon was given but nothing in it was measured (a missing check is not a pass: by default the solver answers).
    """
    trust: str = APPROVED
    escalate: tuple[str, ...] = ("REVIEW", "REJECT")
    tolerance: float | None = None
    limits: tuple[float | None, float | None] | None = None
    unknown_trust: str = "solver"
    unknown_horizon: str = "solver"
    use_domain: bool = True


@dataclass
class FidelityReport:
    """One record per request, and the counts that answer "what did this save, and what did it cost in accuracy"."""
    records: list[dict]
    policy: dict
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.records)

    def summary(self) -> dict:
        routes = [{"route": r["route"], "codes": r["codes"]} for r in self.records]
        s = routing_summary(routes)
        s["trust_counts"] = {k: sum(r["trust"] == k for r in self.records)
                             for k in ("APPROVED", "REVIEW", "REJECT", None)}
        skipped = [r for r in self.records if r["route"] == "surrogate"]
        s["share_served_by_surrogate"] = len(skipped) / self.n if self.n else 0.0
        return s

    def to_dict(self) -> dict:
        return {"method": "adaptive fidelity: surrogate when the model is approved and the case is inside the domain",
                "policy": self.policy, "n": self.n, "summary": self.summary(), "seconds": self.seconds,
                "records": self.records}


def _as_arrays(out):
    """``(mean, lower, upper)`` from a three-tuple, a ``(mean, std)`` pair with ``level``, or a mapping."""
    if isinstance(out, Mapping):
        m = np.asarray(out["mean"], dtype=float).ravel()
        lo = np.asarray(out.get("lower", out.get("lo")), dtype=float).ravel()
        hi = np.asarray(out.get("upper", out.get("hi")), dtype=float).ravel()
        return m, lo, hi
    if isinstance(out, tuple) and len(out) == 3:
        return tuple(np.asarray(a, dtype=float).ravel() for a in out)
    raise ValueError("the surrogate must return (mean, lower, upper) or a mapping with those keys")


def _predict(surrogate, X):
    """Accepts an object with ``predict`` or a plain callable."""
    out = surrogate.predict(X) if hasattr(surrogate, "predict") else surrogate(X)
    return _as_arrays(out)


def _one(t) -> tuple[str | None, list[str]]:
    if t is None:
        return None, []
    if isinstance(t, str):
        return t, []
    if isinstance(t, TrustReport):
        return t.decisao, list(t.motivos)
    if isinstance(t, Trust):
        return _BADGE_VERDICT[t.badge()], [t.badge_reason()]
    if isinstance(t, tuple) and len(t) == 2:
        return str(t[0]), [str(r) for r in (t[1] or [])]
    raise ValueError(f"cannot read a trust verdict from {t!r}")


def _verdicts(trust, X) -> list[tuple[str | None, list[str]]]:
    """One verdict per request. ``trust`` is a ``TrustReport``, a ``Trust`` (its ``badge()``), a verdict string, a
    ``(verdict, reasons)`` pair, or a callable ``X -> `` any of those *or a list of them, one per request* — the last
    form for a verdict that depends on the case (an input outside the model's declared regime, a new geometry)."""
    t = trust(X) if callable(trust) and not isinstance(trust, (Trust, TrustReport)) else trust
    n = np.atleast_2d(X).shape[0]
    if isinstance(t, (list, np.ndarray)) and len(t) == n:
        return [_one(item) for item in t]
    return [_one(t)] * n


def _horizons(horizon, trust, n: int) -> list[Horizon | None]:
    """One horizon per request: the ``horizon`` argument (a single one, or one per request), else the one a ``Trust``
    carries. ``None`` means no horizon was measured for that request, and the horizon condition is then silent."""
    if horizon is not None:
        if isinstance(horizon, Horizon):
            return [horizon] * n
        seq = list(horizon)
        if len(seq) != n:
            raise ValueError(f"horizon gave {len(seq)} horizons for {n} requests")
        for h in seq:
            if h is not None and not isinstance(h, Horizon):
                raise ValueError(f"cannot read a horizon from {h!r}")
        return seq
    if isinstance(trust, Trust):
        return [trust.horizon] * n
    if isinstance(trust, (list, np.ndarray)) and len(trust) == n:
        return [t.horizon if isinstance(t, Trust) else None for t in trust]
    return [None] * n


def _lead(v) -> str:
    """A lead as it is written in a reason: the number with the unit the value itself carries."""
    return f"{v.value:g}" + (f" {v.unit}" if v.unit else "")


def _horizon_condition(h: Horizon | None, policy: FidelityPolicy) -> tuple[list[str], list[str]]:
    """Whether the lead this request asks to cover is inside the lead at which the model stops being trustworthy.

    The trust horizon used is the **earliest** lead the model measured itself to lose there — ``useful`` (the lead at
    which it stops helping) when it was measured, ``no_skill`` otherwise; when both are, the smaller one. A request
    asking for a longer lead than that cannot be answered by a model that measured itself useless at it, whatever its
    verdict, its domain and its interval say. This is deliberately stricter than :meth:`Trust.gates`, whose horizon
    gate only asks whether the prediction carries *any* information (past ``no_skill``): a served answer has to be
    *useful*, which is the ``useful`` lead.
    """
    if h is None:
        return [], []
    requested = h.requested
    leads = [v for v in (h.useful, h.no_skill) if v.is_measured]
    if not requested.is_measured or not leads:
        if policy.unknown_horizon != "solver":
            return [], []
        why = requested.reason if not requested.is_measured else (
            h.useful.reason if not h.useful.is_measured else h.no_skill.reason)
        return ["horizon_not_measured"], [
            f"a horizon was given and nothing in it says the lead this request asks for is still covered: {why}"]
    lead = min(leads, key=lambda v: v.value)  # type: ignore[arg-type,return-value]
    if float(requested.value) > float(lead.value):  # type: ignore[arg-type]
        return ["past_horizon"], [
            f"the lead this request asks to cover ({_lead(requested)}) is past the lead at which the model stops "
            f"being trustworthy ({_lead(lead)}, from {lead.source})"]
    return [], []


def serve(requests, surrogate, solver: Callable, *, policy: FidelityPolicy | None = None,
          domain: ApplicabilityDomain | None = None, trust=None, horizon=None, verify: Sequence[int] | None = None,
          outputs: Sequence[str] | None = None) -> FidelityReport:
    """Answer every request with the surrogate or with the solver, and record the decision with the request.

    Parameters
    ----------
    requests : the inputs, ``(n, d)`` (an array, a list of dicts with the domain's names, or a DataFrame).
    surrogate : an object with ``predict(X) -> (mean, lower, upper)``, or a callable returning that triple. The
        interval is what the tolerance is checked against; a surrogate that cannot say how sure it is cannot be
        routed by tolerance (pass ``tolerance=None`` and rely on the domain and the verdict).
    solver : ``X -> (n,)``, the expensive path. Called once, with all the escalated requests.
    policy : see :class:`FidelityPolicy`.
    domain : an :class:`~pinneapple_analysis.uncertainty.applicability.ApplicabilityDomain` fitted on the surrogate's
        training data. Without it the route depends on the verdict and the interval only.
    trust : the model's ``TrustReport``, a ``Trust`` (its ``badge()`` is the verdict and its ``horizon`` is read),
        a verdict string, a callable ``X -> TrustReport | Trust | str``, or a ``(verdict, reasons)`` pair — the last
        two when the verdict is per request.
    horizon : a :class:`~pinneapple_analysis.trust.record.Horizon` for the model, or one per request. A request whose
        requested lead is past the lead at which the model stops being trustworthy goes to the solver with the
        ``past_horizon`` code, whatever its verdict, domain and interval say. Without it the horizon read from a
        ``Trust`` passed as ``trust`` is used; without either, the condition is not evaluated.
    verify : indices whose solver answer is also computed, so the router's own claim can be measured (its error is
        recorded in the request's ``error``). The escalated requests need no verification: they are the solver's.
    """
    pol = policy or FidelityPolicy()
    X = np.atleast_2d(np.asarray(requests, dtype=float))
    n = X.shape[0]
    if n == 0:
        raise ValueError("no requests")
    t0 = time.perf_counter()
    mean, lo, hi = _predict(surrogate, X)
    if not (mean.shape[0] == lo.shape[0] == hi.shape[0] == n):
        raise ValueError(f"the surrogate returned {mean.shape[0]}/{lo.shape[0]}/{hi.shape[0]} values for {n} requests")
    reading = trust(X) if callable(trust) and not isinstance(trust, (Trust, TrustReport)) else trust
    verdicts = _verdicts(reading, X)
    horizons = _horizons(horizon, reading, n)

    if domain is not None:
        a = domain.assess(X)
        routes = route_cases(a["status"], mean, lo, hi, pol.tolerance, pol.limits)
    else:
        routes = route_cases(np.full(n, "inside"), mean, lo, hi, pol.tolerance, pol.limits)

    out = []
    for i in range(n):
        verdict, trust_reasons = verdicts[i]
        codes, reasons = list(routes[i]["codes"]), list(routes[i]["reasons"])
        code, why = _horizon_condition(horizons[i], pol)
        codes.extend(code)
        reasons.extend(why)
        if verdict is None:
            if pol.unknown_trust == "solver":
                codes.append("no_verdict")
                reasons.append("no trust verdict for this model: the solver answers")
        elif verdict in pol.escalate:
            codes.append(f"trust_{verdict.lower()}")
            reasons.append(f"the model's trust verdict is {verdict}"
                           + (f" ({'; '.join(trust_reasons)})" if trust_reasons else ""))
        route = "high_fidelity" if reasons else "surrogate"
        out.append({"i": i, "route": route, "codes": codes, "reasons": reasons, "trust": verdict,
                    "value": float(mean[i]), "interval": [float(lo[i]), float(hi[i])],
                    "surrogate_value": float(mean[i]), "solver_value": None,
                    "verdict": routes[i].get("verdict"),
                    "cost": SURROGATE_COST if route == "surrogate" else 0})

    hf = [r["i"] for r in out if r["route"] == "high_fidelity"]
    if hf:
        solved = np.asarray(solver(X[hf]), dtype=float).ravel()
        if solved.shape[0] != len(hf):
            raise ValueError(f"the solver returned {solved.shape[0]} values for {len(hf)} requests")
        for i, v in zip(hf, solved, strict=True):
            out[i]["solver_value"] = float(v)
            out[i]["value"] = float(v)
            out[i]["cost"] = SOLVER_COST
    extra = [i for i in (verify or []) if out[i]["route"] == "surrogate"]
    if extra:
        truth = np.asarray(solver(X[extra]), dtype=float).ravel()
        for i, v in zip(extra, truth, strict=True):
            out[i]["error"] = abs(out[i]["surrogate_value"] - float(v))
            out[i]["verified_against"] = float(v)

    known = [h for h in horizons if h is not None]
    report = FidelityReport(records=out, policy={"trust": pol.trust, "escalate": list(pol.escalate),
                                                 "tolerance": pol.tolerance, "limits": list(pol.limits or (None, None)),
                                                 "domain": domain.to_dict() if domain is not None else None,
                                                 "unknown_trust": pol.unknown_trust,
                                                 "unknown_horizon": pol.unknown_horizon,
                                                 "horizon": (known[0].to_dict()
                                                             if known and all(h is known[0] for h in known) else None)},
                            seconds=round(time.perf_counter() - t0, 4))
    if outputs:
        for r in report.records:
            r["output"] = outputs[0]
    return report


def audit(report: FidelityReport) -> dict:
    """How the router did, from its own records: the violations it served, and what it saved.

    ``served_beyond_tolerance`` counts the requests the surrogate answered whose verified error exceeds the policy's
    tolerance — the number that must be zero for the router to be doing its job. ``cost_ratio`` compares the priced
    cost (a solver call is 100 surrogate calls) with what a solver-only policy would have spent. ``reasons`` counts
    every code the records carry, the horizon's (``past_horizon``, ``horizon_not_measured``) included.
    """
    tol = report.policy.get("tolerance")
    verified = [r for r in report.records if "error" in r]
    bad = [r for r in verified if tol is not None and r["error"] > tol]
    spent = sum(r["cost"] for r in report.records)
    return {**report.summary(), "verified": len(verified),
            "max_error_served": max((r["error"] for r in verified), default=None),
            "served_beyond_tolerance": len(bad), "served_beyond_examples": [r["i"] for r in bad[:10]],
            "cost": spent, "cost_solver_only": SOLVER_COST * report.n,
            "cost_ratio": spent / (SOLVER_COST * report.n) if report.n else None,
            "reasons": {c: sum(c in r["codes"] for r in report.records) for c in
                        list(ROUTE_CODES) + ["past_horizon", "horizon_not_measured", "trust_review", "trust_reject",
                                             "no_verdict"]}}


__all__ = ["FidelityPolicy", "FidelityReport", "serve", "audit"]
