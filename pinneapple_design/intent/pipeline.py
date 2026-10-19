"""RequirementSpec -> PhysicsPlan -> multi-fidelity optimisation -> verification and robustness.

The stages, each a plain function returning a typed object:

1. :func:`plan` - the template's physics, variables, models per fidelity (with availability) and failure modes;
2. :func:`optimise` - screen with the closed form (Latin hypercube), optimise with the design-study core
   (:mod:`pinneapple_design.design_optimizer`: NSGA-II on the reduced model, or the GP surrogate-assisted loop when the
   reduced model is slow), pick distinct candidates;
3. :func:`robustness` - Monte-Carlo propagation of the uncertain inputs (:func:`pinneapple_analysis.uncertainty.
   tolerance.propagate`) to the probability that every failure mode stays away, with a Wilson interval;
4. :func:`verify` - the full-fidelity solver on the chosen design, compared output by output with the reduced model;
5. :func:`assess` - the trust report (:class:`pinneapple_analysis.trust.TrustReport`) and the "will it work" verdict.

Every stage records the number of model calls and their wall time per fidelity.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spec import MISSING, RequirementSpec
from .template import CLOSED_FORM, FULL, REDUCED, DesignVariable, FullResult, ProblemTemplate


# ============================================================================================ plan
@dataclass
class PhysicsPlan:
    template: str
    title: str
    physics: list[str]
    assumptions: list[str]
    variables: list[DesignVariable]
    objectives: list[dict[str, str]]
    failure_modes: list[dict[str, Any]]
    fidelities: list[dict[str, Any]]
    uncertain: list[str] = field(default_factory=list)

    def level_available(self, level: str) -> bool:
        return any(f["level"] == level and f["available"] for f in self.fidelities)

    def to_dict(self) -> dict[str, Any]:
        return {"template": self.template, "title": self.title, "physics": self.physics,
                "assumptions": self.assumptions,
                "variables": [{"name": v.name, "label": v.label or v.name, "lower": v.lower, "upper": v.upper,
                               "unit": v.unit, "choices": list(v.choices), "integer": v.integer}
                              for v in self.variables],
                "objectives": self.objectives, "failure_modes": self.failure_modes,
                "fidelities": self.fidelities, "uncertain": self.uncertain}


def plan(spec: RequirementSpec, template: ProblemTemplate) -> PhysicsPlan:
    if spec.questions:
        raise ValueError("the requirement spec has open questions: answer them first "
                         f"({', '.join(q.slot for q in spec.questions)})")
    vs = template.variables(spec)
    nominal = {v.name: v.decode(0.5 * (v.lower + v.upper)) for v in vs}
    return PhysicsPlan(template.name, template.title, list(template.physics), list(template.assumptions), vs,
                       [{"name": o.name, "sense": o.sense, "label": o.label} for o in spec.objectives],
                       [{"name": f.name, "label": f.label, "description": f.description, "critical": f.critical}
                        for f in template.failure_modes],
                       [f.to_dict() for f in template.fidelities],
                       [u.name for u in template.uncertainties(spec, nominal)])


# ============================================================================================ optimisation
@dataclass
class CallLog:
    """Model calls per fidelity: designs evaluated and wall time."""
    calls: dict[str, int] = field(default_factory=lambda: {CLOSED_FORM: 0, REDUCED: 0, FULL: 0})
    wall: dict[str, float] = field(default_factory=lambda: {CLOSED_FORM: 0.0, REDUCED: 0.0, FULL: 0.0})

    def add(self, level: str, n: int, wall: float) -> None:
        self.calls[level] = self.calls.get(level, 0) + int(n)
        self.wall[level] = self.wall.get(level, 0.0) + float(wall)

    def per_call(self, level: str) -> float:
        n = self.calls.get(level, 0)
        return self.wall.get(level, 0.0) / n if n else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"calls": dict(self.calls), "wall_s": {k: round(v, 4) for k, v in self.wall.items()},
                "s_per_call": {k: (None if math.isnan(self.per_call(k)) else self.per_call(k)) for k in self.calls}}


@dataclass
class Candidate:
    x: np.ndarray
    design: dict[str, Any]
    outputs: dict[str, float]          # reduced-model outputs
    closed_form: dict[str, float]      # closed-form outputs at the same design
    rank: int = 0


@dataclass
class OptimisationResult:
    candidates: list[Candidate]
    method: str
    n_screen: int
    n_feasible_screen: int
    n_reduced: int
    log: CallLog
    study: Any = None                  # the design_optimizer result (OptimizationResult / ActiveLearningResult)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "n_screen": self.n_screen, "n_feasible_screen": self.n_feasible_screen,
                "n_reduced": self.n_reduced, "calls": self.log.to_dict(), "notes": self.notes,
                "candidates": [{"rank": c.rank, "design": c.design, "reduced": c.outputs,
                                "closed_form": c.closed_form} for c in self.candidates]}


def _problem(spec: RequirementSpec, template: ProblemTemplate, vs: list[DesignVariable], level: str,
             log: CallLog, **options):
    from pinneapple_design.design_optimizer import (
        Constraint,
        DesignProblem,
        DesignSpace,
        FunctionEvaluator,
        Objective,
        Variable,
    )

    space = DesignSpace([Variable(v.name, v.lower, v.upper, kind="integer" if (v.integer or v.choices) else
                                  "continuous", unit=v.unit, description=v.label) for v in vs])
    outs = [f.name for f in template.failure_modes] + [o.name for o in spec.objectives]

    def fn(X):
        t0 = time.perf_counter()
        y = template.evaluate(level, np.atleast_2d(X), spec, **options)
        log.add(level, len(X), time.perf_counter() - t0)
        return {k: np.asarray(y[k], float) for k in y}

    ev = FunctionEvaluator(fn, list(dict.fromkeys(outs)), name=f"{template.name}:{level}",
                           fidelity="model" if level == CLOSED_FORM else "reference")
    objs = [Objective(o.name, o.sense, label=o.label) for o in spec.objectives]
    cons = [Constraint(f.name, lower=0.0, label=f.label) for f in template.failure_modes]
    return DesignProblem(f"{template.name}-{level}", space, objs, cons, ev), ev


def decode(vs: list[DesignVariable], x: np.ndarray) -> dict[str, Any]:
    return {v.name: v.decode(float(xi)) for v, xi in zip(vs, x, strict=True)}


def _row(Y: dict[str, np.ndarray], i: int) -> dict[str, float]:
    return {k: float(np.asarray(v)[i]) for k, v in Y.items() if np.ndim(v) and np.asarray(v).dtype.kind in "fib"}


def optimise(spec: RequirementSpec, template: ProblemTemplate, *, seed: int = 0, n_screen: int = 2000,
             budget: int = 600, n_candidates: int = 3, slow_budget: int = 48,
             progress: Callable[[str], None] | None = None, options: dict[str, Any] | None = None,
             log: CallLog | None = None) -> OptimisationResult:
    """Screen, optimise, and return up to ``n_candidates`` feasible designs (best first objective, or nearest the
    ideal point of the Pareto front). ``options`` are passed to every ``evaluate`` call (calibration factors)."""
    from pinneapple_design.design_optimizer import run, surrogate_assisted

    say = progress or (lambda m: None)
    log = log if log is not None else CallLog()
    options = dict(options or {})
    vs = template.variables(spec)
    notes: list[str] = []
    # ---- 1. screening with the closed form
    p_cf, _ = _problem(spec, template, vs, CLOSED_FORM, log, **options)
    Xs = p_cf.space.sample(n_screen, method="lhs", seed=seed)
    ev_cf = p_cf.evaluate(Xs, tag="screen")
    feas = ev_cf.feasible
    say(f"screening: {int(feas.sum())} of {len(Xs)} designs feasible by the closed form")
    order = np.lexsort((ev_cf.F[:, 0], ~feas)) if len(ev_cf.F) else np.arange(len(Xs))
    k0 = min(24, len(Xs))
    X0 = Xs[order[:k0]]
    # ---- 2. optimisation on the reduced model
    p_red, ev_red = _problem(spec, template, vs, REDUCED, log, **options)
    import importlib.util
    has_gp = importlib.util.find_spec("sklearn") is not None
    slow = bool(getattr(template, "reduced_is_expensive", False))
    if slow and not has_gp:
        notes.append("the reduced model is expensive and scikit-learn (the GP surrogate) is not installed: NSGA-II "
                     f"ran on the reduced model with a budget of {slow_budget}")
        budget = slow_budget
    if not slow or not has_gp:
        method = "nsga2"
        res = run(p_red, "nsga2", budget=budget, seed=seed, initial=X0[: max(2, min(len(X0), budget // 2))])
        evs = res.evaluations
        study = res
        if res.status != "ok":
            notes.append(f"NSGA-II on the reduced model: {res.status} ({res.message})")
    else:
        method = "surrogate_assisted"
        n_init = max(12, 4 * len(vs))
        al = surrogate_assisted(p_red, ev_red, method="nsga2", n_initial=n_init, rounds=3, n_verify=4,
                                surrogate_budget=1200, seed=seed)
        evs = al.reference_evals
        study = al
        notes.append(f"the reduced model is expensive: a GP surrogate was optimised and {al.n_reference} designs "
                     "were checked with the reduced model")
    say(f"optimisation ({method}): {len(evs)} reduced-model evaluations")
    # ---- 3. candidates: feasible, best first objective, distinct in the normalised space
    idx = np.flatnonzero(evs.feasible)
    if len(idx) == 0:
        notes.append("no design satisfies every failure-mode margin with the reduced model; the least-violating "
                     "designs are reported")
        idx = np.argsort(evs.violation)[: 3 * n_candidates]
        key = evs.violation[idx]
    elif evs.F.shape[1] == 1:
        key = evs.F[idx, 0]
    else:
        # several objectives: the feasible Pareto front, ranked by the distance to the ideal point after scaling
        # each objective to [0, 1] over the front (the "knee" first)
        Fi = evs.F[idx]
        front = np.array([not np.any(np.all(Fi <= f, 1) & np.any(Fi < f, 1)) for f in Fi])
        idx, Fi = idx[front], Fi[front]
        lo_f, hi_f = Fi.min(0), Fi.max(0)
        key = np.linalg.norm((Fi - lo_f) / np.where(hi_f > lo_f, hi_f - lo_f, 1.0), axis=1)
        notes.append(f"{len(idx)} designs on the feasible Pareto front; candidates ranked by distance to the "
                     "ideal point (objectives scaled to [0, 1] over the front)")
    lo = np.array([v.lower for v in vs])
    span = np.array([v.upper - v.lower for v in vs])
    chosen: list[int] = []
    for i in idx[np.argsort(key, kind="stable")]:
        z = (evs.X[i] - lo) / span
        if all(np.linalg.norm(z - (evs.X[j] - lo) / span) > 0.08 for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_candidates:
            break
    cands = []
    Xc = evs.X[chosen] if chosen else np.zeros((0, len(vs)))
    cf = template.evaluate(CLOSED_FORM, Xc, spec, **options) if len(Xc) else {}
    for r, i in enumerate(chosen):
        cands.append(Candidate(evs.X[i].copy(), decode(vs, evs.X[i]), _row(evs.Y, i), _row(cf, r), r))
    return OptimisationResult(cands, method, len(Xs), int(feas.sum()), len(evs), log, study, notes)


# ============================================================================================ robustness
@dataclass
class Robustness:
    p_success: float
    ci: tuple[float, float]
    n: int
    per_mode: dict[str, float]
    margin_p05: dict[str, float]
    inputs: list[dict[str, Any]]
    sensitivity: dict[str, float]      # Spearman correlation of each input with the smallest margin
    level: str
    calibration: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"p_success": self.p_success, "ci95": list(self.ci), "n_samples": self.n, "per_mode": self.per_mode,
                "margin_p05": self.margin_p05, "inputs": self.inputs, "sensitivity": self.sensitivity,
                "model": self.level, "calibration": self.calibration}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def robustness(spec: RequirementSpec, template: ProblemTemplate, cand: Candidate, *, n: int = 2000, seed: int = 0,
               level: str = REDUCED, log: CallLog | None = None,
               calibration: dict[str, float] | None = None, options: dict[str, Any] | None = None) -> Robustness:
    """P(every failure-mode margin >= 0) under the template's uncertain inputs, by Monte Carlo on ``level``.

    ``calibration``: multiplicative factors ``{margin_output: k}`` measured against the full solver, applied to
    ``(1 + margin)`` (margins are ratios ``capacity / demand - 1``)."""
    from pinneapple_analysis.uncertainty.tolerance import propagate

    unc = template.uncertainties(spec, cand.design)
    modes = [f.name for f in template.failure_modes]
    calibration = calibration or {}
    options = dict(options or {})
    if not unc:
        y = template.evaluate(level, cand.x[None, :], spec, **options)
        ok = all(float(y[m][0]) >= 0 for m in modes)
        return Robustness(float(ok), (float(ok), float(ok)), 1, {m: float(float(y[m][0]) >= 0) for m in modes},
                          {m: float(y[m][0]) for m in modes}, [], {}, level, calibration)
    names = [u.name for u in unc]

    def fn(S):
        X = np.repeat(cand.x[None, :], len(S), axis=0)
        t0 = time.perf_counter()
        y = template.evaluate(level, X, spec, **{**options, **template.robust_inputs(X, names, S, spec)})
        if log is not None:
            log.add(level, len(S), time.perf_counter() - t0)
        cols = []
        for m in modes:
            v = np.asarray(y[m], float)
            k = calibration.get(m)
            cols.append((1 + v) * k - 1 if k else v)
        return np.stack(cols, axis=1)

    r = propagate(fn, [u.nominal for u in unc], [u.spread for u in unc], n=n, dist=[u.dist for u in unc],
                  limits=(0.0, None), seed=seed)
    Y = r.outputs
    worst = Y.min(axis=1)
    sens = {u.name: _spearman(r.inputs[:, j], worst) for j, u in enumerate(unc)}
    return Robustness(float(r.p_within), tuple(float(c) for c in r.p_within_ci), int(r.n),
                      {m: float(p) for m, p in zip(modes, r.p_within_each, strict=True)},
                      {m: float(np.percentile(Y[:, j], 5)) for j, m in enumerate(modes)},
                      [{"name": u.name, "nominal": u.nominal, "spread": u.spread, "dist": u.dist, "unit": u.unit,
                        "source": u.source} for u in unc], sens, level, calibration)


# ============================================================================================ verification
@dataclass
class Verification:
    ran: bool
    label: str
    reason: str = ""
    result: FullResult | None = None
    comparison: list[dict[str, Any]] = field(default_factory=list)   # output, full, reduced, closed_form, rel_err

    def to_dict(self) -> dict[str, Any]:
        d = {"ran": self.ran, "label": self.label, "reason": self.reason, "comparison": self.comparison}
        if self.result is not None:
            d.update({"outputs": self.result.outputs, "checks": self.result.checks, "wall_s": self.result.wall_s,
                      "notes": self.result.notes})
        return d


def verify(spec: RequirementSpec, template: ProblemTemplate, cand: Candidate, *, log: CallLog | None = None,
           workdir: str | None = None, options: dict[str, Any] | None = None) -> Verification:
    """Run the full solver on ``cand`` (if installed) and compare with the reduced and closed-form models evaluated
    with the same options as the full run."""
    fid = template.fidelity(FULL)
    if fid is None:
        return Verification(False, "", "this template has no full-fidelity model")
    ok, why = fid.available()
    if not ok:
        return Verification(False, fid.label, why)
    t0 = time.perf_counter()
    res = template.verify(cand.design, spec, workdir)
    if log is not None:
        log.add(FULL, 1, time.perf_counter() - t0)
    rows = []
    if res.compare:
        X = cand.x[None, :]
        opts = {**(options or {}), **res.compare_options}
        yr = template.evaluate(REDUCED, X, spec, **opts)
        yc = template.evaluate(CLOSED_FORM, X, spec, **opts)
        for full_key, red_key in res.compare.items():
            f = res.outputs.get(full_key)
            r = float(np.asarray(yr[red_key])[0]) if red_key in yr else float("nan")
            c = float(np.asarray(yc[red_key])[0]) if red_key in yc else float("nan")
            rel = abs(f - r) / max(abs(f), 1e-12) if f is not None and math.isfinite(r) else float("nan")
            rows.append({"output": full_key, "full": f, "reduced": r, "closed_form": c, "rel_err": rel,
                         "kind": res.compare_kind.get(full_key, "agree")})
    return Verification(True, fid.label, "", res, rows)


# ============================================================================================ assessment
@dataclass
class Assessment:
    trust: Any                         # pinneapple_analysis.trust.TrustReport
    verdict: str                       # "will work" | "likely to work" | "marginal" | "will not work" | "unverified"
    probability: float
    ci: tuple[float, float]
    assumptions: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        t = self.trust.to_dict()
        return {"verdict": self.verdict, "probability": self.probability, "ci95": list(self.ci),
                "assumptions": self.assumptions, "reasons": self.reasons,
                "trust": {"decision": t["decisao"], "score": t["score"], "coverage": t["cobertura"],
                          "checks": t["checagens"]}}


def assess(spec: RequirementSpec, template: ProblemTemplate, cand: Candidate, rob: Robustness,
           ver: Verification, *, agree_tol: float = 0.10) -> Assessment:
    from pinneapple_analysis.trust.trust_report import (
        CONTRADICTS,
        NOT_RUN,
        SUPPORTS,
        Check,
        TrustReport,
    )

    checks = []
    missing = [q.name for q in spec.values.values() if q.provenance == MISSING and
               template.slot_map()[q.name].critical]
    checks.append(Check("requirements complete", CONTRADICTS if missing else SUPPORTS, critical=True,
                        reason=", ".join(missing) or "every critical value stated, answered or defaulted with a "
                                                       "source"))
    for f in template.failure_modes:
        m = cand.outputs.get(f.name, float("nan"))
        checks.append(Check(f"margin: {f.label}", SUPPORTS if m >= 0 else CONTRADICTS, critical=f.critical,
                            value=m, limit=0.0, source="reduced model", reason=f.description))
    cf_rows = [(k, cand.closed_form.get(k), cand.outputs.get(k)) for k in cand.closed_form
               if k in cand.outputs and k in {f.name for f in template.failure_modes}]
    for k, c, r in cf_rows:
        # margins compared as capacity / demand ratios (1 + margin)
        e = abs((1 + c) - (1 + r)) / max(abs(1 + r), 1e-9)
        checks.append(Check(f"closed form agrees with reduced model: {k}", SUPPORTS if e <= 0.25 else CONTRADICTS,
                            weight=0.5, value=e, limit=0.25, source="closed form vs reduced",
                            reason="screening model within 25 % of the reduced model on (1 + margin)"))
    if ver.ran and ver.result is not None:
        for row in ver.comparison:
            e, kind = row["rel_err"], row.get("kind", "agree")
            if kind == "agree":
                checks.append(Check(f"full solver agrees: {row['output']}",
                                    SUPPORTS if math.isfinite(e) and e <= agree_tol else CONTRADICTS, critical=True,
                                    value=e, limit=agree_tol, source=ver.label,
                                    reason=f"reduced {row['reduced']:.4g} vs full {row['full']:.4g}"))
            else:
                ok = row["full"] >= row["reduced"] if kind == "at_least" else row["full"] <= row["reduced"]
                word = ">=" if kind == "at_least" else "<="
                checks.append(Check(f"full solver within the envelope: {row['output']}",
                                    SUPPORTS if ok else CONTRADICTS, critical=True, value=row["full"],
                                    limit=row["reduced"], source=ver.label,
                                    reason=f"full {row['full']:.4g} {word} reduced model at the worst-case envelope "
                                           f"{row['reduced']:.4g}"))
        for c in ver.result.checks:
            checks.append(Check(f"solver check: {c['name']}", SUPPORTS if c["passed"] else CONTRADICTS,
                                critical=bool(c.get("critical", False)), value=c.get("value"), limit=c.get("limit"),
                                source=ver.label, reason=c.get("detail", "")))
        for k, v in ver.result.outputs.items():
            if k.startswith("margin") and v < 0:
                checks.append(Check(f"full-solver margin: {k}", CONTRADICTS, critical=True, value=v, limit=0.0,
                                    source=ver.label))
    else:
        checks.append(Check("full-fidelity verification", NOT_RUN, critical=False,
                            reason=ver.reason or "not run"))
    checks.append(Check("robustness", SUPPORTS if rob.ci[0] >= template.robust_target else CONTRADICTS,
                        value=rob.p_success, limit=template.robust_target, source=f"Monte Carlo ({rob.n} samples)",
                        reason=f"lower 95 % bound {rob.ci[0]:.3f} vs target {template.robust_target}"))
    tr = TrustReport(checks, model=f"{template.name} design")
    reasons = [f"{c.name}: {c.reason}" if c.reason else c.name for c in checks if c.status != SUPPORTS]
    p, (lo, hi) = rob.p_success, rob.ci
    critical_bad = any(c.status == CONTRADICTS and c.critical for c in checks)
    if critical_bad:
        verdict = "will not work" if any(c.name.startswith(("margin", "full-solver margin")) and
                                         c.status == CONTRADICTS for c in checks) else "unverified"
    elif not ver.ran:
        verdict = "unverified" if lo < template.robust_target else "likely to work"
    elif lo >= template.robust_target:
        verdict = "will work"
    elif p >= 0.5:
        verdict = "marginal"
    else:
        verdict = "will not work"
    assumptions = list(template.assumptions) + [f"{q.name} = {q.display()} ({q.source})"
                                                for q in spec.values.values() if q.provenance == "defaulted"]
    return Assessment(tr, verdict, p, (lo, hi), assumptions, reasons)
