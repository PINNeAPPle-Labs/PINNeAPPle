"""``run_study(spec, evaluator)``: baseline -> DOE -> surrogate -> optimiser -> verification of the chosen designs.

The module owns the *study*: how the budget splits, what is measured, what the surrogate may be trusted with, and
what is signed. It owns no *physics*: the evaluator is injected, so the same run drives a catalog tool, an OpenFOAM
case, an analytic model or a test function, and nothing here executes anything external.

The cycle, and what each step is allowed to say
-----------------------------------------------
1. **Baseline.** The reference design (:meth:`DesignStudy.baseline_design`) is evaluated first. Its outputs are the
   only thing that can turn a limit relative to the baseline into a number, so nothing is called feasible before it.
   If the baseline cannot be evaluated the study stops there (status ``refused``): no absolute limit and no baseline
   to improve on.
2. **DOE.** Latin hypercube over the bounded variables (``DesignSpace.sample``), on the reference evaluator, within
   the reference budget. Designs the evaluator refuses are recorded with the reason, never dropped.
3. **Surrogate.** Every declared family of ``pinneapple_autolab.surrogates`` is cross-validated on the measured
   points (``compare_models``); the family with the lowest held-out RMSE per output is refitted on all of them. The
   held-out error is measured in a unit-free way (nRMSE = RMSE / standard deviation of the measured outputs), so the
   verifier's target means the same thing for a drag in N and a temperature in K.
4. **Optimiser.** NSGA-II (``pinneapple_design.design_optimizer.methods.run``) on the surrogate, starting from the
   measured designs, inside the validated domain (:class:`pinneapple_autolab.domain.ValidatedDomain`): a predicted
   design too far from anything measured is infeasible for the optimiser, so the front cannot walk off the evidence.
   The limits it searches against are tightened by the surrogate's own held-out error, because an optimum sits *on*
   a limit and a margin thinner than the model's error is a coin toss (the answer itself is judged against the limits
   as written). These evaluations are cheap and are budgeted separately from the reference ones.
5. **Verification.** The designs that would be recommended are run on the reference evaluator, with the prediction
   recorded *before* the run. The study's answer is a measured design; a design that was only predicted says so.

The result (:class:`StudyRun`) carries the Pareto front (each member flagged measured or predicted), every rejected
design with the reason it was rejected (the evaluator refused it, a limit is violated, or it lies outside the
validated domain), the trust verdict (:class:`pinneapple_analysis.trust.TrustReport`) and the validated domain. The
status is ``done`` only when the trust verdict is ``APPROVED``; ``inconclusive`` means the run happened but a critical
check could not run or failed, and ``refused`` means nothing could be measured. There is no third way to say "fine".

The evaluator
-------------
Injected, in any of three shapes, all called with the design in the *tool's* input names (the spec's mapping plus
the declared conditions):

* a callable ``f(params) -> {"out": value, ...}``; it may also return ``"__fields__"``, ``"__cost_s__"``, or
  ``{"ok": False, "message": ...}``; raising is a rejection with the exception's message;
* an object with ``run(params, workdir) -> SimOutcome`` and ``check(params) -> list[str]`` — a
  :class:`pinneapple_autolab.simulator.Simulator`, whose checks are run before the solver is called;
* an object with an ``evaluate(params)`` method (the same contract as the callable).

Dependencies: numpy at import time; the surrogate board (scikit-learn), the validated domain and the optimiser of
``pinneapple_autolab`` / ``pinneapple_design`` are imported where they are used, so importing this module or validating
a spec pays for none of them. The surrogate step needs scikit-learn (the ``[timeseries]`` extra) and says so, instead
of falling back to a straight line.

    >>> run = run_study(spec, my_tool)                       # doctest: +SKIP
    >>> run.status, run.trust["decision"]                    # doctest: +SKIP
    ('done', 'APPROVED')
    >>> run.pareto[0]["verified"], run.pareto[0]["objectives"]   # doctest: +SKIP
    (True, {'cd': 0.0183, 'cl': 0.512})
"""
from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .spec import KINDS, DesignStudy

__all__ = ["run_study", "StudyRun", "StudyRefused", "RUNNABLE_KINDS"]

#: The kinds this module runs today. The others are described by the schema and refused with the reason that is
#: missing, instead of being silently treated as an optimisation.
RUNNABLE_KINDS = ("optimize", "surrogate")

_VERIFY_MIN_DISTANCE = 0.05          # designs closer than this in the unit cube are the same design
_FRONT_MIN_DISTANCE = 0.01           # the same, for merging near-identical members of the predicted front
_MAX_FRONT_LISTED = 40               # the front is reported at most this long (an even spread, ends kept)
_MAX_REJECTED_LISTED = 40


class StudyRefused(ValueError):
    """The study cannot run as asked: an unsupported kind, a missing dependency, a budget that does not split."""


# --------------------------------------------------------------------------- the injected evaluator
@dataclass
class _Outcome:
    outputs: dict[str, float]
    ok: bool = True
    message: str = ""
    wall_s: float = 0.0
    fields: dict[str, np.ndarray] | None = None


class _Tool:
    """Calls the injected evaluator in whichever of the three shapes it has, and never invents an output."""

    def __init__(self, evaluator: Any, spec: DesignStudy):
        if not callable(evaluator) and not hasattr(evaluator, "run") and not hasattr(evaluator, "evaluate"):
            raise TypeError("the evaluator must be callable, or have run(params, workdir) (a Simulator), or have "
                            f"evaluate(params); got {type(evaluator).__name__}")
        self.fn, self.spec = evaluator, spec
        self.outputs = spec.evaluator.output_names

    def params(self, design: Mapping[str, float]) -> dict[str, Any]:
        return self.spec.evaluator.tool_params(design)

    def check(self, design: Mapping[str, float]) -> list[str]:
        """Why the evaluator refuses this design before running it (empty when it does not say)."""
        if hasattr(self.fn, "check"):
            return [str(r) for r in (self.fn.check(self.params(design)) or [])]
        return []

    def run(self, design: Mapping[str, float], workdir: str) -> _Outcome:
        p = self.params(design)
        t0 = time.perf_counter()
        try:
            res = self.fn.run(p, workdir) if hasattr(self.fn, "run") else (
                self.fn.evaluate(p) if hasattr(self.fn, "evaluate") else self.fn(p))
        except Exception as e:  # noqa: BLE001 - a design the tool cannot evaluate is a rejection, not a crash
            return _Outcome({}, False, f"{type(e).__name__}: {e}", time.perf_counter() - t0)
        wall = time.perf_counter() - t0
        fields = getattr(res, "fields", None)
        if hasattr(res, "ok"):                                   # an autolab SimOutcome
            wall = float(getattr(res, "cost_s", 0.0) or wall)
            if not res.ok:
                return _Outcome({}, False, str(getattr(res, "message", "") or "the evaluator failed"), wall, fields)
            res = res.outputs
        if not isinstance(res, Mapping):
            return _Outcome({}, False, f"the evaluator returned a {type(res).__name__}, not a mapping", wall)
        res = dict(res)
        if res.get("ok") is False:
            return _Outcome({}, False, str(res.get("message") or "the evaluator failed"), wall, fields)
        wall = float(res.pop("__cost_s__", wall) or wall)
        fields = res.pop("__fields__", fields)
        missing = [o for o in self.outputs if o not in res]
        if missing:
            return _Outcome({}, False, f"the evaluator did not return {missing}", wall, fields)
        try:
            vals = {o: float(res[o]) for o in self.outputs}
        except (TypeError, ValueError) as e:
            return _Outcome({}, False, f"an output is not a number: {e}", wall, fields)
        bad = sorted(k for k, v in vals.items() if not np.isfinite(v))
        if bad:
            return _Outcome({}, False, f"non-finite outputs {bad}", wall, fields)
        return _Outcome(vals, True, "", wall, {k: np.asarray(v, float) for k, v in (fields or {}).items()})


# --------------------------------------------------------------------------- the run
@dataclass
class StudyRun:
    """What a study produced, with every number traceable to an evaluation.

    ``rows`` are all reference evaluations (baseline, DOE, verifications) in the order they happened; ``pareto`` the
    answer (measured designs first, then the ones the surrogate recommends and the reference has not confirmed);
    ``rejected`` the designs that were refused or violated something, each with its reason; ``trust`` the verdict;
    ``domain`` the region of the design space the measured designs cover.
    """

    spec: DesignStudy
    status: str                                   # done | inconclusive | refused
    reasons: list[str] = field(default_factory=list)
    created: str = ""
    baseline: dict[str, float] = field(default_factory=dict)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    pareto: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    surrogate: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    trust: dict[str, Any] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def decision(self) -> str:
        """The trust verdict (``APPROVED`` / ``REVIEW`` / ``REJECT``), empty when the study did not run."""
        return str(self.trust.get("decision", ""))

    @property
    def approved(self) -> bool:
        return self.status == "done" and self.decision == "APPROVED"

    @property
    def measured(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["ok"]]

    def summary(self) -> dict[str, Any]:
        p = self.spec.budget.planned(len(self.spec.base.variables))
        return {"status": self.status, "decision": self.decision, "kind": self.spec.kind, "title": self.spec.title,
                "digest": self.spec.digest(), "evaluations": {"used": len(self.rows), **p,
                                                              "max": self.spec.budget.max_evaluations},
                "n_pareto": len(self.pareto), "n_pareto_verified": sum(1 for d in self.pareto if d["verified"]),
                "n_rejected": len(self.rejected), "reasons": list(self.reasons), "wall_s": round(self.wall_s, 3)}

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "summary": self.summary(), "status": self.status,
                "reasons": list(self.reasons), "created": self.created, "baseline": dict(self.baseline),
                "constraints": list(self.constraints), "rows": list(self.rows), "pareto": list(self.pareto),
                "rejected": list(self.rejected), "surrogate": dict(self.surrogate),
                "optimizer": dict(self.optimizer), "trust": dict(self.trust), "domain": dict(self.domain),
                "budget": dict(self.budget), "notes": list(self.notes), "wall_s": self.wall_s}

    def to_markdown(self) -> str:
        """The study as a page a reviewer reads: what was measured, what is on the front, what was rejected, why."""
        f = lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4g}"  # noqa: E731
        u = self.spec.units()
        s = self.summary()
        plan = s["evaluations"]
        out = [f"# Design study: {self.spec.title}", "",
               f"{self.spec.kind} study, spec `{self.spec.digest()[:16]}`, created {self.created}. "
               f"Status **{self.status}**, trust **{self.decision or '—'}**"
               + (f" (score {f(self.trust.get('score'))}, coverage {100 * (self.trust.get('coverage') or 0):.0f} %)"
                  if self.trust else "") + ".", "",
               f"Evaluator `{self.spec.evaluator.tool}` ({self.spec.evaluator.fidelity}); budget "
               f"{plan['used']} of {plan['max']} reference evaluations (1 baseline, {plan['doe']} DOE, "
               f"{plan['verify']} verifications), {self.spec.budget.surrogate_evaluations} on the surrogate.", ""]
        if self.baseline:
            out += ["## Baseline", ""] + [f"- {k} = {f(v)} {u.get(k, '')}" for k, v in self.baseline.items()] + [""]
        if self.constraints:
            out += ["## Limits", ""]
            for c in self.constraints:
                out.append(f"- {c['text']}" + (" (relative to the baseline)" if c["relative"] else ""))
            out.append("")
        out += ["## Pareto front", "",
                "| design | " + " | ".join(c["label"] or c["output"] for c in
                                           [o | {"label": o.get("label", "")}
                                            for o in (x.to_dict() for x in self.spec.objectives)])
                + " | measured |", "|---" * (len(self.spec.objectives) + 2) + "|"]
        for d in self.pareto:
            out.append("| " + ", ".join(f"{k} {f(v)}" for k, v in d["params"].items()) + " | "
                       + " | ".join(f(d["objectives"].get(o.output)) for o in self.spec.objectives) + " | "
                       + ("yes" if d["verified"] else "predicted") + " |")
        if self.surrogate:
            out += ["", "## Surrogate", ""]
            for out_name, row in self.surrogate.get("outputs", {}).items():
                out.append(f"- {out_name}: {row['family']}, held-out nRMSE {f(row['nrmse'])} "
                           f"(target {self.spec.verifier.surrogate_target:g}), R² {f(row['r2'])}, "
                           f"95 % coverage {100 * row['coverage95']:.0f} %")
        out += ["", "## Trust checks", ""]
        for c in self.trust.get("checks", []):
            out.append(f"- {c['status']}: {c['name']} (value {f(c['value'])}"
                       + (f", limit {f(c['limit'])}" if c.get("limit") is not None else "") + ")"
                       + (f" — {c['reason']}" if c.get("reason") else ""))
        if self.rejected:
            out += ["", f"## Rejected designs ({len(self.rejected)})", ""]
            for r in self.rejected:
                out.append(f"- {'{'}" + ", ".join(f"{k} {f(v)}" for k, v in r["params"].items()) + "}: "
                           + "; ".join(r["reasons"]) + f" [{r['source']}]")
        if self.notes or self.reasons:
            out += ["", "## Why", ""] + [f"- {n}" for n in self.reasons + self.notes]
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- helpers
def _refused(spec: DesignStudy, reasons: Sequence[str], status: str = "refused", **kw) -> StudyRun:
    return StudyRun(spec=spec, status=status, reasons=list(reasons),
                    created=datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw)


def _space(spec: DesignStudy):
    from pinneapple_design.design_optimizer.study import DesignSpace, Variable
    return DesignSpace([Variable(name=v.name, lower=v.lower, upper=v.upper, kind=v.kind, unit=v.unit,
                                 description=v.description) for v in spec.base.variables])


def _problem(spec: DesignStudy, constraints, evaluator):
    from pinneapple_design.design_optimizer.study import DesignProblem, Objective
    objectives = [Objective(o.output, o.sense, unit=spec.evaluator.unit_of(o.output), label=o.label)
                  for o in spec.objectives]
    return DesignProblem(spec.title, _space(spec), objectives, list(constraints), evaluator,
                         description=spec.statement)


def _trust_dict(rep) -> dict[str, Any]:
    return {"decision": rep.decisao, "score": rep.score, "coverage": rep.cobertura, "reasons": rep.motivos,
            "checks": [{"name": c.name, "status": c.status, "value": c.value, "limit": c.limit,
                        "critical": c.critical, "weight": c.weight, "source": c.source, "reason": c.reason}
                       for c in rep.checks],
            "rules": "any critical check failing -> REJECT; any other failing, coverage below the minimum or the "
                     "score below it -> REVIEW; else APPROVED (pinneapple_analysis.trust.TrustReport)"}


def _relative_error(pred: float, true: float) -> float:
    return abs(pred - true) / max(abs(true), 1e-12)


def run_study(spec: DesignStudy | Mapping, evaluator: Any, *, progress: Callable[[dict], None] | None = None,
              workdir: str | None = None) -> StudyRun:
    """Run one study. See the module docstring for the cycle; ``progress`` is called with a dict after each step
    (``{"event", "done", "planned", "design", "outputs"}``) and ``workdir`` is the scratch folder handed to a
    Simulator-shaped evaluator (a temporary one is created and removed when it is not given)."""
    spec = spec if isinstance(spec, DesignStudy) else DesignStudy.from_dict(spec)
    if spec.kind not in RUNNABLE_KINDS:
        return _refused(spec, [f"a {spec.kind!r} study is described by the schema "
                               f"({', '.join(KINDS)}) but not run yet: this module runs {', '.join(RUNNABLE_KINDS)}. "
                               "What is missing is named in the issue (#1036), item by item."])
    tell = progress or (lambda event: None)
    started = time.perf_counter()
    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="pinneapple-design-study-")
    try:
        return _run(spec, _Tool(evaluator, spec), tell, workdir, started)
    finally:
        if own_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def _run(spec: DesignStudy, tool: _Tool, tell: Callable[[dict], None], workdir: str,
         started: float) -> StudyRun:
    names = spec.base.names
    plan = spec.budget.planned(len(names))
    space = _space(spec)
    units = spec.units()
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def evaluate(design: Mapping[str, float], tag: str, predicted: Mapping[str, float] | None = None,
                 predicted_std: Mapping[str, float] | None = None) -> dict[str, Any]:
        issues = tool.check(design)
        if issues:
            out = _Outcome({}, False, "; ".join(issues))
        else:
            out = tool.run(design, workdir)
        row = {"index": len(rows), "tag": tag, "params": {k: float(design[k]) for k in names},
               "outputs": dict(out.outputs), "ok": bool(out.ok), "message": out.message,
               "wall_s": round(float(out.wall_s), 4),
               "predicted": None if predicted is None else {k: float(v) for k, v in predicted.items()},
               "predicted_std": None if predicted_std is None else {k: float(v) for k, v in predicted_std.items()}}
        rows.append(row)
        tell({"event": tag, "done": len(rows), "planned": spec.budget.max_evaluations, "design": dict(row["params"]),
              "ok": row["ok"], "outputs": dict(row["outputs"])})
        return row

    # ------------------------------------------------------------------ 1. baseline
    baseline_design = spec.baseline_design()
    base = evaluate(baseline_design, "baseline")
    if not base["ok"]:
        return _refused(spec, [f"the baseline design {baseline_design} could not be evaluated: {base['message']}. "
                               "A study starts from a measured baseline: without it no limit relative to the "
                               "baseline is a number and there is nothing to improve on"], "refused",
                        rows=rows, wall_s=time.perf_counter() - started)
    baseline = dict(base["outputs"])
    notes.append(f"Baseline measured on `{spec.evaluator.tool}`: "
                 + ", ".join(f"{k} = {v:.6g} {units.get(k, '')}" for k, v in baseline.items()) + ".")
    try:
        constraints = [lim.resolved(baseline) for lim in spec.constraints]
    except Exception as e:  # noqa: BLE001 - a limit that cannot be resolved stops the study with the reason
        return _refused(spec, [f"a limit could not be resolved against the baseline: {type(e).__name__}: {e}"],
                        "inconclusive", rows=rows, baseline=baseline, wall_s=time.perf_counter() - started)

    # ------------------------------------------------------------------ 2. DOE
    n_doe = min(plan["doe"], spec.budget.max_evaluations - 1)
    X_doe = space.sample(n_doe, method="lhs", seed=spec.budget.seed) if n_doe else np.zeros((0, len(names)))
    seen = {tuple(np.round([baseline_design[k] for k in names], 9))}
    for x in X_doe:
        d = space.as_dicts(x)[0]
        key = tuple(np.round([d[k] for k in names], 9))
        if key in seen:
            continue
        seen.add(key)
        if len(rows) >= spec.budget.max_evaluations - plan["verify"]:
            notes.append(f"The reference budget stopped the DOE at {len(rows) - 1} of {n_doe} points.")
            break
        evaluate(d, "doe")

    ok_rows = [r for r in rows if r["ok"]]
    if len(ok_rows) < 3:
        return _refused(spec, [f"only {len(ok_rows)} design(s) could be evaluated (the baseline and the DOE); a "
                               "surrogate needs at least 3"], "inconclusive", rows=rows, baseline=baseline,
                        wall_s=time.perf_counter() - started)
    targets = spec.outputs or spec.evaluator.output_names
    X = np.array([[r["params"][k] for k in names] for r in ok_rows], float)
    Y = {k: np.array([r["outputs"][k] for r in ok_rows], float) for k in targets}
    U = space.to_unit(X)

    # ------------------------------------------------------------------ 3. surrogate
    try:
        from pinneapple_autolab.surrogates import compare_models
        board = compare_models(U, Y, families=list(spec.model.families), k=spec.verifier.folds,
                               seed=spec.budget.seed)
    except ImportError as e:
        return _refused(spec, [f"the surrogate step needs scikit-learn (pinneapple_autolab.surrogates): {e}. "
                               "Install the extra (pip install 'pinneapple[timeseries]') or hand the study a "
                               "different set of families"], "refused", rows=rows, baseline=baseline,
                        wall_s=time.perf_counter() - started)
    tell({"event": "surrogate", "done": len(rows), "planned": spec.budget.max_evaluations, "design": None,
          "ok": None, "outputs": {}, "chosen": dict(board.chosen)})
    surrogate = {"families": list(spec.model.families), "n_train": board.n_train, "folds": spec.verifier.folds,
                 "chosen": dict(board.chosen), "outputs": {}, "table": {k: v for k, v in board.table.items()}}
    for out in targets:
        fam = board.chosen[out]
        m = board.table[out][fam]
        sd = float(np.std(Y[out], ddof=1)) if len(Y[out]) > 1 else 0.0
        surrogate["outputs"][out] = {"family": fam, "unit": units.get(out, ""), "nrmse": float(m["rmse"] / sd)
                                     if sd > 0 else float("inf"), "rmse": float(m["rmse"]), "r2": float(m["r2"]),
                                     "coverage95": float(m["coverage95"]), "rel_mae": float(m["rel_mae"]),
                                     "sd": sd}

    from pinneapple_autolab.domain import ValidatedDomain
    from pinneapple_design.design_optimizer.study import VALID, FunctionEvaluator
    # bounds so the verdicts speak in physical units ("taper_ratio = 0.30 is below the simulated range")
    domain = ValidatedDomain(U, names, bounds=space.bounds)
    domain_info = {"n_simulated": int(len(U)), "spacing": float(domain.spacing),
                   "rule": f"inside the simulated ranges (widened by {domain.margin:g}) and no farther than "
                           f"{domain.max_ratio:g} x the typical spacing from the nearest measured design"}

    def predict(Xq: np.ndarray) -> dict[str, np.ndarray]:
        Uq = space.to_unit(np.atleast_2d(Xq))
        means = {k: v[0] for k, v in board.predict(Uq).items()}
        means[VALID] = np.array([v.inside for v in domain.assess(Uq)], dtype=bool)
        return means

    surrogate_evaluator = FunctionEvaluator(predict, targets, name="surrogate", fidelity="surrogate",
                                            description="the study's cross-validated surrogate")
    # An optimum sits *on* a limit, and there the surrogate's own error decides feasibility: a design whose predicted
    # margin is thinner than the model's held-out error is a coin toss, not a feasible design. So the surrogate
    # optimises against the limits tightened by that error, while the answer is judged against the real ones.
    tightened, margins = [], {}
    for c in constraints:
        rel = max(surrogate["outputs"].get(c.name, {}).get("nrmse", 0.0) or 0.0, 0.0)
        margins[c.name] = rel
        up = None if c.upper is None else c.upper - abs(c.upper) * rel
        lo = None if c.lower is None else c.lower + abs(c.lower) * rel
        if (up is not None and lo is not None and up <= lo) or rel == 0:
            tightened.append(c)
        else:
            tightened.append(type(c)(c.name, upper=up, lower=lo, unit=c.unit, kind=c.kind, label=c.label))
    problem = _problem(spec, constraints, surrogate_evaluator)                  # judges the measured designs
    optimizer_problem = _problem(spec, tightened, surrogate_evaluator)          # the surrogate searches inside
    if any(m > 0 for m in margins.values()):
        notes.append("The surrogate searched against the limits tightened by its own held-out error ("
                     + ", ".join(f"{k} {100 * v:.1f} %" for k, v in margins.items() if v > 0)
                     + "): a margin thinner than the model's error is not a margin. The answer is judged against "
                       "the limits as written.")

    pareto: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    optimizer: dict[str, Any] = {}
    verified_designs: list[dict[str, Any]] = []
    ev = None

    def measured_arrays():
        rs = [r for r in rows if r["ok"]]
        Xm = np.array([[r["params"][k] for k in names] for r in rs], float)
        Ym = {k: np.array([r["outputs"][k] for r in rs], float) for k in spec.evaluator.output_names}
        Ym[VALID] = np.ones(len(rs), bool)
        return rs, Xm, Ym

    # ------------------------------------------------------------------ 4. optimiser
    if spec.kind == "optimize":
        from pinneapple_design.design_optimizer import methods as optimizers
        result = optimizers.run(optimizer_problem, "nsga2", budget=spec.budget.surrogate_evaluations, seed=spec.budget.seed,
                                initial=X)
        optimizer = result.summary() | {"n_initial": int(len(X))}
        front = result.pareto()
        ev = result.evaluations
        # designs the surrogate could reach but the evidence does not cover: rejected with that reason
        if ev is not None and len(ev):
            feasible = np.all(ev.G <= 0, 1) & np.all(np.isfinite(ev.F), 1)
            valid = np.asarray(ev.Y.get(VALID, np.ones(len(ev), bool)), dtype=bool)
            outside = np.where(feasible & ~valid)[0]
            if len(outside):
                verdicts = domain.assess(space.to_unit(ev.X[outside]))
                for i, v in zip(outside[:_MAX_REJECTED_LISTED], verdicts[:_MAX_REJECTED_LISTED], strict=True):
                    rejected.append({"params": space.as_dicts(ev.X[i])[0], "source": "domain",
                                     "outputs": {k: float(ev.Y[k][i]) for k in targets},
                                     "reasons": v.reasons or ["outside the validated domain"]})
                if len(outside) > _MAX_REJECTED_LISTED:
                    rejected.append({"params": {}, "source": "domain", "outputs": {},
                                     "reasons": [f"{len(outside) - _MAX_REJECTED_LISTED} more predicted designs "
                                                 "were outside the validated domain"]})

        # ------------------------------------------------------------------ 5. verification
        order = list(front) if len(front) else np.array([i for i in [result.best(0)] if i is not None], int)
        chosen: list[int] = []
        for i in order:
            u = space.to_unit(ev.X[i:i + 1])[0]
            if all(np.linalg.norm(u - space.to_unit(ev.X[j:j + 1])[0]) > _VERIFY_MIN_DISTANCE for j in chosen):
                chosen.append(int(i))
            if len(chosen) >= plan["verify"]:
                break
        for i in chosen:
            d = space.as_dicts(ev.X[i])[0]
            if len(rows) >= spec.budget.max_evaluations:
                rejected.append({"params": d, "source": "budget", "outputs": {},
                                 "reasons": ["the reference budget was spent before this design could be measured"]})
                continue
            mean = {k: float(ev.Y[k][i]) for k in targets}
            std = {}
            for k in targets:
                pred = board.predict(space.to_unit(ev.X[i:i + 1])).get(k)
                std[k] = float(pred[1][0]) if pred is not None else float("nan")
            row = evaluate(d, "verification", predicted=mean, predicted_std=std)
            if row["ok"]:
                verified_designs.append(row)
    tell({"event": "optimize" if spec.kind == "optimize" else "measured", "done": len(rows),
          "planned": spec.budget.max_evaluations, "design": None, "ok": None, "outputs": {}})

    # ------------------------------------------------------------------ measured answer
    _, Xm, Ym = measured_arrays()
    measured = problem.record(Xm, Ym, fidelity="reference")
    from pinneapple_design.design_optimizer.study import feasibility_report
    front_measured = measured.pareto()
    verified_keys = {tuple(np.round([r["params"][k] for k in names], 9)) for r in verified_designs}
    for i in front_measured:
        d = space.as_dicts(measured.X[i])[0]
        key = tuple(np.round(measured.X[i], 9))
        pareto.append({"params": d, "objectives": {o.output: float(measured.Y[o.output][i]) for o in spec.objectives},
                       "outputs": {k: float(measured.Y[k][i]) for k in measured.Y if k != VALID},
                       "verified": key in verified_keys, "predicted": None, "violation": 0.0})
    if spec.kind == "optimize" and ev is not None and len(front):
        # near-identical members of the surrogate's front are one design to the precision that matters here
        kept = [space.to_unit(measured.X[i:i + 1])[0] for i in front_measured]
        for i in front:
            d = space.as_dicts(ev.X[i])[0]
            u = space.to_unit(ev.X[i:i + 1])[0]
            if any(np.linalg.norm(u - k) < _FRONT_MIN_DISTANCE for k in kept):
                continue
            kept.append(u)
            pareto.append({"params": d, "objectives": {o.output: float(ev.Y[o.output][i])
                                                       for o in spec.objectives},
                           "outputs": {k: float(ev.Y[k][i]) for k in targets}, "verified": False,
                           "predicted": True, "violation": 0.0})
        notes.append(f"The predicted members of the front are merged to {_FRONT_MIN_DISTANCE:g} of the range: "
                     "designs closer than that are the same design.")
        n_front = len(pareto)
        if n_front > _MAX_FRONT_LISTED:
            measured_part = [p for p in pareto if p["predicted"] is None]
            predicted_part = [p for p in pareto if p["predicted"] is not None]
            room = max(_MAX_FRONT_LISTED - len(measured_part), 1)
            if len(predicted_part) > room:
                keep = np.unique(np.linspace(0, len(predicted_part) - 1, room).astype(int))
                pareto = measured_part + [predicted_part[i] for i in keep]
            notes.append(f"The front is listed at {len(pareto)} of its {n_front} designs: every measured member, and "
                         "an even spread of the predicted ones, ends kept.")

    # measured designs that violate a limit, with the limit that they violate
    viol = np.where(~measured.feasible)[0]
    if len(viol):
        report = feasibility_report(problem, measured, viol)
        for row_r, i in zip(report, viol, strict=True):
            reasons = []
            for c in row_r["constraints"]:
                if c["status"] == "violated":
                    reasons.append(f"{c['rule']} — {c['value']:.6g} {c['unit']}".strip()
                                   + (f", violated by {abs(c['margin']):.3g}" if c["margin"] is not None else ""))
            rejected.append({"params": space.as_dicts(measured.X[i])[0], "source": "constraint",
                             "outputs": {k: float(measured.Y[k][i]) for k in measured.Y if k != VALID},
                             "reasons": reasons or ["outside the evaluator's validity range"]})
    for r in rows:
        if not r["ok"]:
            rejected.append({"params": dict(r["params"]), "source": "evaluator", "outputs": {},
                             "reasons": [r["message"] or "the evaluator refused the design"]})

    # ------------------------------------------------------------------ domain and trust
    domain_info["verified"] = []
    for r in verified_designs:
        v = domain.assess(space.to_unit(np.array([[r["params"][k] for k in names]])))[0]
        domain_info["verified"].append({"params": dict(r["params"]), "inside": bool(v.inside),
                                        "ratio": float(v.ratio), "reasons": v.reasons})
    domain_info["outside_simulated_ranges"] = {n: [float(domain.lo[i]), float(domain.hi[i])]
                                               for i, n in enumerate(names)}

    trust = _trust(spec, surrogate, targets, pareto, verified_designs, domain_info, constraints)
    decision = trust.decisao
    status = "done" if decision == "APPROVED" else "inconclusive"
    reasons = list(trust.motivos) if status != "done" else []
    if spec.kind == "surrogate":
        notes.append("A surrogate study measures and fits; it does not optimise, so its front is the measured "
                     "front and nothing had to be verified against a prediction.")
    if not verified_designs and spec.kind == "optimize":
        notes.append("No design was re-measured: the front is the surrogate's, and says so.")
    return StudyRun(spec=spec, status=status, reasons=reasons, created=created, baseline=baseline,
                    constraints=[{"output": c.name, "lower": c.lower, "upper": c.upper, "unit": c.unit,
                                  "kind": c.kind, "label": c.label, "text": c.describe(),
                                  "relative": any(lim.relative for lim in spec.constraints
                                                  if lim.output == c.name)} for c in constraints],
                    rows=rows, pareto=pareto, rejected=rejected, surrogate=surrogate, optimizer=optimizer,
                    trust=_trust_dict(trust), domain=domain_info,
                    budget={"planned": plan, "max_evaluations": spec.budget.max_evaluations,
                            "used": len(rows),
                            "wall_s": round(sum(r["wall_s"] for r in rows), 4),
                            "surrogate_evaluations": spec.budget.surrogate_evaluations},
                    notes=notes, wall_s=time.perf_counter() - started)


def _trust(spec: DesignStudy, surrogate, targets, pareto, verified, domain_info, constraints):
    """The verdict: every check below ran on data the surrogate did not train on, and none is assumed."""
    from pinneapple_analysis.trust.trust_report import (
        CONTRADICTS,
        NOT_RUN,
        SUPPORTS,
        Check,
        TrustReport,
    )
    checks: list[Check] = []
    tol = spec.verifier
    for out in targets:
        row = surrogate["outputs"][out]
        checks.append(Check(f"surrogate held-out nRMSE of {out} <= {tol.surrogate_target:g}",
                            SUPPORTS if row["nrmse"] <= tol.surrogate_target else CONTRADICTS,
                            value=row["nrmse"], limit=tol.surrogate_target, critical=True,
                            source=f"{spec.verifier.folds}-fold cross-validation, family {row['family']}"))
    # A predictive interval is only claimed by the families that have one: the quadratic response surface's std is
    # the residual std of its own fit (homoscedastic), so its coverage says nothing about new designs. The tolerated
    # coverage is 80 %, not the nominal 95 %: on a few dozen held-out points one point is already several percent,
    # and an interval that misses one point in five is what cannot be called predictive.
    coverage_floor = 0.80
    predictive = [o for o in targets if surrogate["outputs"][o]["family"] in ("gp", "mlp_ensemble")]
    if predictive:
        worst = min(surrogate["outputs"][o]["coverage95"] for o in predictive)
        checks.append(Check(f"the surrogate's 95 % interval holds at least {100 * coverage_floor:.0f} % of the "
                            "held-out points", SUPPORTS if worst >= coverage_floor else CONTRADICTS, value=worst,
                            limit=coverage_floor, weight=0.5,
                            source="cross-validation of " + ", ".join(predictive)))
    else:
        families = ", ".join(sorted({row["family"] for row in surrogate["outputs"].values()}))
        checks.append(Check(f"the surrogate's 95 % interval holds at least {100 * coverage_floor:.0f} % of the "
                            "held-out points", NOT_RUN, weight=0.5,
                            reason=f"the chosen families ({families}) claim no predictive interval: a quadratic "
                                   "response surface's std is the residual std of its own fit"))
    if verified:
        worst_err, source = 0.0, ""
        for r in verified:
            for o in spec.objectives:
                e = _relative_error(r["predicted"][o.output], r["outputs"][o.output])
                if e > worst_err:
                    worst_err, source = e, f"{o.output} at {r['params']}"
        limit = max(tol.tolerance(o.output) for o in spec.objectives)
        checks.append(Check("the surrogate agrees with the reference on the designs it recommended",
                            SUPPORTS if worst_err <= limit else CONTRADICTS, value=worst_err, limit=limit,
                            critical=True, source=source))
        violated = [r for r in verified if not _satisfies(constraints, r["outputs"])]
        checks.append(Check("every re-measured design satisfies the limits", SUPPORTS if not violated
                            else CONTRADICTS, value=float(len(violated)), limit=0.0, critical=True,
                            source=f"{len(verified)} re-measured designs"))
        bad = [d for d in domain_info["verified"] if not d["inside"]]
        checks.append(Check("every re-measured design is inside the validated domain", SUPPORTS if not bad
                            else CONTRADICTS, value=float(len(bad)), limit=0.0, weight=0.5,
                            source="ValidatedDomain over the measured designs"))
        checks.append(Check(f"at least {tol.min_verified} design(s) measured, not predicted",
                            SUPPORTS if len(verified) >= tol.min_verified else CONTRADICTS,
                            value=float(len(verified)), limit=float(tol.min_verified), critical=True,
                            source="the reference evaluator"))
        on_front = [d for d in pareto if d["verified"]]
        checks.append(Check("the answer includes a measured design", SUPPORTS if on_front else CONTRADICTS,
                            value=float(len(on_front)), limit=1.0, critical=True, source="the Pareto front"))
    elif spec.kind == "optimize":
        checks.append(Check("at least one design measured, not predicted", NOT_RUN, critical=True,
                            reason="no design could be re-measured inside the budget"))
    front = [d for d in pareto if d["params"]]
    if spec.kind == "surrogate":
        checks.append(Check("every design on the front satisfies the limits",
                            SUPPORTS if all(_satisfies(constraints, d["outputs"]) for d in front) else CONTRADICTS,
                            value=float(sum(1 for d in front if not _satisfies(constraints, d["outputs"]))),
                            limit=0.0, critical=True, source=f"{len(front)} measured designs"))
        checks.append(Check("the surrogate agrees with the reference on the front", NOT_RUN, weight=0.5,
                            reason="a surrogate study does not optimise: there is no predicted design to check"))
    return TrustReport(checks, model=f"{spec.evaluator.tool} + {spec.model.families[0]}",
                       details={"spec": spec.digest(), "kind": spec.kind})


def _satisfies(constraints, outputs: Mapping[str, float]) -> bool:
    return all(bool(np.all(c.violation(np.array([outputs[c.name]])) <= 0)) for c in constraints)
