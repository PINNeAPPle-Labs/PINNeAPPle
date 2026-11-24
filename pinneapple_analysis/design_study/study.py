"""``run_study(spec, evaluator)``: baseline -> DOE -> surrogate -> optimiser -> verification of the chosen designs,
once, or in rounds while a :class:`.loop.StopRule` says the reference budget still buys something.

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

The loop, when a stop rule is given (#79)
-----------------------------------------
Passing ``stop=StopRule(...)`` turns the cycle into rounds: each round refits the surrogate on **everything measured
so far**, optimises on it, and measures what it recommends on the reference evaluator — so the search's share of the
budget is spent on evidence instead of being left over. :func:`.loop.decide` ends the loop on the first of four
numbers (the budget, a decision-grade surrogate, an objective that stopped moving, the round cap) and the reason,
with the value and the limit it compared, is :attr:`StudyRun.loop`. Without a stop rule the cycle runs exactly once,
which is the run this module has always done.

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
from .loop import StopRule, decide

__all__ = ["run_study", "StudyRun", "StudyRefused", "RUNNABLE_KINDS", "StopRule"]

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
    ``domain`` the region of the design space the measured designs cover; ``loop`` the rounds and the stop decision
    (``rule``, ``rounds`` and ``stop``: the reason, the value that decided it and the limit it was compared with).
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
    loop: dict[str, Any] = field(default_factory=dict)
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
    def stop(self) -> dict[str, Any]:
        """Why the loop stopped: ``reason`` (``budget``/``decision``/``no_improvement``/``max_rounds``/
        ``single_pass``), the ``round`` it stopped on and the ``value`` against the ``limit`` that decided it."""
        return dict(self.loop.get("stop") or {})

    @property
    def measured(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["ok"]]

    def summary(self) -> dict[str, Any]:
        p = self.spec.budget.planned(len(self.spec.base.variables))
        return {"status": self.status, "decision": self.decision, "kind": self.spec.kind, "title": self.spec.title,
                "digest": self.spec.digest(), "evaluations": {"used": len(self.rows), **p,
                                                              "max": self.spec.budget.max_evaluations},
                "n_pareto": len(self.pareto), "n_pareto_verified": sum(1 for d in self.pareto if d["verified"]),
                "n_rejected": len(self.rejected), "rounds": len(self.loop.get("rounds") or []),
                "stop": str(self.stop.get("reason", "")), "reasons": list(self.reasons),
                "wall_s": round(self.wall_s, 3)}

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "summary": self.summary(), "status": self.status,
                "reasons": list(self.reasons), "created": self.created, "baseline": dict(self.baseline),
                "constraints": list(self.constraints), "rows": list(self.rows), "pareto": list(self.pareto),
                "rejected": list(self.rejected), "surrogate": dict(self.surrogate),
                "optimizer": dict(self.optimizer), "trust": dict(self.trust), "domain": dict(self.domain),
                "budget": dict(self.budget), "loop": dict(self.loop), "notes": list(self.notes),
                "wall_s": self.wall_s}

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
        rounds = self.loop.get("rounds") or []
        if rounds:
            stop = self.stop
            out += ["", "## Budgeted loop", "",
                    f"{len(rounds)} round(s); stopped on **{stop.get('reason', '—')}**"
                    + (f" after round {stop['round']}" if stop.get("round") else "")
                    + (f" ({stop['text']})" if stop.get("text") else "") + ".", "",
                    "| round | evaluations | primary objective (best measured) | gain | held-out nRMSE | regret |",
                    "|---" * 6 + "|"]
            for r in rounds:
                g = r.get("improvement")
                rg = r.get("regret")
                prim = r.get("primary") or ""
                per = (r.get("surrogate") or {}).get(prim) or {}
                out.append(f"| {r['round']} | {r['evaluations']} of {self.spec.budget.max_evaluations} | "
                           f"{f(r.get('best', {}).get(prim))} {u.get(prim, '')} | "
                           + (f"{100 * g:.2f} %" if g is not None else "—") + " | "
                           + f"{f(per.get('nrmse'))} | "
                           + (f"{100 * rg:.2f} %" if rg is not None else "—") + " |")
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


def _round_record(spec: DesignStudy, rows: list[dict[str, Any]], constraints, index: int, board,
                  surrogate: Mapping[str, Any], verified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What one round cost and what it bought, as a line a reader can check instead of trusting the runner.

    ``best``: the best *measured* value of every objective, over the measured designs that satisfy the limits (or
    over all of them when none does, so the number is never silently empty). ``regret``: the decision layer's answer
    over the designs this loop recommended and the reference then measured
    (:func:`pinneapple_analysis.decision.regret`) — what the recommendation gave up against the best value the
    reference found among them, as a fraction of that set's spread. The predictions were recorded before the runs,
    so the regret is measured on predictions the model had not been trained on; it stays ``None`` until the reference
    has measured at least two recommendations, which is what makes it a number instead of an anecdote.
    """
    measured = [r for r in rows if r["ok"]]
    feasible = [r for r in measured if _satisfies(constraints, r["outputs"])] or measured
    best: dict[str, float] = {}
    for o in spec.objectives:
        vals = [float(r["outputs"][o.output]) for r in feasible]
        if vals:
            best[o.output] = min(vals) if o.sense == "min" else max(vals)
    primary = spec.objectives[0]
    rec: dict[str, Any] = {"round": index, "evaluations": len(rows), "n_measured": len(measured), "best": best,
                           "primary": primary.output, "improvement": None, "regret": None, "measured": 0,
                           "surrogate": {k: {"family": v["family"], "nrmse": v["nrmse"]}
                                         for k, v in surrogate["outputs"].items()}}
    checked = [r for r in verified if r.get("predicted") and r["ok"] and primary.output in r["predicted"]]
    if len(checked) >= 2:
        from pinneapple_analysis.decision import regret
        truth = np.array([r["outputs"][primary.output] for r in checked], float)
        pred = np.array([r["predicted"][primary.output] for r in checked], float)
        try:
            r = regret(truth, pred, lower_is_better=primary.sense == "min")
            rec["regret"] = float(r.relative) if np.isfinite(r.relative) else None
        except ValueError:                          # fewer usable points than the metric needs
            rec["regret"] = None
    return rec


def run_study(spec: DesignStudy | Mapping, evaluator: Any, *, progress: Callable[[dict], None] | None = None,
              workdir: str | None = None, stop: StopRule | Mapping | None = None) -> StudyRun:
    """Run one study. See the module docstring for the cycle; ``progress`` is called with a dict after each step
    (``{"event", "done", "planned", "design", "outputs"}``) and ``workdir`` is the scratch folder handed to a
    Simulator-shaped evaluator (a temporary one is created and removed when it is not given).

    ``stop``: the loop's rule (:class:`.loop.StopRule`, or its dict form). One round when it is not given; with it,
    the study refits on everything measured after each round and keeps measuring the surrogate's recommendation until
    the rule's number says the budget has bought all it can."""
    spec = spec if isinstance(spec, DesignStudy) else DesignStudy.from_dict(spec)
    if spec.kind not in RUNNABLE_KINDS:
        return _refused(spec, [f"a {spec.kind!r} study is described by the schema "
                               f"({', '.join(KINDS)}) but not run yet: this module runs {', '.join(RUNNABLE_KINDS)}. "
                               "What is missing is named in the issue (#1036), item by item."])
    rule = stop if stop is None or isinstance(stop, StopRule) else StopRule.from_dict(dict(stop))
    tell = progress or (lambda event: None)
    started = time.perf_counter()
    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="pinneapple-design-study-")
    try:
        return _run(spec, _Tool(evaluator, spec), tell, workdir, started, rule)
    finally:
        if own_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def _run(spec: DesignStudy, tool: _Tool, tell: Callable[[dict], None], workdir: str,
         started: float, rule: StopRule | None = None) -> StudyRun:
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
    pareto: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    optimizer: dict[str, Any] = {}
    verified_designs: list[dict[str, Any]] = []
    ev = None

    from pinneapple_autolab.domain import ValidatedDomain
    from pinneapple_design.design_optimizer.study import VALID, FunctionEvaluator
    try:
        from pinneapple_autolab.surrogates import compare_models
    except ImportError as e:
        return _refused(spec, [f"the surrogate step needs scikit-learn (pinneapple_autolab.surrogates): {e}. "
                               "Install the extra (pip install 'pinneapple[timeseries]') or hand the study a "
                               "different set of families"], "refused", rows=rows, baseline=baseline,
                        wall_s=time.perf_counter() - started)

    def fit_round(measured_rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any], Any]:
        """Cross-validate the declared families on **everything measured so far**: the board, its report, and the
        domain the measured designs cover."""
        Xm = np.array([[r["params"][k] for k in names] for r in measured_rows], float)
        Ym = {k: np.array([r["outputs"][k] for r in measured_rows], float) for k in targets}
        Um = space.to_unit(Xm)
        board_ = compare_models(Um, Ym, families=list(spec.model.families), k=spec.verifier.folds,
                                seed=spec.budget.seed)
        report = {"families": list(spec.model.families), "n_train": board_.n_train, "folds": spec.verifier.folds,
                  "chosen": dict(board_.chosen), "outputs": {}, "table": {k: v for k, v in board_.table.items()}}
        for out in targets:
            fam = board_.chosen[out]
            m = board_.table[out][fam]
            sd = float(np.std(Ym[out], ddof=1)) if len(Ym[out]) > 1 else 0.0
            report["outputs"][out] = {"family": fam, "unit": units.get(out, ""),
                                      "nrmse": float(m["rmse"] / sd) if sd > 0 else float("inf"),
                                      "rmse": float(m["rmse"]), "r2": float(m["r2"]),
                                      "coverage95": float(m["coverage95"]), "rel_mae": float(m["rel_mae"]),
                                      "sd": sd}
        # bounds so the verdicts speak in physical units ("taper_ratio = 0.30 is below the simulated range")
        return board_, report, ValidatedDomain(Um, names, bounds=space.bounds)

    # ------------------------------------------------------------------ 3..5. the rounds
    primary = spec.objectives[0]
    rounds: list[dict[str, Any]] = []
    stop_info: dict[str, Any] = {}
    incumbent: float | None = None
    board = domain = problem = None
    front: np.ndarray = np.zeros(0, int)
    noted_tightening = False
    # a surrogate study does not optimise: with no prediction to measure there is nothing for another round to buy
    max_rounds = 1 if rule is None or spec.kind != "optimize" else rule.max_rounds
    for index in range(1, max_rounds + 1):
        measured_rows = [r for r in rows if r["ok"]]
        board, surrogate, domain = fit_round(measured_rows)
        tell({"event": "surrogate", "done": len(rows), "planned": spec.budget.max_evaluations, "design": None,
              "ok": None, "outputs": {}, "chosen": dict(surrogate["chosen"]), "round": index})

        def predict(Xq: np.ndarray, board=board, domain=domain) -> dict[str, np.ndarray]:
            Uq = space.to_unit(np.atleast_2d(Xq))
            means = {k: v[0] for k, v in board.predict(Uq).items()}
            means[VALID] = np.array([v.inside for v in domain.assess(Uq)], dtype=bool)
            return means

        surrogate_evaluator = FunctionEvaluator(predict, targets, name="surrogate", fidelity="surrogate",
                                                description="the study's cross-validated surrogate")
        # An optimum sits *on* a limit, and there the surrogate's own error decides feasibility: a design whose
        # predicted margin is thinner than the model's held-out error is a coin toss, not a feasible design. So the
        # surrogate optimises against the limits tightened by that error, while the answer is judged against the real
        # ones.
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
        if any(m > 0 for m in margins.values()) and not noted_tightening:
            noted_tightening = True
            notes.append("The surrogate searched against the limits tightened by its own held-out error ("
                         + ", ".join(f"{k} {100 * v:.1f} %" for k, v in margins.items() if v > 0)
                         + "): a margin thinner than the model's error is not a margin. The answer is judged "
                           "against the limits as written.")

        n_before = len(verified_designs)
        if spec.kind == "optimize":
            from pinneapple_design.design_optimizer import methods as optimizers
            result = optimizers.run(optimizer_problem, "nsga2", budget=spec.budget.surrogate_evaluations,
                                    seed=spec.budget.seed + index - 1,
                                    initial=np.array([[r["params"][k] for k in names] for r in measured_rows],
                                                     float))
            optimizer = result.summary() | {"n_initial": int(len(measured_rows)), "rounds": index}
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
            # The round measures designs the reference has not seen: spending a reference evaluation on a design
            # already measured would buy nothing and would show up as a round with no progress.
            tried = [space.to_unit(np.array([[r["params"][k] for k in names]]))[0] for r in measured_rows]
            order = list(front) if len(front) else np.array([i for i in [result.best(0)] if i is not None], int)
            chosen: list[int] = []
            wanted = plan["verify"] if rule is None else rule.per_round
            for i in order:
                u = space.to_unit(ev.X[i:i + 1])[0]
                if any(np.linalg.norm(u - t) <= _VERIFY_MIN_DISTANCE for t in tried):
                    continue
                if all(np.linalg.norm(u - space.to_unit(ev.X[j:j + 1])[0]) > _VERIFY_MIN_DISTANCE for j in chosen):
                    chosen.append(int(i))
                if len(chosen) >= wanted:
                    break
            for i in chosen:
                d = space.as_dicts(ev.X[i])[0]
                if len(rows) >= spec.budget.max_evaluations:
                    rejected.append({"params": d, "source": "budget", "outputs": {},
                                     "reasons": ["the reference budget was spent before this design could be "
                                                 "measured"]})
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
              "planned": spec.budget.max_evaluations, "design": None, "ok": None, "outputs": {},
              "round": index})

        # ------------------------------------------------------------------ did the round buy anything?
        rec = _round_record(spec, rows, constraints, index, board, surrogate, verified_designs)
        if incumbent is not None:
            cur = rec["best"].get(primary.output)
            if cur is not None:
                den = abs(incumbent) if incumbent else 1.0
                rec["improvement"] = ((incumbent - cur) if primary.sense == "min" else (cur - incumbent)) / den
        inc = rec["best"].get(primary.output)
        if inc is not None:
            incumbent = inc
        rec["measured"] = len(verified_designs) - n_before
        rounds.append(rec)
        budget_left = spec.budget.max_evaluations - len(rows)
        if rule is None:
            stop_info = {"stop": True, "reason": "single_pass", "round": index, "evaluations": len(rows),
                         "budget_left": int(budget_left), "value": None, "limit": None,
                         "text": "one cycle: baseline, DOE, surrogate, NSGA-II and the verifications (no StopRule "
                                 "was given)"}
        elif spec.kind != "optimize":
            stop_info = {"stop": True, "reason": "single_pass", "round": index, "evaluations": len(rows),
                         "budget_left": int(budget_left), "value": None, "limit": None,
                         "text": f"a {spec.kind} study does not optimise: there is no recommendation for another "
                                 "round to measure"}
        else:
            stop_info = decide(rule, rounds, budget_left=budget_left)
        tell({"event": "round", "done": len(rows), "planned": spec.budget.max_evaluations, "design": None,
              "ok": None, "outputs": {}, "round": index, "stop": dict(stop_info)})
        if stop_info["stop"]:
            break

    # ------------------------------------------------------------------ measured answer
    measured_rows = [r for r in rows if r["ok"]]
    Xm = np.array([[r["params"][k] for k in names] for r in measured_rows], float)
    Ym = {k: np.array([r["outputs"][k] for r in measured_rows], float) for k in spec.evaluator.output_names}
    Ym[VALID] = np.ones(len(measured_rows), bool)
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
    domain_info = {"n_simulated": int(len(measured_rows)), "spacing": float(domain.spacing),
                   "rule": f"inside the simulated ranges (widened by {domain.margin:g}) and no farther than "
                           f"{domain.max_ratio:g} x the typical spacing from the nearest measured design",
                   "verified": []}
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
    loop = {"rule": None if rule is None else rule.to_dict(), "rounds": rounds, "stop": stop_info}
    if rule is not None and spec.kind == "optimize":
        notes.append(f"The loop ran {len(rounds)} round(s) of {max_rounds} and stopped on {stop_info['reason']}: "
                     f"{stop_info.get('text', '')}")
    return StudyRun(spec=spec, status=status, reasons=reasons, created=created, baseline=baseline,
                    constraints=[{"output": c.name, "lower": c.lower, "upper": c.upper, "unit": c.unit,
                                  "kind": c.kind, "label": c.label, "text": c.describe(),
                                  "relative": any(lim.relative for lim in spec.constraints
                                                  if lim.output == c.name)} for c in constraints],
                    rows=rows, pareto=pareto, rejected=rejected, surrogate=surrogate, optimizer=optimizer,
                    trust=_trust_dict(trust), domain=domain_info, loop=loop,
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
