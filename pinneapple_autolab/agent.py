"""The lab driven by an agent: a mission in words -> a Mission -> tool calls on the lab -> an audited answer.

    from pinneapple_autolab.agent import run_agent
    res = run_agent("Find a lower-drag design at 70 m/s. Preserve the downforce ... 24 CFD runs.", sim, "runs/agent",
                    domain=racecar_domain())                        # deterministic planner
    res = run_agent(text, sim, folder, domain=..., provider="anthropic")   # an LLM decides each step (needs a key)

The agent decides one step at a time (:func:`pinneapple_autolab.core.act`): :class:`LabPlanner` (deterministic, no
network) or :class:`~pinneapple_autolab.core.LLMStepPlanner` (a language model), over the same toolbox
(:func:`lab_toolbox`): a closed set of :class:`~pinneapple_autolab.core.Tool` with declared inputs. There is no tool
to change the operating conditions, the bounds, the limits or the solver settings; a step with an argument a tool
does not declare is refused and the refusal is recorded as evidence. Every number a tool returns
comes from the lab's ledger or from a surrogate prediction labelled as such. The final answer is audited
(:func:`audit_claims`): every number in it must appear in what the tools returned, and the recommended run must be a
successful simulation that meets every limit; otherwise the answer is marked unsupported.

:func:`interpret_mission` turns the text into a :class:`~.mission.Mission` from a domain description
(:class:`MissionDomain`): a deterministic reading of the request (objective words, "preserve/keep" limits, the
budget, the operating point), optionally an LLM extraction restricted to the same closed options. A request for an
operating point the simulator does not run (e.g. 80 m/s when the solver is set up at 70 m/s) is refused with the
reason, never silently changed.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer.study import Objective, Variable

from .core import Evidence, LLMStepPlanner, Run, Step, Tool, Toolbox, act
from .lab import AutonomousLab
from .ledger import BudgetExhaustedError
from .mission import Budget, Limit, Mission
from .simulator import Simulator

__all__ = ["MissionDomain", "MissionReading", "interpret_mission", "lab_toolbox", "LabPlanner", "audit_claims",
           "AgentResult", "run_agent", "MissionRefused", "AGENT_RULES"]


class MissionRefused(ValueError):
    """The request cannot be run as written (with the reason and what is needed)."""


# ================================================================================================ mission reading
@dataclass
class MissionDomain:
    """What a simulator can be asked: its variables and baseline, outputs by the words that name them, the fixed
    operating point, default acceptance tolerances. ``output_words``: output -> words (e.g. "drag_N": ["drag"]);
    ``operating_point``: name -> (value, unit, regex capturing the value in a request)."""
    title: str
    variables: list[Variable]
    baseline: dict[str, float]
    output_words: dict[str, list[str]]
    units: dict[str, str]
    operating_point: dict[str, tuple[float, str, str]]
    acceptance: dict[str, float]
    conditions: dict[str, Any]
    extra_limits: list[Limit] = field(default_factory=list)       # limits engineering practice adds, stated
    hypotheses: list[str] = field(default_factory=list)
    default_budget: int = 24
    default_reserve: int = 3
    domain: str = ""


@dataclass
class MissionReading:
    mission: Mission
    readings: list[dict[str, str]]            # what was read, from which words
    assumptions: list[str]                    # what was not in the text and how it was filled (stated)

    def to_dict(self) -> dict[str, Any]:
        return {"mission": self.mission.to_dict(), "readings": self.readings, "assumptions": self.assumptions}


_LOWER = ("lower", "reduce", "minimi", "less", "smaller", "decrease", "cut")
_HIGHER = ("higher", "increase", "maximi", "more", "larger", "raise")
_KEEP = ("preserve", "keep", "maintain", "retain", "at least", "no less", "not lose", "without losing")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+", text) if s.strip()]


def interpret_mission(text: str, dom: MissionDomain, llm: Any = None) -> MissionReading:
    """Read ``text`` into a Mission over ``dom``. ``llm`` (optional) has ``extract(text, schema) -> dict`` (e.g.
    :class:`pinneapple_design.intent.CallLLM`); its answer is accepted only where it names closed options and quotes
    words present in the text."""
    low = text.lower()
    readings: list[dict[str, str]] = []
    assumptions: list[str] = []
    # operating point: must match what the simulator runs
    for name, (value, unit, rx) in dom.operating_point.items():
        m = re.search(rx, text, re.I)
        if m:
            asked = float(m.group(1))
            readings.append({"what": f"{name} = {asked:g} {unit}", "from": m.group(0)})
            if abs(asked - value) > 1e-6 * max(abs(value), 1):
                raise MissionRefused(f"the request asks for {name} = {asked:g} {unit}, but the reference simulator is "
                                     f"set up at {value:g} {unit}; changing it needs a new solver qualification "
                                     "(mesh and convergence) before any design can be compared")
        else:
            assumptions.append(f"{name} not stated: the simulator's {value:g} {unit}")
    objectives: list[Objective] = []
    limits: list[Limit] = []
    for sent in _sentences(text):
        s = sent.lower()
        for out, words in dom.output_words.items():
            hit = next((w for w in words if re.search(rf"\b{re.escape(w)}", s)), None)
            if hit is None:
                continue
            before = s[:s.index(hit)]
            if any(k in before[-40:] for k in _KEEP) or any(k in s for k in ("preserve", "keep", "maintain")):
                limits.append(Limit(out, lower_frac_of_baseline=1.0, unit=dom.units.get(out, ""), label=hit))
                readings.append({"what": f"limit: {out} >= baseline", "from": sent})
            elif any(k in before[-30:] or k in s[s.index(hit):s.index(hit) + 20] for k in _LOWER):
                objectives.append(Objective(out, "min", dom.units.get(out, ""), hit))
                readings.append({"what": f"objective: minimise {out}", "from": sent})
            elif any(k in before[-30:] for k in _HIGHER):
                objectives.append(Objective(out, "max", dom.units.get(out, ""), hit))
                readings.append({"what": f"objective: maximise {out}", "from": sent})
    if llm is not None:
        schema = {"objectives": [{"output": {"one of": list(dom.output_words)}, "sense": {"one of": ["min", "max"]},
                                  "quote": "exact words"}],
                  "keep_at_least_baseline": [{"output": {"one of": list(dom.output_words)}, "quote": "exact words"}],
                  "budget_runs": "integer or null", "budget_quote": "exact words"}
        try:
            ans = llm.extract(text, schema)
        except Exception as e:  # noqa: BLE001 - the deterministic reading stands
            assumptions.append(f"LLM extraction failed ({type(e).__name__}); deterministic reading used")
            ans = {}
        for o in ans.get("objectives", []) or []:
            if o.get("output") in dom.output_words and o.get("sense") in ("min", "max") and \
                    str(o.get("quote", "")).lower() in low and all(x.name != o["output"] for x in objectives):
                objectives.append(Objective(o["output"], o["sense"], dom.units.get(o["output"], "")))
                readings.append({"what": f"objective ({o['sense']}) {o['output']} [LLM]", "from": o["quote"]})
        for k in ans.get("keep_at_least_baseline", []) or []:
            if k.get("output") in dom.output_words and str(k.get("quote", "")).lower() in low and \
                    all(x.name != k["output"] for x in limits):
                limits.append(Limit(k["output"], lower_frac_of_baseline=1.0, unit=dom.units.get(k["output"], "")))
                readings.append({"what": f"limit {k['output']} >= baseline [LLM]", "from": k["quote"]})
    objectives = list({o.name: o for o in objectives}.values())
    limits = [lim for lim in {lim.name: lim for lim in limits}.values() if all(o.name != lim.name for o in objectives)]
    if not objectives:
        raise MissionRefused("no objective found: say what to lower or raise (e.g. 'lower drag'); outputs this "
                             f"simulator computes: {', '.join(w[0] for w in dom.output_words.values())}")
    m = re.search(r"(\d+)\s*(?:cfd|simulation|run|solver)", low)
    if m:
        budget = int(m.group(1))
        readings.append({"what": f"budget = {budget} simulations", "from": m.group(0)})
    else:
        budget = dom.default_budget
        assumptions.append(f"budget not stated: {budget} reference simulations")
    reserve = dom.default_reserve if re.search(r"\b(validat|verif|confirm)", low) else 1
    if re.search(r"\b(validat|verif|confirm)", low):
        readings.append({"what": f"final verification: {reserve} simulations kept for it",
                         "from": re.search(r"\b(validat\w*|verif\w*|confirm\w*)[^.]*", low).group(0)})
    if re.search(r"geometr\w* constraint", low):
        readings.append({"what": "geometry constraints: variable bounds and the simulator's geometry checks",
                         "from": re.search(r"[^.]*geometr\w* constraint[^.]*", low).group(0).strip()})
    for lim in dom.extra_limits:
        if all(x.name != lim.name for x in limits):
            limits.append(lim)
            assumptions.append(f"added by engineering practice (not in the request): {lim.describe()}")
    mission = Mission(dom.title, text.strip(), list(dom.variables), objectives, limits, dict(dom.baseline),
                      Budget(budget, min(reserve, budget - 2)), dict(dom.acceptance), dict(dom.conditions),
                      list(dom.hypotheses), dom.domain)
    return MissionReading(mission, readings, assumptions)


# ================================================================================================ tools
def _r(v: Any, nd: int = 5) -> Any:
    if isinstance(v, dict):
        return {k: _r(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_r(x, nd) for x in v]
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return float(f"{f:.{nd}g}") if np.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _int(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise TypeError(f"{name} must be an integer, got {v!r}")
    return int(v)


def lab_toolbox(lab: AutonomousLab, reading: MissionReading | None = None) -> Toolbox:
    """The closed tool set an agent drives ``lab`` with. Every tool returns what the lab measured or predicted
    (predictions are labelled), rounded to 5 significant digits; ``answer`` records the final answer."""
    outs = lab.mission.outputs()
    shown: set[int] = set()

    def mission(state):
        return _r(reading.to_dict() if reading else {"mission": lab.mission.to_dict()})

    def plan(state, n_doe=None):
        p = lab.plan(None if n_doe is None else _int("n_doe", n_doe))
        return {"allocation": p["allocation"], "objectives": p["objectives"], "limits": p["limits"]}

    def baseline(state):
        r = lab.baseline()
        return _r({"run": r.id, "outputs": {k: r.outputs[k] for k in outs},
                   "limits": [c.describe() for c in lab.mission.constraints_absolute()], "cost_s": r.cost_s})

    def space_filling(state, n):
        n = _int("n", n)
        left = lab.ledger.remaining() - lab.mission.budget.reserve_verify
        if n > left:
            raise BudgetExhaustedError(f"{n} space-filling runs requested, {left} left before the verification reserve")
        recs = lab.doe(n)
        return _r({"runs": [{"run": r.id, "ok": r.ok, "outputs": {k: r.outputs.get(k) for k in outs}} for r in recs],
                   "budget_left": lab.ledger.remaining()})

    def check_design(state, params):
        if not isinstance(params, dict):
            raise TypeError("params must be an object {variable: value}")
        return {"issues": lab.check_design({k: float(v) for k, v in params.items()})}

    def learn(state):
        b = lab.learn()
        return _r({"n_train": b.n_train, "chosen": b.chosen,
                   "held_out_rel_error": {o: {f: v["rel_mae"] for f, v in d.items() if "rel_mae" in v}
                                          for o, d in b.table.items()},
                   "coverage95": {o: d[b.chosen[o]]["coverage95"] for o, d in b.table.items()},
                   "field_rmse": {k: v["rmse"] for k, v in lab.field_cv.get("per_region", {}).items()}})

    def optimize_and_assess(state, top=3):
        top = _int("top", top)
        lab.optimize()
        cands = lab.assess()
        o = lab.mission.objectives[0]
        sgn = 1 if o.sense == "min" else -1
        out = {"counts": {lb: sum(c.label == lb for c in cands)
                          for lb in ("favourable", "uncertain", "out_of_domain", "unlikely")},
               "best": {}, "note": "predictions, not simulations"}
        for lb in ("favourable", "uncertain", "out_of_domain"):
            pool = sorted([c for c in cands if c.label == lb and c.p_feasible >= 0.05],
                          key=lambda c: sgn * c.predicted[o.name])[:top]
            shown.update(c.index for c in pool)
            out["best"][lb] = [{"index": c.index, "predicted": {k: c.predicted[k] for k in outs},
                                "std": {k: c.std[k] for k in outs}, "p_feasible": c.p_feasible,
                                "reasons": c.reasons, "trust": c.report.decisao} for c in pool]
        return _r(out)

    def recommend_next(state):
        nxt = lab.choose_next()
        if nxt.candidate is None:
            return {"index": None, "rule": nxt.rule, "reason": nxt.reason}
        shown.add(nxt.candidate.index)
        return _r({"index": nxt.candidate.index, "label": nxt.candidate.label, "rule": nxt.rule,
                   "reason": nxt.reason, "alternatives": nxt.alternatives})

    def simulate_candidate(state, index, reason):
        index = _int("index", index)
        if index not in shown or index >= len(lab.candidates):
            raise ValueError(f"candidate {index} was not returned by optimize_and_assess or recommend_next")
        if lab.ledger.remaining() <= lab.mission.budget.reserve_verify:
            raise BudgetExhaustedError("only the final-verification reserve is left: call verify_best")
        c = lab.candidates[index]
        rec = lab.simulate(c.params, "acquisition", str(reason), predicted={"mean": c.predicted, "std": c.std,
                                                                            "label": c.label})
        e = lab._record_outcome(c, rec, None)
        e["rule"], e["reason"] = "agent's choice", str(reason)
        return _r({"run": e["run"], "judged_before": e["label_before"], "predicted": e["predicted"],
                   "simulated": e["reference"], "rel_error": e["rel_error"], "after": e["label_after"],
                   "budget_left": lab.ledger.remaining()})

    def verify_best(state, n=None):
        done = lab.verify(None if n is None else _int("n", n))
        return _r({"verified": [{"run": lab.decisions[-len(done) + i]["run"] if done else None, "params": c.params,
                                 "predicted": c.predicted, "simulated": c.reference, "rel_error": c.rel_error,
                                 "label": c.label} for i, c in enumerate(done)],
                   "budget_left": lab.ledger.remaining()})

    def status(state):
        best = lab.best_simulated()
        return _r({"used": lab.ledger.used_runs, "left": lab.ledger.remaining(),
                   "reserve_verify": lab.mission.budget.reserve_verify,
                   "best": None if best is None else {"run": best.id, "params": best.params,
                                                      "outputs": {k: best.outputs[k] for k in outs}},
                   "runs": [{"run": r.id, "purpose": r.purpose, "ok": r.ok} for r in lab.ledger.runs]})

    def answer(state, text, recommended_run=None):
        return {"text": str(text), "recommended_run": recommended_run}

    full = lambda out: out  # noqa: E731 - the evidence keeps the whole (already compact) result
    P = "design variables: " + ", ".join(f"{v.name} [{v.lower:g}, {v.upper:g}] {v.unit}" for v in lab.mission.variables)
    spec = [
        ("read_mission", mission, "The mission as read from the request: objectives, limits, variables, budget, fixed "
         "conditions, what was read from which words and what was assumed.", {}),
        ("plan", plan, "Split the simulation budget (baseline, space-filling, chosen one at a time, verification).",
         {"n_doe": "integer, number of space-filling simulations (optional)"}),
        ("simulate_baseline", baseline, "Simulate the starting design; fixes the limits relative to it.", {}),
        ("space_filling", space_filling, "Simulate n space-filling designs (Latin hypercube).", {"n": "integer"}),
        ("check_design", check_design, "Geometry and boundary-condition checks of a design (no simulation).",
         {"params": P}),
        ("learn", learn, "Train and cross-validate every surrogate family on all simulations so far.", {}),
        ("optimize_and_assess", optimize_and_assess, "NSGA-II on the surrogate, then triage of every candidate "
         "(favourable / uncertain / out_of_domain / unlikely); returns counts and the best of each label by index.",
         {"top": "integer, candidates shown per label (optional)"}),
        ("recommend_next", recommend_next, "The lab's written rule for the next simulation, with its reason.", {}),
        ("simulate_candidate", simulate_candidate, "Simulate a candidate returned by optimize_and_assess or "
         "recommend_next; its prediction is recorded before the run and compared with the result.",
         {"index": "integer", "reason": "why this simulation (text)"}),
        ("verify_best", verify_best, "Simulate the n best favourable candidates as the final verification.",
         {"n": "integer (optional)"}),
        ("status", status, "Budget used and left, the best simulated design that meets every limit, the runs.", {}),
        ("answer", answer, "The final answer: text and the recommended simulated run id.",
         {"text": "the answer", "recommended_run": "run id (e.g. run019-verification) or null"}),
    ]
    return Toolbox(f"autolab:{lab.mission.domain or lab.mission.title}",
                   [Tool(n, f, d, i, full) for n, f, d, i in spec])


AGENT_RULES = ("Measure the baseline before anything that depends on it. The recommendation must be a simulated run "
               "that meets every limit (status gives it); keep the last simulations for verify_best. Finish with "
               "the answer tool, then done.")


# ================================================================================================ planner
class LabPlanner:
    """Deterministic step planner over :func:`lab_toolbox` (no network): read the mission, plan, baseline,
    space-filling designs, then (learn, assess, follow the lab's rule, simulate) until only the verification reserve
    is left, verify, read the status, answer. A stand-in for a language model, not a different path: the same
    tools, refusals and audit apply."""

    name = "deterministic"

    def __init__(self, lab: AutonomousLab, n_doe: int | None = None):
        self.lab, self.n_doe = lab, n_doe

    def next(self, mission: str, toolbox: Toolbox, state: dict[str, Any], evidence: list[Evidence]) -> Step | None:
        done = {e.tool for e in evidence if e.ok}
        last = evidence[-1] if evidence else None
        lab = self.lab
        if "answer" in done:
            return None
        if "read_mission" not in done:
            return Step("read_mission", {}, "start from what the request asks and what was assumed")
        if "plan" not in done:
            return Step("plan", {} if self.n_doe is None else {"n_doe": self.n_doe}, "split the budget")
        if "simulate_baseline" not in done:
            return Step("simulate_baseline", {}, "the limits are relative to the baseline: measure it first")
        if "space_filling" not in done:
            n = lab.plan_["allocation"]["doe"]
            return Step("space_filling", {"n": n}, f"{n} space-filling designs before any model is trusted")
        reserve = lab.mission.budget.reserve_verify
        if lab.ledger.remaining() > reserve and last is not None and last.ok:
            if last.tool in ("space_filling", "simulate_candidate"):
                return Step("learn", {}, "retrain on every simulation so far and measure the held-out errors")
            if last.tool == "learn":
                return Step("optimize_and_assess", {"top": 3}, "search the surrogate and triage the candidates")
            if last.tool == "optimize_and_assess":
                return Step("recommend_next", {}, "ask the written rule which simulation teaches the most")
            if last.tool == "recommend_next" and last.summary.get("index") is not None:
                rec = last.summary
                return Step("simulate_candidate", {"index": rec["index"], "reason": f"{rec['rule']}: {rec['reason']}"},
                            rec["rule"])
        if lab.ledger.remaining() > 0 and "verify_best" not in done:
            return Step("verify_best", {"n": min(reserve, lab.ledger.remaining())},
                        "spend the reserve on confirming the best favourable designs")
        if last is None or last.tool != "status":
            return Step("status", {}, "read the best simulated design before answering")
        st, base = last.summary, next(e.summary for e in evidence if e.tool == "simulate_baseline" and e.ok)
        o = lab.mission.objectives[0].name
        if not st.get("best"):
            return Step("answer", {"text": "No simulated design meets every limit; no recommendation.",
                                   "recommended_run": None}, "nothing feasible was simulated")
        b = st["best"]
        text = (f"Recommended design: {b['run']} (simulated). {o} {b['outputs'][o]} against the baseline's "
                f"{base['outputs'][o]}. Limits checked on its simulation: {'; '.join(base['limits'])}. Its simulated "
                "values: " + ", ".join(f"{k} {v}" for k, v in b["outputs"].items()) + ".")
        return Step("answer", {"text": text, "recommended_run": b["run"]}, "budget spent: the best simulated design")


# ================================================================================================ audit
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numbers(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x) for x in _NUM.findall(obj)]
    return out


def audit_claims(answer: str, tool_log: Sequence[Mapping[str, Any]], lab: AutonomousLab,
                 recommended_run: str | None = None, rtol: float = 5e-3) -> dict[str, Any]:
    """Every number in ``answer`` must match (within ``rtol``, or as a rounding) a number the tools returned or a
    run id/variable/budget figure; the recommended run must exist, have succeeded and meet every limit."""
    known = _numbers([e.get("result") for e in tool_log if "result" in e]) + _numbers(lab.ledger.to_dict()["runs"])
    known_arr = np.array(known) if known else np.zeros(1)
    unsupported = []
    for tok in _NUM.findall(re.sub(r"run\d+-\w+", "", answer)):
        v = float(tok)
        dec = len(tok.split(".")[1]) if "." in tok else 0
        tol = np.maximum(rtol * np.abs(known_arr), 0.5 * 10 ** (-dec))
        if not np.any(np.abs(known_arr - v) <= tol):
            unsupported.append(tok)
    rec_ok, why = None, ""
    if recommended_run is not None:
        r = next((x for x in lab.ledger.runs if x.id == recommended_run), None)
        if r is None:
            rec_ok, why = False, "no such run"
        elif not r.ok:
            rec_ok, why = False, "the run failed"
        else:
            viol = [c.describe() for c in lab.mission.constraints_absolute()
                    if c.violation(r.outputs.get(c.name, np.nan)) > 0]
            rec_ok, why = (not viol), ("violates " + "; ".join(viol)) if viol else "simulated and meets every limit"
    return {"supported": not unsupported and rec_ok is not False, "unsupported_numbers": unsupported,
            "recommended_run": recommended_run, "recommendation_ok": rec_ok, "recommendation_check": why,
            "n_numbers_checked": len(_NUM.findall(answer))}


# ================================================================================================ run
@dataclass
class AgentResult:
    reading: MissionReading | None
    refused: str | None
    answer: str | None
    audit: dict[str, Any]
    run: Run | None
    summary: dict[str, Any] | None
    wall_s: float
    planner: str

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [] if self.run is None else [{"tool": e.tool, "args": e.args, "why": e.why, "ok": e.ok,
                                             "error": e.error, "seconds": e.seconds} for e in self.run.evidence]

    def to_dict(self) -> dict[str, Any]:
        return _r({"planner": self.planner, "refused": self.refused, "answer": self.answer, "audit": self.audit,
                   "reading": None if self.reading is None else self.reading.to_dict(),
                   "run": None if self.run is None else self.run.to_dict(), "wall_s": self.wall_s,
                   "summary": self.summary})


def run_agent(text: str, simulator: Simulator, folder: str, *, domain: MissionDomain, planner: Any = None,
              provider: str | None = None, model: str | None = None, llm: Callable[[str, str], str] | None = None,
              mission_llm: Any = None, n_doe: int | None = None, max_steps: int = 200,
              lab_options: dict[str, Any] | None = None,
              listeners: list[Callable[[dict], None]] | None = None) -> AgentResult:
    """Interpret ``text``, build the lab and let the agent drive it through :func:`lab_toolbox`. Planner: the
    given ``planner``, else an :class:`~pinneapple_autolab.core.LLMStepPlanner` when ``provider`` or ``llm`` is
    given ("anthropic", "openai", "openai_compatible", "ollama"), else :class:`LabPlanner`."""
    t0 = time.time()
    try:
        reading = interpret_mission(text, domain, mission_llm)
    except MissionRefused as e:
        return AgentResult(None, str(e), None, {"supported": True, "unsupported_numbers": []}, None, None,
                           time.time() - t0, "none")
    lab = AutonomousLab(reading.mission, simulator, folder, listeners=listeners, **(lab_options or {}))
    lab.ledger.event("interpret", "mission read: " + "; ".join(r["what"] for r in reading.readings),
                     reading=reading.to_dict())
    box = lab_toolbox(lab, reading)
    if planner is None:
        planner = (LLMStepPlanner(provider=provider or "anthropic", model=model, llm=llm, rules=AGENT_RULES)
                   if (provider or llm) else LabPlanner(lab, n_doe))
    run = act(text, box, planner, max_steps=max_steps,
              log=lambda s: lab.ledger.event("agent", s))
    ans = run.state.get("answer") or {}
    log = [{"tool": e.tool, "result": e.summary} for e in run.evidence if e.ok and e.tool != "answer"]
    audit = audit_claims(ans.get("text", ""), log, lab, ans.get("recommended_run"))
    summary = lab.summary() if lab.ledger.runs else None
    lab.ledger.event("deliver", f"agent answer ({'supported' if audit['supported'] else 'UNSUPPORTED'}): "
                     f"{ans.get('text')}", audit=audit)
    lab.ledger.save()
    return AgentResult(reading, None, ans.get("text"), audit, run, summary, time.time() - t0,
                       getattr(planner, "name", "planner"))
