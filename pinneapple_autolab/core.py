"""Mission -> plan -> controlled tool calls -> evidence: the agent loop of the Autonomous Physics Lab.

The agent never runs code of its own. It picks steps from a :class:`Toolbox` of library calls registered for one
kind of mission, with arguments checked against each tool's declared inputs, and every call is logged as
:class:`Evidence` (arguments, a short summary of the result, wall time, error). Two planners share one interface:

* :class:`TemplatePlanner`: deterministic, the steps a mission kind always needs (no network, no API key);
* :class:`LLMPlanner`: asks a language model (``pinneapple_llm.call_llm``: Anthropic / OpenAI / Ollama) for a JSON
  plan over the same toolbox; any unknown tool, unknown argument or failure falls back to the template plan and the
  reason is recorded in the plan.

    box = Toolbox("board_cooling"); box.add(Tool("simulate", fn, "...", {"design": "..."}))
    plan = TemplatePlanner([Step("simulate", {"design": "baseline"})]).plan("cool this board", box)
    run = execute(plan, box)            # run.state: outputs by tool name; run.evidence: one entry per call
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Tool:
    """A library call the agent may make: ``fn(state, **args) -> dict`` (the dict lands in ``state[name]``).
    ``inputs``: argument name -> description (the only arguments accepted); ``summary``: result -> short dict for
    the evidence log (default: the scalar entries of the result)."""
    name: str
    fn: Callable[..., dict]
    description: str
    inputs: dict[str, str] = field(default_factory=dict)
    summary: Callable[[dict], dict] | None = None


class Toolbox:
    """The controlled set of tools of one mission kind."""

    def __init__(self, kind: str, tools: list[Tool] | None = None):
        self.kind = kind
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> Toolbox:
        self.tools[tool.name] = tool
        return self

    def names(self) -> list[str]:
        return list(self.tools)

    def check(self, step: Step) -> str | None:
        """Why ``step`` cannot run (unknown tool or argument), or None."""
        t = self.tools.get(step.tool)
        if t is None:
            return f"unknown tool {step.tool!r} (toolbox {self.kind}: {', '.join(self.tools)})"
        bad = sorted(set(step.args) - set(t.inputs))
        return f"{step.tool}: unknown argument(s) {bad}" if bad else None

    def describe(self) -> list[dict[str, Any]]:
        """The tool list as plain data (what a language model is shown)."""
        return [{"tool": t.name, "description": t.description, "inputs": t.inputs} for t in self.tools.values()]


@dataclass
class Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""


@dataclass
class Plan:
    mission: str
    steps: list[Step]
    planner: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"mission": self.mission, "planner": self.planner, "notes": self.notes,
                "steps": [asdict(s) for s in self.steps]}


class Planner(Protocol):
    def plan(self, mission: str, toolbox: Toolbox) -> Plan: ...


class TemplatePlanner:
    """The fixed sequence of steps of a mission kind."""

    name = "template"

    def __init__(self, steps: list[Step]):
        self.steps = steps

    def plan(self, mission: str, toolbox: Toolbox) -> Plan:
        for s in self.steps:
            err = toolbox.check(s)
            if err:
                raise ValueError(f"template plan: {err}")
        return Plan(mission, [Step(s.tool, dict(s.args), s.why) for s in self.steps], self.name)


class LLMPlanner:
    """A language model orders the toolbox's tools into a plan; the template plan is the fallback.

    ``llm``: any callable ``(prompt, system) -> str`` returning JSON ``{"steps": [{"tool", "args", "why"}]}``; by
    default :func:`pinneapple_llm.call_llm` with ``provider`` / ``model``. The plan must contain every tool of
    ``required`` (the steps that produce the evidence a decision needs) or it is rejected."""

    name = "llm"

    def __init__(self, fallback: TemplatePlanner, *, provider: str = "anthropic", model: str | None = None,
                 llm: Callable[[str, str], str] | None = None, required: tuple[str, ...] = ()):
        self.fallback, self.provider, self.model, self._llm, self.required = fallback, provider, model, llm, required

    def _ask(self, prompt: str, system: str) -> str:
        if self._llm is not None:
            return self._llm(prompt, system)
        from pinneapple_llm import call_llm
        out = call_llm(prompt, provider=self.provider, model=self.model, system=system, json_mode=True)
        return out if isinstance(out, str) else json.dumps(out)

    def plan(self, mission: str, toolbox: Toolbox) -> Plan:
        system = ("You plan an engineering study. Use only these tools, with only their listed inputs, and answer "
                  'with JSON {"steps": [{"tool": name, "args": {...}, "why": text}]}: '
                  + json.dumps(toolbox.describe()))
        try:
            raw = json.loads(self._ask(mission, system))
            steps = [Step(str(s["tool"]), dict(s.get("args") or {}), str(s.get("why", ""))) for s in raw["steps"]]
            errs = [e for e in (toolbox.check(s) for s in steps) if e]
            missing = [r for r in self.required if r not in {s.tool for s in steps}]
            if errs or missing or not steps:
                raise ValueError("; ".join(errs + [f"missing required step {m!r}" for m in missing]) or "empty plan")
            return Plan(mission, steps, f"{self.name}:{self.provider}")
        except Exception as e:  # noqa: BLE001 - any LLM failure falls back to the deterministic plan
            p = self.fallback.plan(mission, toolbox)
            p.planner = f"{self.fallback.name} (LLM fallback)"
            p.notes.append(f"LLM plan not used: {type(e).__name__}: {e}")
            return p


@dataclass
class Evidence:
    step: int
    tool: str
    args: dict[str, Any]
    why: str
    summary: dict[str, Any]
    seconds: float
    ok: bool
    error: str = ""


@dataclass
class Run:
    plan: Plan
    state: dict[str, Any]
    evidence: list[Evidence]

    @property
    def ok(self) -> bool:
        return all(e.ok for e in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.to_dict(), "ok": self.ok, "evidence": [asdict(e) for e in self.evidence]}


def _scalars(out: dict) -> dict:
    return {k: v for k, v in out.items()
            if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 120)}


def execute(plan: Plan, toolbox: Toolbox, state: dict[str, Any] | None = None, *, stop_on_error: bool = True,
            log: Callable[[str], None] | None = None) -> Run:
    """Run the plan's steps in order through the toolbox only. Each tool sees the shared ``state`` (earlier
    outputs by tool name, plus whatever the caller put there; ``_plan`` and ``_evidence`` hold the plan and the
    evidence so far) and returns a dict stored under its name."""
    state = {} if state is None else state
    ev: list[Evidence] = []
    state["_plan"], state["_evidence"] = plan, ev          # a reporting tool can cite the run so far
    for i, s in enumerate(plan.steps):
        err = toolbox.check(s)
        t0 = time.perf_counter()
        if err:
            ev.append(Evidence(i, s.tool, s.args, s.why, {}, 0.0, False, err))
        else:
            tool = toolbox.tools[s.tool]
            try:
                out = tool.fn(state, **s.args) or {}
                state[s.tool] = out
                ev.append(Evidence(i, s.tool, s.args, s.why, (tool.summary or _scalars)(out),
                                   round(time.perf_counter() - t0, 3), True))
            except Exception as e:  # noqa: BLE001 - recorded as evidence, the caller decides
                ev.append(Evidence(i, s.tool, s.args, s.why, {}, round(time.perf_counter() - t0, 3), False,
                                   f"{type(e).__name__}: {e}"))
        if log:
            e = ev[-1]
            log(f"[{i + 1}/{len(plan.steps)}] {e.tool} {'ok' if e.ok else 'FAILED'} in {e.seconds:.1f} s"
                + (f": {e.error}" if e.error else ""))
        if stop_on_error and not ev[-1].ok:
            break
    return Run(plan, state, ev)


# ================================================================================================ step-by-step loop
class StepPlanner(Protocol):
    """Chooses the next step after seeing every result so far (``None``: done)."""
    name: str

    def next(self, mission: str, toolbox: Toolbox, state: dict[str, Any], evidence: list[Evidence]) -> Step | None: ...


class LLMStepPlanner:
    """A language model chooses each next step from the toolbox, seeing the mission, the tools and the evidence so
    far (summaries only: the numbers the tools returned). It answers ``{"tool", "args", "why"}`` or
    ``{"done": true}``. An unknown tool or argument is not run: the refusal is recorded as evidence and shown to the
    model on the next turn. ``llm``: ``(prompt, system) -> str``; default :func:`pinneapple_llm.call_llm`."""

    name = "llm-step"

    def __init__(self, *, provider: str = "anthropic", model: str | None = None,
                 llm: Callable[[str, str], str] | None = None, rules: str = ""):
        self.provider, self.model, self._llm, self.rules = provider, model, llm, rules
        self.name = f"llm-step:{provider}" if llm is None else "llm-step:custom"

    def _ask(self, prompt: str, system: str) -> str:
        if self._llm is not None:
            return self._llm(prompt, system)
        from pinneapple_llm import call_llm
        out = call_llm(prompt, provider=self.provider, model=self.model, system=system, json_mode=True)
        return out if isinstance(out, str) else json.dumps(out)

    def next(self, mission, toolbox, state, evidence):
        system = ("You run an engineering study one tool call at a time. Use only these tools with only their listed "
                  "inputs; never state a number a tool did not return. " + self.rules + " Answer with JSON "
                  '{"tool": name, "args": {...}, "why": text} or {"done": true}. Tools: '
                  + json.dumps(toolbox.describe()))
        hist = [{"step": e.step, "tool": e.tool, "args": e.args, "ok": e.ok, "result": e.summary, "error": e.error}
                for e in evidence[-30:]]
        raw = self._ask(json.dumps({"mission": mission, "history": hist}, default=str), system)
        d = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        if d.get("done"):
            return None
        return Step(str(d["tool"]), dict(d.get("args") or {}), str(d.get("why", "")))


def act(mission: str, toolbox: Toolbox, planner: StepPlanner, state: dict[str, Any] | None = None, *,
        max_steps: int = 200, max_refusals: int = 5, log: Callable[[str], None] | None = None) -> Run:
    """Decide-then-act loop: ask ``planner`` for one step, check it against the toolbox, run it, record the
    evidence, repeat until the planner is done, ``max_steps`` or ``max_refusals`` consecutive refused / failed
    steps. Same :class:`Tool` / :class:`Evidence` contract as :func:`execute`; the returned plan lists the steps
    taken."""
    state = {} if state is None else state
    ev: list[Evidence] = []
    taken: list[Step] = []
    plan = Plan(mission, taken, getattr(planner, "name", "step"))
    state["_plan"], state["_evidence"] = plan, ev
    bad = 0
    for i in range(max_steps):
        t0 = time.perf_counter()
        try:
            s = planner.next(mission, toolbox, state, ev)
        except Exception as e:  # noqa: BLE001 - an unreadable answer is a refused step
            ev.append(Evidence(i, "?", {}, "", {}, round(time.perf_counter() - t0, 3), False,
                               f"planner answer not usable: {type(e).__name__}: {e}"))
            bad += 1
            if bad >= max_refusals:
                plan.notes.append(f"stopped after {bad} unusable planner answers")
                break
            continue
        if s is None:
            break
        taken.append(s)
        err = toolbox.check(s)
        if err:
            ev.append(Evidence(i, s.tool, s.args, s.why, {}, 0.0, False, f"refused: {err}"))
        else:
            tool = toolbox.tools[s.tool]
            try:
                out = tool.fn(state, **s.args) or {}
                state[s.tool] = out
                ev.append(Evidence(i, s.tool, s.args, s.why, (tool.summary or _scalars)(out),
                                   round(time.perf_counter() - t0, 3), True))
            except Exception as e:  # noqa: BLE001 - recorded as evidence, the planner sees it
                ev.append(Evidence(i, s.tool, s.args, s.why, {}, round(time.perf_counter() - t0, 3), False,
                                   f"{type(e).__name__}: {e}"))
        e = ev[-1]
        if log:
            log(f"[{i + 1}] {e.tool} {'ok' if e.ok else 'FAILED'} in {e.seconds:.1f} s" + (f": {e.error}" if e.error else ""))
        bad = 0 if e.ok else bad + 1
        if bad >= max_refusals:
            plan.notes.append(f"stopped after {bad} refused or failed steps in a row")
            break
    return Run(plan, state, ev)
