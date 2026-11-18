"""The contract a pipeline suggestion is written in: a brief (what the study is), the stages it needs (what to do),
and the decisions behind them (why, with the numbers that decided).

One plan answers ten questions in a fixed order, and every answer names the PINNeAPPle call that carries it out:

1. is the physical model right (equations, initial and boundary conditions) -- and if not, which one is;
2. does the problem need a 3-D geometry at all;
3. (3-D) a mesh is needed -- unless the domain is analytic and the solver builds its own grid;
4. the geometry and the mesh are imported when the user has them, generated when they do not;
5. a training dataset is imported when the user has it, generated when it does not;
6. generating it needs the solver for this physics, with the time and the cost per simulation;
7. cheap solves -> sweep the solver; expensive ones -> train a surrogate;
8. how much data there is decides between a PINN, a DeepONet, an FNO, a graph network or a PINO;
9. how the dataset is generated (design of experiments, coverage, split);
10. what to look at and what to check at every stage.

Everything here is data: the plan is JSON first (`StudyPlan.to_dict`), and the diagram is a rendering of the same
object (`pinneapple_advisor.render`), so a web app shows the plan without a second source of truth. A blind spot is
recorded as an open question (`pinneapple_advisor.questions`), never as a silent default.

Honesty: this is a recommendation, not a measurement. Every estimate carries its formula and its basis
(`Estimate.basis`), the architecture recommendation is PINNeAPPle's own documented-scope match
(``pinneapple_analysis.verification.architecture_recommendation``), and nothing here runs the pipeline it suggests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "pinneapple.advisor/1"

# What a stage can be, in the order the ten questions ask it.
STAGE_KINDS = ("physics", "geometry", "mesh", "solver", "dataset", "training", "inverse", "optimization",
               "verification", "visualization")

# How a stage stands for this user: already done, to be built, to be confirmed, optional, or not needed.
STATUSES = ("have", "generate", "confirm", "optional", "skip")

# A bilingual string: {"pt": ..., "en": ...}. `text_of` reads one, mirroring pinneapple_reason.
Text = dict[str, str]


def text_of(value: Any, lang: str = "en") -> str:
    """Read a bilingual text (or a plain string) in ``lang``, falling back to the other language."""
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or value.get("pt") or "")
    return "" if value is None else str(value)


def bi(pt: str, en: str) -> Text:
    return {"pt": pt, "en": en}


@dataclass
class Estimate:
    """A cost estimate and, just as important, what it was computed from.

    ``confidence`` is ``"order of magnitude"`` for anything produced by the default cost model: it is a formula with
    published-order coefficients, not a measurement on the user's machine. A measured ``solve_seconds`` passed in by
    the caller replaces the formula and sets ``confidence="measured"``.
    """
    seconds_per_evaluation: float
    n_evaluations: int
    total_seconds: float
    cost_usd: float | None = None
    cells: int | None = None
    basis: str = ""
    confidence: str = "order of magnitude"
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"seconds_per_evaluation": round(self.seconds_per_evaluation, 3), "n_evaluations": self.n_evaluations,
                "total_seconds": round(self.total_seconds, 1), "total_hours": round(self.total_seconds / 3600, 3),
                "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 2), "cells": self.cells,
                "basis": self.basis, "confidence": self.confidence, "assumptions": list(self.assumptions)}


@dataclass
class Tool:
    """One PINNeAPPle call a stage uses, named so both a person and a script can follow it."""
    call: str
    note: Text = field(default_factory=lambda: bi("", ""))
    docs: str = ""            # a path in this repository or a URL

    def to_dict(self) -> dict[str, Any]:
        return {"call": self.call, "note": self.note, "docs": self.docs}


@dataclass
class Stage:
    """One step of the pipeline: what, why, with which PINNeAPPle calls, at what estimated cost, checked how."""
    id: str
    kind: str
    title: Text
    why: Text
    status: str = "generate"
    tools: list[Tool] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    estimate: Estimate | None = None
    checks: list[dict[str, str]] = field(default_factory=list)   # [{"what", "tool"}]
    views: list[dict[str, str]] = field(default_factory=list)    # [{"what", "tool"}]
    question: str = ""        # the open question whose answer would change this stage ("" = none)
    # Which of the ten questions this stage answers (1-based), for the report and the diagram legend.
    answers: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in STAGE_KINDS:
            raise ValueError(f"stage kind {self.kind!r} is not one of {STAGE_KINDS}")
        if self.status not in STATUSES:
            raise ValueError(f"stage status {self.status!r} is not one of {STATUSES}")
        # ``views`` reads as [{"what", "tool"}]; a :class:`Tool` in the list is the same thing, spelled by the caller.
        self.views = [{"what": v.note, "tool": v.call} if isinstance(v, Tool) else dict(v) for v in self.views]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "why": self.why, "status": self.status,
                "answers": list(self.answers),
                "tools": [t.to_dict() for t in self.tools], "inputs": list(self.inputs),
                "outputs": list(self.outputs), "depends_on": list(self.depends_on),
                "estimate": None if self.estimate is None else self.estimate.to_dict(),
                "checks": list(self.checks), "views": list(self.views), "question": self.question}


@dataclass
class Decision:
    """A choice the plan makes, the numbers that made it, and what the alternatives were."""
    topic: str
    choice: str
    why: Text
    alternatives: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    source: str = ""          # the API that computed it, e.g. "architecture_recommendation.recommend_architecture"

    def to_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "choice": self.choice, "why": self.why, "alternatives": self.alternatives,
                "values": self.values, "source": self.source}


@dataclass
class StudyPlan:
    """The suggestion: stages, the edges between them, the decisions, the open questions and the diagram."""
    problem: Text
    stages: list[Stage] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[Text] = field(default_factory=list)
    brief: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    # ---- reading it ---------------------------------------------------------------------------------------------
    def stage(self, stage_id: str) -> Stage:
        for s in self.stages:
            if s.id == stage_id:
                return s
        raise KeyError(stage_id)

    def needed(self) -> list[Stage]:
        """The stages that cost work: everything but ``skip`` (and ``optional`` stays optional)."""
        return [s for s in self.stages if s.status in ("generate", "confirm")]

    def total_estimate(self) -> Estimate:
        """The campaign cost: every ``generate``/``confirm`` stage's estimate added up, with the basis kept."""
        secs = sum((s.estimate.total_seconds if s.estimate else 0.0) for s in self.needed())
        usd = [s.estimate.cost_usd for s in self.stages if s.estimate and s.estimate.cost_usd is not None]
        cells = max([s.estimate.cells for s in self.stages if s.estimate and s.estimate.cells] or [0])
        n = max([s.estimate.n_evaluations for s in self.stages if s.estimate] or [1])
        return Estimate(seconds_per_evaluation=secs / max(n, 1), n_evaluations=n, total_seconds=secs,
                        cost_usd=(sum(usd) if usd else None), cells=cells or None,
                        basis="sum of the estimates of the stages that still cost work",
                        confidence=("measured" if all(s.estimate.confidence == "measured" for s in self.needed()
                                                       if s.estimate) and self.needed() else "order of magnitude"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "problem": self.problem, "brief": self.brief,
                "stages": [s.to_dict() for s in self.stages],
                "edges": [{"from": s.id, "to": d, "kind": "depends"} for s in self.stages for d in s.depends_on],
                "decisions": [d.to_dict() for d in self.decisions],
                "open_questions": list(self.open_questions),
                "warnings": list(self.warnings),
                "estimate": self.total_estimate().to_dict(),
                "provenance": dict(self.provenance)}

    def json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def diagram(self, kind: str = "mermaid", lang: str | None = None) -> str:
        """``mermaid``, ``dot``, ``svg``, ``html`` or ``markdown`` -- all from this same plan. ``lang`` overrides the
        language the plan was composed in (``pt`` or ``en``)."""
        from . import render
        return render.diagram(self, kind, lang)

    def write(self, folder: str, *, diagrams: tuple = ("mermaid", "svg", "html", "markdown")) -> list[str]:
        """Write ``plan.json`` and each diagram into ``folder``; returns the paths written."""
        import os

        os.makedirs(folder, exist_ok=True)
        written = [os.path.join(folder, "plan.json")]
        with open(written[0], "w", encoding="utf-8") as fh:
            fh.write(self.json())
        for kind in diagrams:
            ext = {"markdown": "md"}.get(kind, kind)
            path = os.path.join(folder, f"pipeline.{ext}")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.diagram(kind))
            written.append(path)
        return written


__all__ = ["SCHEMA", "STAGE_KINDS", "STATUSES", "Text", "text_of", "bi", "Estimate", "Tool", "Stage", "Decision",
           "StudyPlan"]
