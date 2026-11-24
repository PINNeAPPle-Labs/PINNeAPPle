"""``DesignPipeline``: the problem, the budget, the loop, the validated design and the report, in one call (#33).

The three pieces existed apart. :class:`~.spec.DesignStudy` is the problem written as data (what is varied, who
evaluates it, what is asked, how much may be spent); :func:`~.study.run_study` is the loop that spends the budget,
and with a :class:`~.loop.StopRule` it spends it in rounds and stops on a number; :meth:`~.study.StudyRun.to_markdown`
is the page a reviewer reads. What was still manual is the step between them: deciding **which** of the designs the
run measured is the answer, on what rule, and writing the two files a project keeps.

That is what this module does, and it does it once, in one place, so the decision is a field a test can read instead
of a paragraph in a notebook:

* :meth:`DesignPipeline.from_problem` takes the user's sentence and the parts it names and builds the study, which is
  the "from the problem" half of the issue — the statement stays in the spec as written, so the report can print the
  question next to the answer;
* :meth:`DesignPipeline.run` runs the study (in rounds when a stop rule is given), selects the final design, and
  writes ``report.md`` and ``study.json`` into an output folder;
* :class:`PipelineRun` is the result: the whole :class:`~.study.StudyRun`, the answer
  (:attr:`PipelineRun.answer`: the design, its outputs, the rule that chose it and how many candidates it beat), the
  decision and where the two files went.

The selection rule, and why it is the one written down
------------------------------------------------------
The answer is the **best measured design on the Pareto front**: measured, because a design the reference never ran is
a prediction and a project cannot be built on one; on the front, because a design that violates a limit is not a
design anyone asked for; and best in the **primary objective** (the first one declared), because a single number has
to be named and an average of objectives would be a weighting nobody asked for. The other objectives of the chosen
design are printed beside it, and the count of candidates it beat is in the record: a rule that picked without saying
what it picked from is an opinion.

When the run measured nothing (a budget too small, a solver that refused every design, a study whose limits nothing
met) the pipeline answers ``None`` and says why, instead of handing back the surrogate's recommendation with a
footnote. That is the point of the whole exercise: the answer is a design the reference solver has confirmed.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .loop import StopRule
from .spec import Base, Budget, DesignStudy, Evaluator, Limit, Model, Objective, Verifier
from .study import StudyRun, run_study

__all__ = ["DesignPipeline", "PipelineRun", "select"]

#: The rule the answer is chosen by, in the words the report prints.
SELECTION_RULE = ("best measured design on the Pareto front, by the primary objective (the first one declared); "
                  "only designs the reference evaluator ran, and only designs that satisfy every limit")


@dataclass
class PipelineRun:
    """A pipeline's answer: the study it ran, the design it selected, the decision, and the two files it wrote.

    ``answer`` is ``None`` when no measured design could be selected, and ``answer["why"]`` says which of the three
    conditions failed. ``report_path``/``study_path`` are the files written, empty when the run was asked not to write
    them (``out_dir=None``), which is what a test does when it only wants the numbers.
    """

    study: StudyRun
    answer: dict[str, Any] | None
    report_path: str = ""
    study_path: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def decision(self) -> str:
        """The trust verdict of the study (``APPROVED`` / ``REVIEW`` / ``REJECT``)."""
        return self.study.decision

    @property
    def approved(self) -> bool:
        """True when the study is ``done``, its verdict is ``APPROVED`` and there is a validated design."""
        return self.study.approved and bool(self.answer and self.answer.get("verified"))

    def to_dict(self) -> dict[str, Any]:
        return {"answer": dict(self.answer) if self.answer else None, "decision": self.decision,
                "approved": self.approved, "created": self.created, "report_path": self.report_path,
                "study_path": self.study_path, "study": self.study.to_dict()}

    def summary(self) -> dict[str, Any]:
        a = self.answer or {}
        return {"decision": self.decision, "approved": self.approved,
                "design": a.get("design"), "objective": a.get("value"),
                "verified": bool(a.get("verified")), "stop": self.study.stop.get("reason", ""),
                "evaluations": self.study.budget.get("used", 0), "report": self.report_path}


def select(run: StudyRun) -> dict[str, Any] | None:
    """The final design of a run, or ``None`` with the reason no design could be selected.

    The rule is :data:`SELECTION_RULE`; every number it used is in the returned dict, so a reader can redo the choice.
    """
    spec = run.spec
    primary = spec.objectives[0]
    front = [d for d in run.pareto if d.get("params")]
    measured = [d for d in front if d.get("verified") and d.get("predicted") is None]
    if not measured:
        return None
    feasible = [d for d in measured if not d.get("violation")]
    pool = feasible or measured
    best = min(pool, key=lambda d: float(d["objectives"][primary.output])) if primary.sense == "min" \
        else max(pool, key=lambda d: float(d["objectives"][primary.output]))
    return {"design": dict(best["params"]), "outputs": dict(best["outputs"]),
            "objectives": {o.output: float(best["objectives"][o.output]) for o in spec.objectives},
            "primary": primary.output, "value": float(best["objectives"][primary.output]),
            "sense": primary.sense, "verified": True, "rule": SELECTION_RULE,
            "n_candidates": len(pool), "n_measured_on_front": len(measured),
            "units": {k: spec.evaluator.unit_of(k) for k in best["outputs"] if k in spec.evaluator.output_names},
            "evaluator": {"tool": spec.evaluator.tool, "fidelity": spec.evaluator.fidelity}}


def _no_answer(run: StudyRun) -> dict[str, Any]:
    measured = [d for d in run.pareto if d.get("verified")]
    if run.status == "refused":
        why = f"the study was refused: {run.reasons[0] if run.reasons else 'no reason given'}"
    elif not measured:
        why = ("no design on the front was re-measured by the reference inside the budget: the answer would be a "
               "prediction, and a project cannot be built on one")
    else:
        why = "the measured designs on the front violate a limit"
    return {"design": None, "verified": False, "rule": SELECTION_RULE, "why": why}


class DesignPipeline:
    """One problem, one budget, one answer, one report.

    ``out_dir`` is where ``report.md`` and ``study.json`` are written; ``None`` runs the pipeline without writing
    (the numbers are still in the returned :class:`PipelineRun`). ``stop`` is the loop's rule
    (:class:`.loop.StopRule`): without it the study runs one cycle, with it the study spends its budget in rounds
    until the rule's number says it has bought all it can.
    """

    def __init__(self, spec: DesignStudy | Mapping, evaluator: Any, *, stop: StopRule | Mapping | None = None,
                 out_dir: str | os.PathLike | None = None, progress: Callable[[dict], None] | None = None):
        self.spec = spec if isinstance(spec, DesignStudy) else DesignStudy.from_dict(spec)
        self.evaluator = evaluator
        self.stop = stop if stop is None or isinstance(stop, StopRule) else StopRule.from_dict(dict(stop))
        self.out_dir = None if out_dir is None else os.fspath(out_dir)
        self.progress = progress

    # ------------------------------------------------------------------ from the problem
    @classmethod
    def from_problem(cls, statement: str, *, base: Base | Mapping, evaluator: Evaluator | Mapping,
                     objectives: Sequence[Objective | Mapping], kind: str = "optimize",
                     constraints: Sequence[Limit | Mapping] = (), budget: Budget | Mapping | None = None,
                     verifier: Verifier | Mapping | None = None, model: Model | Mapping | None = None,
                     title: str = "", metadata: Mapping[str, Any] | None = None,
                     stop: StopRule | Mapping | None = None, out_dir: str | os.PathLike | None = None,
                     progress: Callable[[dict], None] | None = None) -> DesignPipeline:
        """Build the study from the user's sentence and the parts it names, then run it as a pipeline.

        The statement is kept in the spec as it was written: it is the question, and the report prints it above the
        answer. Everything else is the schema's own input, validated by it — this method only decides the title when
        the caller does not give one (the statement's first clause), so a study is never saved nameless.
        """
        text = " ".join(str(statement).split())
        if not text:
            raise ValueError("a pipeline starts from the problem: the statement is what the report prints above the "
                             "answer, so it cannot be empty")
        name = str(title).strip() or (text.split(".")[0][:120] or text[:120])
        spec = DesignStudy.create(kind=kind, title=name, statement=text, base=base, evaluator=evaluator,
                                  objectives=objectives, constraints=constraints, budget=budget, verifier=verifier,
                                  model=model, metadata=dict(metadata or {}))
        return cls(spec, None, stop=stop, out_dir=out_dir, progress=progress)

    # ------------------------------------------------------------------ the run
    def run(self, evaluator: Any = None) -> PipelineRun:
        """Run the study, select the answer and write the report. ``evaluator`` overrides the one given to the
        constructor (a pipeline built by :meth:`from_problem` has none: the evaluator is the caller's tool)."""
        fn = evaluator if evaluator is not None else self.evaluator
        if fn is None:
            raise ValueError("a pipeline needs the evaluator: the tool, case or solver the study runs")
        study = run_study(self.spec, fn, stop=self.stop, progress=self.progress)
        answer = select(study) or _no_answer(study)
        report_path = study_path = ""
        if self.out_dir is not None:
            report_path, study_path = self._write(PipelineRun(study, answer))
        return PipelineRun(study=study, answer=answer, report_path=report_path, study_path=study_path)

    # ------------------------------------------------------------------ the two files
    def _write(self, result: PipelineRun) -> tuple[str, str]:
        os.makedirs(os.fspath(self.out_dir), exist_ok=True)
        report = os.path.join(os.fspath(self.out_dir), "report.md")
        study = os.path.join(os.fspath(self.out_dir), "study.json")
        with open(report, "w", encoding="utf-8") as f:
            f.write(self.report(result))
        with open(study, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        return report, study

    def report(self, result: PipelineRun) -> str:
        """The page a reviewer reads: the question, the answer, how it was chosen, and the study as its appendix."""
        spec, run, a = self.spec, result.study, result.answer or {}
        units = spec.units()
        f = lambda v: "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))   # noqa: E731
        out = [f"# Design pipeline: {spec.title}", "", f"**The problem.** {spec.statement or spec.title}", "",
               f"**Decision.** {result.decision or 'not run'}"
               + (" — the reference confirmed the design below" if result.approved else
                  (" — see why below" if result.answer else " — no design was validated")), "",
               f"**Stopped on.** {run.stop.get('reason', '—')}"
               + (f" at round {run.stop['round']}" if run.stop.get("round") else "")
               + f"; {run.budget.get('used', 0)} of {spec.budget.max_evaluations} reference evaluations "
                 f"({run.wall_s:.1f} s wall).", ""]
        if a.get("design"):
            out += ["## The final design", "", "| variable | value | unit |", "|---|---|---|"]
            variables = {v.name: v.unit for v in spec.base.variables}
            for k, v in a["design"].items():
                out.append(f"| {k} | {f(float(v))} | {variables.get(k, '')} |")
            out += ["", "| objective | value | unit |", "|---|---|---|"]
            for o in spec.objectives:
                out.append(f"| {o.label or o.output} ({o.sense}) | {f(a['objectives'][o.output])} | "
                           f"{units.get(o.output, '')} |")
            out += ["", f"Confirmed by `{a['evaluator']['tool']}` ({a['evaluator']['fidelity']}): "
                        + ", ".join(f"{k} = {f(v)} {units.get(k, '')}" for k, v in a["outputs"].items()) + ".", ""]
        else:
            out += ["## No validated design", "", f"{a.get('why', 'no design could be selected')}.", ""]
        out += ["## How it was chosen", "", f"- Rule: {a.get('rule', SELECTION_RULE)}.",
                f"- Candidates: {a.get('n_candidates', 0)} measured design(s)"
                + (f" of {a['n_measured_on_front']} measured members of the front" if a.get("design") else "") + ".", ""]
        if result.study.stop:
            out.append(f"- Loop: {result.study.stop.get('text', '')}.")
        if result.report_path:
            out.append(f"- Written: `{os.path.basename(result.report_path)}` and "
                       f"`{os.path.basename(result.study_path)}`.")
        body = run.to_markdown()
        out += ["", "---", "", body.split("\n", 1)[1].lstrip("\n") if body.startswith("# ") else body]
        return "\n".join(out) + "\n"
