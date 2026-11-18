"""PINNeAPPle pipeline advisor: a problem in words (or in answers) in, a study pipeline out -- as a process diagram
and as JSON, with the PINNeAPPle call that carries out every step.

    import pinneapple as pp

    plan = pp.advise("transient heat in a 3-D bracket, I have the STEP but no mesh")
    print(plan.diagram("mermaid"))
    plan.write("advice")                     # plan.json, pipeline.mermaid, pipeline.svg, pipeline.html

What it answers, in this order: is the physical model right; is it 3-D; a mesh; do you already have the geometry, the
mesh and the dataset; which solver and at what cost per run; sweep the solver or train a surrogate; which
architecture; how to generate the dataset; what to check and what to look at. Each of the ten is a stage of the plan,
with its tools, its estimate and its checks (see :mod:`pinneapple_advisor.compose`).

Two ways in, and they compose:

* free text: :func:`pinneapple_advisor.extract.detect` reads the facts the sentence states, and a local model
  (``ollama serve``; nothing leaves the machine) fills what is still open, validated field by field;
* the predefined questions: :func:`questions` lists them and :func:`next_questions` returns only what the sentence
  did not answer, so a web app can ask them one at a time.

``advise(..., llm=False)`` works with no model at all: the rules read the sentence, the questions fill the rest.

Nothing here runs the pipeline it suggests, and the plan says that itself. The physics, the solver family and the
architecture come from PINNeAPPle's own recommendation APIs, not from the model, so the same brief always produces
the same plan.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .brief import FIELDS, Brief, BriefError
from .compose import Composition, build
from .estimate import CostModel
from .extract import answer as _answer
from .extract import detect, extract
from .questions import QUESTIONS
from .questions import next_questions as _next_questions
from .questions import question as question
from .questions import questions as _questions
from .schema import SCHEMA, StudyPlan

__version__ = "1.0.0"


def brief(problem: str, *, answers: dict[str, Any] | None = None, llm: bool | Callable | None = None,
          provider: str = "ollama", model: str = "", **llm_kwargs: Any) -> Brief:
    """The facts of a study: the sentence read by the rules, the model for what is left, then the answers."""
    call: bool | Callable | None = False if llm is False else (llm if callable(llm) else None)
    b = extract(problem, call=call, provider=provider, model=model, **llm_kwargs)
    if answers:
        b.merge(answers, source="answer")
    return b


def advise(problem: str | Brief, *, answers: dict[str, Any] | None = None,
           llm: bool | Callable | None = None, provider: str = "ollama", model: str = "",
           cost: CostModel | None = None, language: str = "pt", out_dir: str | None = None,
           available_families: dict[str, Any] | None = None, solve_seconds_measured: float | None = None,
           **llm_kwargs: Any) -> StudyPlan:
    """The study pipeline for a problem description (or a :class:`~pinneapple_advisor.brief.Brief`), ten questions
    answered in order.

    ``answers`` is the web app's path: ``{"dimension": 3, "geometry_present": True, "goal": "inverse"}`` fills the
    brief directly (validated; a bad value raises :class:`~pinneapple_advisor.brief.BriefError`). ``llm`` is
    ``False`` (no model), a callable (a test or an alternative provider) or ``None`` (the local Ollama server, whose
    failure only costs the fields the rules could not read). ``cost`` replaces the cost model, and
    ``solve_seconds_measured`` replaces its formula with a real number.
    """
    b = problem if isinstance(problem, Brief) else brief(problem, answers=answers, llm=llm, provider=provider,
                                                         model=model, **llm_kwargs)
    if isinstance(problem, Brief) and answers:
        b.merge(answers, source="answer")
    composition: Composition = build(b, cost=cost, available_families=available_families,
                                     solve_seconds_measured=solve_seconds_measured, language=language)
    plan = composition.plan
    if out_dir:
        plan.write(out_dir)
    return plan


def advise_detailed(problem: str | Brief, *, answers: dict[str, Any] | None = None,
                    llm: bool | Callable | None = None, provider: str = "ollama", model: str = "",
                    cost: CostModel | None = None, language: str = "pt",
                    available_families: dict[str, Any] | None = None,
                    solve_seconds_measured: float | None = None, **llm_kwargs: Any) -> Composition:
    """``advise`` without the wrapper: the plan **and** the raw solver/architecture recommendations it came from."""
    b = problem if isinstance(problem, Brief) else brief(problem, answers=answers, llm=llm, provider=provider,
                                                         model=model, **llm_kwargs)
    if isinstance(problem, Brief) and answers:
        b.merge(answers, source="answer")
    return build(b, cost=cost, available_families=available_families,
                 solve_seconds_measured=solve_seconds_measured, language=language)


# ---- the questions ----------------------------------------------------------------------------------------------

def questions() -> list[dict[str, Any]]:
    """Every predefined question (id, field, kind, text in Portuguese and English, the choices)."""
    return _questions()


def next_questions(b: Brief, limit: int | None = None) -> list[dict[str, Any]]:
    """What is still open for this brief, in asking order."""
    return _next_questions(b, limit)


def answer(question_id: str, raw: Any, *, llm: bool | Callable | None = None, provider: str = "ollama",
           model: str = "") -> Any:
    """One answer, read against its question's closed set of values (a model maps it when the words do not match)."""
    call = False if llm is False else (llm if callable(llm) else None)
    return _answer(question_id, raw, call=call, provider=provider, model=model)


__all__ = ["advise", "advise_detailed", "brief", "questions", "next_questions", "question", "answer", "detect",
           "extract", "build", "Brief", "BriefError", "FIELDS", "CostModel", "StudyPlan", "SCHEMA", "QUESTIONS",
           "__version__"]
