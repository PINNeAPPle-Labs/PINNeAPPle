"""Generic Physics AI studies, creatable without code: the schema ``pinneapple-design-study/v1`` and its runner.

A study is described once (:class:`.spec.DesignStudy`) and run once (:func:`.study.run_study`), with the evaluator —
a catalog tool, a solver case, an analytic model — injected by the caller:

    from pinneapple_analysis.design_study import run_study
    from pinneapple_analysis.design_study.examples import wing_study, wing_vlm

    run = run_study(wing_study(), wing_vlm())        # baseline -> DOE -> surrogate -> NSGA-II -> verify
    run.status, run.trust["decision"]                # ('done', 'APPROVED')
    run.pareto[0]["params"], run.rejected             # the answer, and what was refused with the reason

What is here:

* :class:`.spec.DesignStudy` — kind (optimise, surrogate, PINN/operator, inverse, twin), base geometry or system,
  evaluator with its input mapping and outputs **with units**, objectives, limits absolute or relative to the
  baseline, budget, verifier. ``to_dict`` / ``from_dict`` are strict, and an output without a unit is refused;
* :func:`.study.run_study` — the cycle and its answer (:class:`.study.StudyRun`): Pareto front with each member
  flagged measured or predicted, rejected designs with the reason, the trust verdict and the validated domain.
  ``optimize`` and ``surrogate`` run today, and ``pinn`` (:mod:`.pinn`) fits the fields a study measured with a
  physics-informed network; ``inverse`` and ``twin`` are described and refused with what is missing.
  With a :class:`.loop.StopRule` the cycle runs in rounds, refitting on everything measured each time, and stops on
  the first number that says another round buys nothing (:attr:`.study.StudyRun.loop` says which);
* :mod:`.loop` — that rule and its decision (:func:`.loop.decide`), as data: four reasons, each decided on a number;
* :mod:`.pinn` — the per-sample fields a parametric study fits, the collocation of the physics loss on the mesh the
  study measured, and the error of the fit measured against the reference fields (#136);
* :mod:`.examples` — five studies with the evaluator the tests run: a wing and a heatsink on closed-form models, a
  cantilever beam on a numerical solver, a polar (surrogate) and a cooling slab (pinn, with fields).

The schema is the contract the Studio wizard and the Engineer fill (#1036); this package is the library side of it.
"""
from __future__ import annotations

from . import examples
from .loop import StopRule, decide
from .spec import (
    BASE_KINDS,
    FIDELITIES,
    KINDS,
    KINDS_BEYOND_OPTIMISATION,
    SCHEMA,
    SURROGATE_FAMILIES,
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Limit,
    Model,
    Objective,
    Quantity,
    Variable,
    Verifier,
)
from .study import RUNNABLE_KINDS, StudyRefused, StudyRun, run_study

__all__ = [
    "SCHEMA", "KINDS", "KINDS_BEYOND_OPTIMISATION", "BASE_KINDS", "FIDELITIES", "SURROGATE_FAMILIES",
    "RUNNABLE_KINDS", "Quantity", "Variable", "Base", "Evaluator", "Objective", "Limit", "Budget", "Verifier",
    "Model", "DesignStudy", "StudyRun", "StudyRefused", "run_study", "StopRule", "decide", "examples",
]
