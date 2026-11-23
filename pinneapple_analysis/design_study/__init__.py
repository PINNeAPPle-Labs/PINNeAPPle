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
  ``optimize`` and ``surrogate`` run today; the other kinds are described and refused with what is missing;
* :mod:`.examples` — two optimisation studies (a wing on a lifting-line model, a forced-convection heatsink) and a
  surrogate study, each with the analytic evaluator the tests run.

The schema is the contract the Studio wizard and the Engineer fill (#1036); this package is the library side of it.
"""
from __future__ import annotations

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
    "Model", "DesignStudy", "StudyRun", "StudyRefused", "run_study",
]
